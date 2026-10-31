"""
Perplexity evaluation for CMP-Q quantized models.

Standard methodology matching GPTQ/AWQ papers:
- WikiText-2: concatenate all test text, tokenize once, chunk into seq_len windows, evaluate
- C4: sample 256 docs with ≥seq_len tokens, take random window from each, evaluate

References:
  - https://github.com/IST-DASLab/gptq/blob/main/datautils.py
  - https://github.com/Efficient-ML/Qwen3-Quantization/blob/main/eval_my/evaluate_.py
"""
import random
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_wikitext2_testenc(tokenizer: AutoTokenizer) -> torch.Tensor:
    """Load WikiText-2 test set as a single tokenized sequence.

    Standard approach: concatenate all test text with '\\n\\n', tokenize once.
    Returns input_ids tensor of shape [1, total_tokens].
    """
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return testenc.input_ids


def _get_c4_testenc(tokenizer: AutoTokenizer, n_samples: int = 256, seed: int = 0, seqlen: int = 2048) -> torch.Tensor:
    """Load C4 validation set: sample n_samples docs with ≥seqlen tokens, take random window.

    Returns input_ids tensor of shape [1, n_samples * seqlen].
    """
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    random.seed(seed)
    valenc = []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    return valenc


@torch.no_grad()
def eval_ppl(
    model: AutoModelForCausalLM,
    testenc: torch.Tensor,
    seq_len: int = 2048,
    device: torch.device | str | None = None,
) -> tuple[float, int]:
    """Evaluate perplexity on a pre-tokenized sequence using non-overlapping chunks.

    Standard GPTQ evaluation loop: split into seq_len chunks, forward each,
    compute cross-entropy loss, average, exponentiate.

    Args:
        model: Loaded causal LM
        testenc: Token ids of shape [1, total_tokens]
        seq_len: Chunk size (default: 2048)
        device: Device to run on (default: model.device)

    Returns:
        Tuple of (perplexity, total_tokens_evaluated)
    """
    if device is None:
        device = model.device

    nsamples = testenc.numel() // seq_len
    model.eval()

    nlls = []
    for i in tqdm(range(nsamples), desc="Evaluating PPL"):
        batch = testenc[:, (i * seq_len):((i + 1) * seq_len)].to(device)
        outputs = model(batch, use_cache=False)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = testenc[:, (i * seq_len):((i + 1) * seq_len)][:, 1:].to(device)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seq_len
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))
    total_tokens = nsamples * seq_len
    return ppl.item(), total_tokens


def evaluate_wikitext2(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    seq_len: int = 2048,
) -> tuple[float, int]:
    """Evaluate WikiText-2 perplexity using standard GPTQ methodology."""
    print(f"Loading WikiText-2 test set...")
    testenc = _get_wikitext2_testenc(tokenizer)
    nsamples = testenc.numel() // seq_len
    print(f"  Total tokens: {testenc.numel():,}, chunks: {nsamples} x {seq_len}")
    return eval_ppl(model, testenc, seq_len=seq_len)


def evaluate_c4(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    n_samples: int = 256,
    seed: int = 0,
    seq_len: int = 2048,
) -> tuple[float, int]:
    """Evaluate C4 perplexity using standard GPTQ methodology."""
    print(f"Loading C4 validation ({n_samples} samples, seed={seed})...")
    testenc = _get_c4_testenc(tokenizer, n_samples=n_samples, seed=seed, seqlen=seq_len)
    nsamples = testenc.numel() // seq_len
    print(f"  Total tokens: {testenc.numel():,}, chunks: {nsamples} x {seq_len}")
    return eval_ppl(model, testenc, seq_len=seq_len)


def load_model_and_tokenizer(
    model_path: str,
    device_map: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer for evaluation."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


