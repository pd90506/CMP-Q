import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import logging
from ..core.config import CompressorConfig

logger = logging.getLogger(__name__)


def load_model(config: CompressorConfig):
    """Load model and tokenizer from HuggingFace."""
    model_name = config.model.name
    hf_token = config.model.hf_token

    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(
        model_name,
        attn_implementation="eager",
        token=hf_token,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        # No device_map: keep model on CPU, pipeline handles GPU offloading per-layer (GPTQ-style)
    )

    model.eval()
    model.seqlen = config.model.seq_len

    return model, tokenizer


def _get_wikitext2(nsamples: int, seed: int, seqlen: int, tokenizer) -> list[dict]:
    """Load WikiText-2 calibration data following the standard GPTQ approach.

    Concatenates all training texts, tokenizes as one long sequence, then
    randomly samples `nsamples` windows of `seqlen` tokens.
    """
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    random.seed(seed)
    samples = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        samples.append({"input_ids": inp})
    return samples


def _get_c4(nsamples: int, seed: int, seqlen: int, tokenizer) -> list[dict]:
    """Load C4 calibration data following the standard GPTQ approach.

    Loads the first shard of C4, randomly picks documents with enough tokens,
    then randomly picks a window of `seqlen` tokens within each document.
    """
    traindata = load_dataset(
        "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    random.seed(seed)
    samples = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        samples.append({"input_ids": inp})
    return samples


def _get_hellaswag(nsamples: int, seed: int, seqlen: int, tokenizer) -> list[dict]:
    """Load HellaSwag train calibration data.

    Concatenates context + correct ending for each example, joins with "\\n\\n",
    tokenizes as one long sequence, then randomly samples `nsamples` windows of
    `seqlen` tokens (mirrors wikitext2 approach).

    HellaSwag train has 39905 examples (Rowan/hellaswag). Avg ctx+ending length
    is ~80-150 tokens, so concatenated corpus is large enough for any
    nsamples × seqlen budget.
    """
    traindata = load_dataset("Rowan/hellaswag", split="train")
    # Build full text: "{activity_label}: {ctx} {correct_ending}"
    texts = []
    for ex in traindata:
        try:
            label = int(ex["label"])
        except (ValueError, TypeError):
            continue
        ending = ex["endings"][label]
        ctx = ex["ctx"]
        act = ex["activity_label"]
        texts.append(f"{act}: {ctx} {ending}")
    trainenc = tokenizer("\n\n".join(texts), return_tensors="pt")

    random.seed(seed)
    samples = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        samples.append({"input_ids": inp})
    return samples


def _tokenizer_adds_bos(tokenizer) -> bool:
    """True iff the tokenizer prepends BOS to plain text by default (Llama-2, Mistral, ...)."""
    bos_id = tokenizer.bos_token_id
    return bos_id is not None and tokenizer("hello").input_ids[0] == bos_id


def _replace_first_with_bos(samples: list[dict], tokenizer) -> list[dict]:
    """Overwrite token 0 of every calibration window with BOS (window length unchanged).

    Random C4/WT2 windows almost never start at a document start, so a model that
    always sees BOS at position 0 at eval time is never calibrated on it. No-op
    for tokenizers that do not add BOS by default (Qwen).
    """
    if not _tokenizer_adds_bos(tokenizer):
        logger.info("calibration.add_bos=true: tokenizer does not add BOS, skipping (windows unchanged)")
        return samples
    bos_id = tokenizer.bos_token_id
    n_replaced = 0
    for s in samples:
        ids = s["input_ids"].clone()
        if ids[0, 0].item() != bos_id:
            n_replaced += 1
        ids[0, 0] = bos_id
        s["input_ids"] = ids
    logger.info(f"calibration.add_bos=true: set token 0 of {len(samples)} windows to BOS "
                f"(bos_id={bos_id}; {n_replaced} windows did not already start with BOS)")
    return samples


def load_calibration_samples(
    config: CompressorConfig,
    tokenizer,
    *,
    nsamples: int | None = None,
    seqlen: int | None = None,
) -> list[dict]:
    """Load calibration samples following the standard GPTQ approach.

    For WikiText-2: concatenates all training text, tokenizes, then randomly
    samples fixed-length windows.
    For C4: loads the first shard, randomly picks documents long enough, and
    randomly picks a window within each.

    Dataset downloads are cached by HuggingFace datasets library.

    Args:
        config: CompressorConfig (provides dataset, seed, and defaults).
        tokenizer: HF tokenizer.
        nsamples: Optional override for number of samples.
        seqlen: Optional override for sequence length.

    Returns a list of dicts with 'input_ids' tensor of shape (1, seq_len).
    """
    nsamples = nsamples if nsamples is not None else config.data.n_samples
    seed = config.experiment.seed
    seqlen = seqlen if seqlen is not None else config.model.seq_len
    dataset_name = config.data.dataset_name.lower()

    if "wikitext" in dataset_name:
        logger.info(f"Loading WikiText-2 calibration: {nsamples} samples, seqlen={seqlen}, seed={seed}")
        samples = _get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in dataset_name:
        logger.info(f"Loading C4 calibration: {nsamples} samples, seqlen={seqlen}, seed={seed}")
        samples = _get_c4(nsamples, seed, seqlen, tokenizer)
    elif "hellaswag" in dataset_name:
        logger.info(f"Loading HellaSwag train calibration: {nsamples} samples, seqlen={seqlen}, seed={seed}")
        samples = _get_hellaswag(nsamples, seed, seqlen, tokenizer)
    else:
        raise ValueError(f"Unsupported calibration dataset: {config.data.dataset_name}")

    calibration = getattr(config, "calibration", None)
    if calibration is not None and calibration.add_bos:
        samples = _replace_first_with_bos(samples, tokenizer)

    logger.info(f"Loaded {len(samples)} calibration samples of length {seqlen}")
    return samples


def capture_layer0_inputs(model, samples: list[dict], dev, seq_len: int):
    """Capture the inputs (and forward kwargs) of the first decoder layer (GPTQ-style Catcher).

    The embedding and layer 0 must already be on `dev`. Returns (inps, layer_kwargs),
    inps of shape [n_samples, seq_len, hidden] on `dev`.
    """
    hidden_size = model.config.hidden_size
    dtype = next(model.parameters()).dtype

    inps = torch.zeros(
        (len(samples), seq_len, hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'kwargs': {}}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name: str):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp[0]
            cache['i'] += 1
            cache['kwargs'] = kwargs
            raise ValueError

    layers = model.model.layers
    layers[0] = Catcher(layers[0])

    for sample in samples:
        try:
            model(sample["input_ids"].to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module

    kw = cache['kwargs']
    kw.pop("past_key_value", None)
    kw.pop("past_key_values", None)
    kw["use_cache"] = False

    return inps, kw
