"""Tests for the CMP-Q solver: V accumulation and Ĥ = H + λ·(tr H / tr V)·V."""
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cmpq.utils.gptq_rev import GPTQREV


FIXTURES = Path(__file__).parent / "fixtures"


def _make_layer(d_in: int, d_out: int, seed: int) -> nn.Linear:
    torch.manual_seed(seed)
    layer = nn.Linear(d_in, d_out, bias=False)
    layer.weight.data = torch.randn(d_out, d_in)
    return layer.cpu()


def test_zero_delta_matches_no_damping():
    """ΔX = 0 ⇒ V = 0 ⇒ Ĥ equals the undamped GPTQ Hessian ⇒ Q matches."""
    d_in, d_out, n_tokens = 8, 4, 32
    torch.manual_seed(0)
    X = torch.randn(1, n_tokens, d_in)

    # Path A: no V ⇒ damping term disabled
    layer_a = _make_layer(d_in, d_out, seed=42)
    gptq_a = GPTQREV(layer_a, name="a", wbits=4)
    gptq_a.add_batch(X)
    gptq_a.fasterquant(V_matrix=None, cmpq_lambda=0.01, percdamp=0.01, actorder=False)
    Q_a = layer_a.weight.data.clone()

    # Path B: V = 0 ⇒ damping term is λ·trace_ratio·0 = 0
    layer_b = _make_layer(d_in, d_out, seed=42)
    gptq_b = GPTQREV(layer_b, name="b", wbits=4)
    gptq_b.add_batch(X)
    gptq_b.add_batch_delta(torch.zeros_like(X))
    gptq_b.fasterquant(V_matrix=gptq_b.V, cmpq_lambda=0.01, percdamp=0.01, actorder=False)
    Q_b = layer_b.weight.data.clone()

    assert torch.allclose(Q_a, Q_b, atol=1e-6), (
        f"Zero-ΔX run diverged from undamped GPTQ (max diff {(Q_a - Q_b).abs().max().item():.2e})"
    )


def test_rank1_delta_yields_outer_product():
    """ΔX = u·sᵀ ⇒ V should equal 2 · (sᵀs) · u uᵀ (single batch, tmp=1).

    The 2/N normalization in add_batch_delta with tmp=1 gives a factor of 2
    (matching add_batch's behavior on a single sample).
    """
    d_in, n_tokens = 8, 32
    torch.manual_seed(1)
    u = torch.randn(d_in)
    s = torch.randn(n_tokens)
    # Shape [1, n_tokens, d_in] so token t is u * s[t]
    delta = (u.unsqueeze(0) * s.unsqueeze(1)).unsqueeze(0)

    layer = _make_layer(d_in, 4, seed=0)
    gptq = GPTQREV(layer, name="t", wbits=4)
    gptq.add_batch_delta(delta)

    expected = 2.0 * (s @ s).item() * torch.outer(u, u)
    assert torch.allclose(gptq.V, expected, atol=1e-5), (
        f"V structure mismatch (max diff {(gptq.V - expected).abs().max().item():.2e})"
    )


def test_streaming_matches_single_batch():
    """V from streaming (one sample at a time) equals V from a single batched call.

    Confirms the running-mean normalization in add_batch_delta is correct.
    """
    d_in, n_samples, n_tokens = 6, 4, 16
    torch.manual_seed(2)
    deltas = [torch.randn(1, n_tokens, d_in) for _ in range(n_samples)]

    layer_s = _make_layer(d_in, 4, seed=0)
    gptq_s = GPTQREV(layer_s, name="s", wbits=4)
    for d in deltas:
        gptq_s.add_batch_delta(d)
    V_stream = gptq_s.V.clone()

    layer_b = _make_layer(d_in, 4, seed=0)
    gptq_b = GPTQREV(layer_b, name="b", wbits=4)
    gptq_b.add_batch_delta(torch.cat(deltas, dim=0))
    V_batch = gptq_b.V.clone()

    assert torch.allclose(V_stream, V_batch, atol=1e-5), (
        f"Streaming V diverged from single-batch V "
        f"(max diff {(V_stream - V_batch).abs().max().item():.2e})"
    )


def test_lambda_zero_matches_gptq():
    """λ=0 invariant: a non-zero V has no effect, the result is plain GPTQ exactly."""
    d_in, d_out, n_tokens = 8, 4, 32
    torch.manual_seed(6)
    X = torch.randn(1, n_tokens, d_in)
    delta = torch.randn(1, n_tokens, d_in)

    layer_v = _make_layer(d_in, d_out, seed=13)
    gptq_v = GPTQREV(layer_v, name="v0", wbits=4)
    gptq_v.add_batch(X)
    gptq_v.add_batch_delta(delta)
    gptq_v.fasterquant(V_matrix=gptq_v.V, cmpq_lambda=0.0, percdamp=0.01, actorder=False)
    Q_v = layer_v.weight.data.clone()

    layer_base = _make_layer(d_in, d_out, seed=13)
    gptq_base = GPTQREV(layer_base, name="base", wbits=4)
    gptq_base.add_batch(X)
    gptq_base.fasterquant(V_matrix=None, cmpq_lambda=0.0, percdamp=0.01, actorder=False)
    Q_base = layer_base.weight.data.clone()

    assert torch.equal(Q_v, Q_base), (
        f"λ=0 invariant broken: CMP-Q differs from GPTQ "
        f"(max diff {(Q_v - Q_base).abs().max().item():.2e})"
    )


def test_bit_identical_regression():
    """CMP-Q output must be byte-stable across changes (groups, act-order, MSE clip).

    The fixture was generated with the code that produced the paper's results.
    If a change is intentional, delete the fixture and rerun to regenerate it.
    """
    d_in, d_out, n_tokens = 64, 16, 128
    torch.manual_seed(3)
    X = torch.randn(2, n_tokens, d_in) * (torch.rand(d_in) * 3)
    delta = 0.05 * torch.randn(2, n_tokens, d_in)

    layer = _make_layer(d_in, d_out, seed=7)
    gptq = GPTQREV(layer, name="r", wbits=2)
    gptq.add_batch(X)
    gptq.add_batch_delta(delta)
    gptq.fasterquant(V_matrix=gptq.V, cmpq_lambda=0.1, percdamp=0.01, groupsize=16,
                     actorder=True, mse_clip=True)
    Q = layer.weight.data.clone()

    fixture_path = FIXTURES / "cmpq_q_reference.pt"
    if not fixture_path.exists():
        FIXTURES.mkdir(parents=True, exist_ok=True)
        torch.save(Q, fixture_path)
        pytest.skip(f"Fixture written to {fixture_path}; rerun to verify.")

    Q_ref = torch.load(fixture_path)
    assert torch.equal(Q, Q_ref), (
        f"CMP-Q Q drift detected (max diff {(Q - Q_ref).abs().max().item():.2e}). "
        f"If intentional, delete {fixture_path} and regenerate."
    )
