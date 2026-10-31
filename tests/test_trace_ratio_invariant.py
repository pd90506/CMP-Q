"""Numerical micro-check for the CMP-Q damping convention.

Verifies that fasterquant applies
exactly Ĥ = H + λ · (tr(H)/tr(V)) · V before Cholesky. Synthetic W/X/ΔX —
no model load, no GPU needed.
"""
import torch
import torch.nn as nn

from cmpq.utils.gptq_rev import GPTQREV


def _build_solver(d_in: int, d_out: int, n_tokens: int, seed: int = 0):
    torch.manual_seed(seed)
    layer = nn.Linear(d_in, d_out, bias=False).cpu()
    layer.weight.data = torch.randn(d_out, d_in)

    X = torch.randn(1, n_tokens, d_in)               # (B, T, d_in)
    dX = 0.1 * torch.randn(1, n_tokens, d_in)        # ΔX, distinct distribution

    g = GPTQREV(layer, name="test_layer", wbits=4)
    g.add_batch(X)
    g.add_batch_delta(dX)
    return g, X, dX


def _expected_H_V(X: torch.Tensor, dX: torch.Tensor):
    """Reference: H = (2/n_batches) · sum_b X_bᵀ X_b. Note add_batch counts BATCHES
    (X.shape[0]), not tokens — (2/N) in the docstring refers to batch count, not
    token count. Each batch X_b is (T, d_in), contributing X_bᵀ X_b which itself
    sums over T tokens."""
    n_batches = X.shape[0]
    Xf = X.reshape(-1, X.shape[-1]).float()
    dXf = dX.reshape(-1, dX.shape[-1]).float()
    H = (2.0 / n_batches) * (Xf.T @ Xf)
    V = (2.0 / n_batches) * (dXf.T @ dXf)
    return H, V


def test_h_and_v_match_reference():
    g, X, dX = _build_solver(d_in=16, d_out=8, n_tokens=64)
    H_expected, V_expected = _expected_H_V(X, dX)
    err_H = (g.H - H_expected).abs().max().item()
    err_V = (g.V - V_expected).abs().max().item()
    print(f"  H max-abs error vs (2/N)·Xᵀ·X     : {err_H:.3e}")
    print(f"  V max-abs error vs (2/N)·ΔXᵀ·ΔX  : {err_V:.3e}")
    assert err_H < 1e-3, f"H mismatch: {err_H}"
    assert err_V < 1e-3, f"V mismatch: {err_V}"


def test_h_modification_equals_lambda_times_trace_ratio_times_V():
    """Inject a sentinel: monkey-patch torch.linalg.cholesky to capture H_modified.
    Verify H_modified ≈ H_orig + percdamp_term + λ·trace_ratio·V.
    """
    g, X, dX = _build_solver(d_in=16, d_out=8, n_tokens=64, seed=3)
    H_orig = g.H.clone()
    V_orig = g.V.clone()
    lam = 0.7

    expected_trace = (torch.diag(H_orig).sum() /
                      torch.diag(V_orig).sum().clamp_min(1e-12)).item()
    percdamp = 0.01
    damp = percdamp * torch.mean(torch.diag(H_orig)).item()

    captured = {}
    real_chol = torch.linalg.cholesky

    def spy(mat, *a, **kw):
        if "H_mod" not in captured:
            captured["H_mod"] = mat.clone()
        return real_chol(mat, *a, **kw)

    torch.linalg.cholesky = spy
    try:
        g.fasterquant(
            blocksize=128,
            percdamp=percdamp,
            groupsize=-1,
            actorder=False,
            V_matrix=V_orig,
            cmpq_lambda=lam,
        )
    finally:
        torch.linalg.cholesky = real_chol

    H_mod = captured["H_mod"].cpu()
    expected = H_orig.cpu().clone()
    diag = torch.arange(expected.shape[0])
    expected[diag, diag] += damp
    expected = expected + lam * expected_trace * V_orig.cpu()

    err = (H_mod - expected).abs().max().item()
    rel = err / expected.abs().max().clamp_min(1e-12).item()
    print(f"  ||Ĥ_actual − Ĥ_expected||_∞      = {err:.3e}")
    print(f"  relative                          = {rel:.3e}")
    assert rel < 1e-4, f"Ĥ mismatch (rel err {rel:.3e})"


if __name__ == "__main__":
    print("=== micro-check: trace-ratio convention ===\n")
    print("[1/2] H/V match (2/N)·XXᵀ reference")
    test_h_and_v_match_reference()
    print("       ✓\n")
    print("[2/2] Ĥ_actual = H + percdamp + λ·trace_ratio·V")
    test_h_modification_equals_lambda_times_trace_ratio_times_V()
    print("       ✓\n")
    print("ALL CHECKS PASSED")
