"""Small CPU checks for the QuIP# port (cmpq/utils/quip_sharp.py)."""
import torch

from cmpq.utils.quip_sharp import E8PCodebook, _paley1, hadamard_matrix, quip_quantize


def test_hadamard_orders():
    for q in (11, 19, 67):  # orders 12, 20, 68 (Qwen3 1.7B/8B, 14B hidden, 14B MLP)
        H = _paley1(q)
        n = q + 1
        assert torch.equal(H @ H.T, n * torch.eye(n))
    for n in (128, 1024, 2048, 5120, 6144):
        H = hadamard_matrix(n)
        assert torch.equal(H @ H.T, n * torch.eye(n)), n


def test_codebook_sizes_and_mse():
    torch.manual_seed(0)
    X = torch.randn(200_000, 8)
    cb2 = E8PCodebook(2, torch.device("cpu"))
    # quip-sharp E8P12: 256 abs patterns × parity-restricted signs.
    assert cb2.grid_part.shape[1] == 8
    mse2 = (cb2.quantize(X * cb2.scale) / cb2.scale - X).pow(2).mean().item()
    cb3 = E8PCodebook(3, torch.device("cpu"))
    assert cb3.e81b_grid.shape == (256, 8)
    mse3 = (cb3.quantize(X * cb3.scale) / cb3.scale - X).pow(2).mean().item()
    print(f"grid_part={tuple(cb2.grid_part.shape)} mse2={mse2:.4f} mse3={mse3:.4f}")
    # QuIP# paper: E8P 2-bit Gaussian MSE ≈ 0.089; 3-bit RVQ is far lower.
    assert 0.07 < mse2 < 0.11
    assert mse3 < 0.5 * mse2


def test_quip_quantize_beats_rtn_proxy():
    torch.manual_seed(0)
    m, n = 256, 512
    X = torch.randn(4096, n) * torch.linspace(0.2, 3.0, n)
    H = X.T @ X / X.shape[0]
    W = torch.randn(m, n) * 0.02
    for bits in (2, 3):
        cb = E8PCodebook(bits, torch.device("cpu"))
        hatW, proxy = quip_quantize(W, H, cb, torch.Generator().manual_seed(1), tune_iters=2)
        assert hatW.shape == W.shape and torch.isfinite(hatW).all()
        print(f"bits={bits} proxy={proxy:.4f}")
        assert proxy < (0.15 if bits == 2 else 0.05)
        # Deterministic given the generator seed.
        hatW2, _ = quip_quantize(W, H, cb, torch.Generator().manual_seed(1), tune_iters=2)
        assert torch.equal(hatW, hatW2)


if __name__ == "__main__":
    test_hadamard_orders()
    test_codebook_sizes_and_mse()
    test_quip_quantize_beats_rtn_proxy()
    print("OK")
