"""
QuIP# solver (Tseng et al., ICML 2024), ported from Cornell-RelaxML/quip-sharp
(GPL-3.0; lib/algo/quip.py, lib/codebook/latticee8_padded12*.py,
lib/utils/math_utils.py) for the CMPQ + QuIP# composition experiment.

What is kept from quip-sharp (defaults of quantize_finetune_llama.py, ft_epochs=0):
  - H regularization: H / mean(diag H) + sigma_reg·I, sigma_reg = 1e-2
  - incoherence: random-sign Hadamard on both sides (incoh_mode="had")
  - BlockLDLQ with block = codebook dim 8, quip_tune_iters = 10
  - codebooks: E8P12 (2 bit, scale_override 0.9 as the README suggests) and
    E8P12RVQ3B (3 bit, default scales)
What differs:
  - Hadamard transforms are applied as dense orthogonal matrices (no
    fast_hadamard_transform / quiptools CUDA kernels). Orders 12/20/68 come from
    the Paley-I construction instead of Sloane's tables; any Hadamard matrix of
    that order gives a valid RHT.
  - Weights are written back dequantized (fake-quant), so evaluation runs on the
    standard HF model. Accuracy matches the packed kernels up to fp16 rounding.
  - No fine-tuning stage.

CMPQ hook: the caller passes H_hat = H + λ·(tr H / tr V)·V instead of H. With
λ=0 (or V=None) this is plain QuIP#.
"""
import logging
import math
import torch

logger = logging.getLogger(__name__)

_CODESZ = 8


# ---------------------------------------------------------------------------
# E8P codebooks (quantize side only)
# ---------------------------------------------------------------------------

def _get_norm12() -> torch.Tensor:
    # 29 elements of norm 12 in E8 + 1/4
    return torch.tensor([
        [3, 1, 1, 1, 3, 3, 3, 3], [1, 3, 1, 1, 3, 3, 3, 3], [1, 1, 3, 1, 3, 3, 3, 3],
        [1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 1, 3, 3, 1, 1], [3, 3, 3, 1, 3, 1, 3, 1],
        [3, 3, 3, 1, 1, 3, 3, 1], [3, 3, 3, 1, 3, 1, 1, 3], [3, 3, 3, 1, 1, 3, 1, 3],
        [3, 3, 3, 1, 1, 1, 3, 3], [3, 3, 1, 3, 3, 3, 1, 1], [3, 3, 1, 3, 3, 1, 3, 1],
        [3, 3, 1, 3, 1, 3, 3, 1], [3, 3, 1, 3, 3, 1, 1, 3], [3, 3, 1, 3, 1, 3, 1, 3],
        [3, 3, 1, 3, 1, 1, 3, 3], [3, 1, 3, 3, 3, 3, 1, 1], [3, 1, 3, 3, 3, 1, 3, 1],
        [3, 1, 3, 3, 1, 3, 3, 1], [3, 1, 3, 3, 3, 1, 1, 3], [3, 1, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3], [1, 3, 3, 3, 3, 3, 1, 1], [1, 3, 3, 3, 3, 1, 3, 1],
        [1, 3, 3, 3, 1, 3, 3, 1], [1, 3, 3, 3, 3, 1, 1, 3], [1, 3, 3, 3, 1, 3, 1, 3],
        [1, 1, 3, 3, 1, 3, 3, 3], [3, 3, 1, 1, 3, 3, 3, 1],
    ]) / 2


def _get_abs_grid() -> torch.Tensor:
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * _CODESZ).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1) ** 2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    return torch.concat([d8abs, _get_norm12()], dim=0)


def _get_packed_abs_grid() -> torch.Tensor:
    cba = _get_abs_grid()
    cba = cba[:, [0, 2, 4, 6, 1, 3, 5, 7]]
    cba[:, 7] *= (1 - 2 * (cba.sum(1) % 2))
    cba = (cba * 2 + 8).to(torch.int32)
    acc = cba[:, 0]
    for i in range(7):
        acc = acc | (cba[:, (i + 1)] << ((i + 1) * 4))
    return acc


def _get_grid_part() -> torch.Tensor:
    """quip-sharp's `grid_part`: the parity-odd half of the 2^16-entry E8P grid,
    shifted by +1/4 and restricted to the sign patterns fast_quantize_part uses.
    Built exactly as get_full_grid() + E8P12_codebook.__init__ do."""
    packed_abs = _get_packed_abs_grid().tolist()
    shuffle_map = [0, 4, 1, 5, 2, 6, 3, 7]
    rows = []
    for c in range(1 << 16):
        signs = c & 255
        parity = bin(signs).count("1") & 1
        if not parity:
            continue
        signs ^= parity
        abs_code = packed_abs[c >> 8]
        row = []
        for i in range(8):
            ii = shuffle_map[i]
            v = (((abs_code >> (4 * ii)) & 15) - 8) * 0.5
            row.append(-v if (signs >> ii) & 1 else v)
        rows.append(row)                  # (codeword − 0.25) + 0.25
    grid_part = torch.tensor(rows)
    keep = ((grid_part[:, :7] < 0).sum(dim=-1) <= 1) * (grid_part[:, :7].min(dim=-1).values >= -0.5)
    return grid_part[torch.where(keep)[0]]


def _get_e81b_grid() -> torch.Tensor:
    intr = torch.arange(-4, 4)
    hintr = intr + 1 / 2
    gintr = torch.cartesian_prod(*[intr] * 8)
    ghintr = torch.cartesian_prod(*[hintr] * 8)
    ge8 = torch.concat([gintr, ghintr], dim=0)
    ge8m2 = (ge8.sum(dim=-1) % 2 == 0)
    ge8n = ge8.norm(dim=-1) ** 2 <= 2
    e8 = ge8[torch.where(ge8m2 * ge8n)[0]]
    norm4 = torch.zeros(15, 8)
    for i in range(8):
        norm4[i, i] = 2
    for i in range(7):
        norm4[8 + i, i] = -2
    return torch.concat([e8, norm4], dim=0)


class E8PCodebook:
    """E8P12 (2 bit) and, with rvq=True, E8P12RVQ3B (3 bit = E8P + 1-bit E8 residual)."""

    def __init__(self, bits: int, device: torch.device, dtype: torch.dtype = torch.float32):
        assert bits in (2, 3), f"QuIP# port supports 2 and 3 bits, got {bits}"
        self.bits = bits
        self.codesz = _CODESZ
        self.rvq = bits == 3
        # quip-sharp README: scale_override 0.9 for E8P12, default (opt_scale) for 3 bit.
        self.scale = 0.9 if bits == 2 else 0.98
        self.resid_scale = 2.04
        self.grid_part = _get_grid_part().to(device, dtype)
        self.grid_part_norm = self.grid_part.norm(dim=-1) ** 2
        if self.rvq:
            self.e81b_grid = _get_e81b_grid().to(device, dtype)
            self.e81b_grid_norm = self.e81b_grid.norm(dim=-1) ** 2

    @staticmethod
    def _round(X: torch.Tensor, grid: torch.Tensor, grid_norm: torch.Tensor) -> torch.Tensor:
        return grid[(2 * X @ grid.T - grid_norm).argmax(-1)]

    def _quantize_part(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 7] = -X_part[X_odd, 7]
        mask = 1 - 2 * (X < 0).to(X.dtype)
        mask[X_odd, 7] = -mask[X_odd, 7]
        vals = self._round(X_part, self.grid_part, self.grid_part_norm) * mask
        return vals, (X - vals).norm(dim=-1)

    def _quantize_e8p(self, X: torch.Tensor) -> torch.Tensor:
        plus_vals, plus_err = self._quantize_part(X + 1 / 4)
        minus_vals, minus_err = self._quantize_part(X - 1 / 4)
        which = plus_err < minus_err
        return torch.where(which.unsqueeze(-1), plus_vals - 1 / 4, minus_vals + 1 / 4)

    def quantize(self, X: torch.Tensor) -> torch.Tensor:
        """X: [..., 8] → nearest codeword values (same shape)."""
        shape = X.shape
        X = X.reshape(-1, self.codesz)
        vals = self._quantize_e8p(X)
        if self.rvq:
            resid = (X - vals) * self.resid_scale
            vals = vals + self._round(resid, self.e81b_grid, self.e81b_grid_norm) / self.resid_scale
        return vals.reshape(shape)


# ---------------------------------------------------------------------------
# Randomized Hadamard transform (dense)
# ---------------------------------------------------------------------------

def _is_prime(p: int) -> bool:
    return p >= 2 and all(p % d for d in range(2, int(math.isqrt(p)) + 1))


def _paley1(q: int) -> torch.Tensor:
    """Paley construction I: Hadamard matrix of order q+1 for prime q ≡ 3 (mod 4)."""
    residues = {(x * x) % q for x in range(1, q)}
    chi = torch.tensor([0.0] + [1.0 if a in residues else -1.0 for a in range(1, q)])
    idx = torch.arange(q)
    Q = chi[(idx[None, :] - idx[:, None]) % q]           # Jacobsthal, skew-symmetric
    S = torch.zeros(q + 1, q + 1)
    S[0, 1:] = 1.0
    S[1:, 0] = -1.0
    S[1:, 1:] = Q
    return S + torch.eye(q + 1)


def hadamard_matrix(n: int) -> torch.Tensor:
    """Unnormalized Hadamard matrix of order n = K·2^a, K ∈ {1, 2, q+1 (Paley I)}."""
    a = 0
    while n % (2 ** (a + 1)) == 0:
        a += 1
    base = None
    for j in range(a + 1):
        K = (n >> a) << j
        if K in (1, 2):
            base = torch.ones(1, 1)
            j_used = j
            break
        if _is_prime(K - 1) and (K - 1) % 4 == 3:
            base = _paley1(K - 1)
            j_used = j
            break
    if base is None:
        raise ValueError(f"No Hadamard construction for n={n}")
    H = base
    H2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    while H.shape[0] < n:
        H = torch.kron(H2, H)
    assert H.shape[0] == n, (n, H.shape)
    return H


_HAD_CACHE: dict[int, torch.Tensor] = {}


def random_hadamard(n: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """Public alias of the RHT used by QuIP# (quantization.rht=true in the CMPQ path)."""
    return _rht(n, gen, device)


def _rht(n: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """Orthogonal U = (Had_n / √n) · diag(s), s random signs."""
    if n not in _HAD_CACHE:
        Hd = hadamard_matrix(n).to(device)
        err = (Hd @ Hd.T - n * torch.eye(n, device=device)).abs().max().item()
        assert err < 1e-3, f"Hadamard({n}) check failed: {err}"
        _HAD_CACHE[n] = Hd / math.sqrt(n)
    s = (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).to(device, torch.float32)
    return _HAD_CACHE[n].to(device) * s[None, :]


# ---------------------------------------------------------------------------
# BlockLDLQ (port of quip-sharp LDLQ_buffered_lowmem + block_LDL)
# ---------------------------------------------------------------------------

def _block_ldl(H: torch.Tensor, b: int) -> torch.Tensor | None:
    n = H.shape[0]
    m = n // b
    try:
        L = torch.linalg.cholesky(H)
    except RuntimeError:
        return None
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]
    L = L.reshape(n, n)
    if L.isnan().any():
        return None
    return L


def _ldlq(Wr: torch.Tensor, Hr: torch.Tensor, L: torch.Tensor, cb: E8PCodebook,
          tune_iters: int, buf_cols: int = 128) -> torch.Tensor:
    (m, n) = Wr.shape
    cs = cb.codesz
    hatWr = torch.zeros_like(Wr)
    assert n % buf_cols == 0 and buf_cols % cs == 0
    buf_size = buf_cols // cs

    prod_cache = torch.zeros_like(Wr)
    for cur_col in range(n // cs, 0, -buf_size):
        lo, hi = cs * (cur_col - buf_size), cs * cur_col
        b_Wr = Wr[:, lo:hi]
        b_hatWr = hatWr[:, lo:hi]
        b_L = L[lo:hi]
        b_prod = prod_cache[:, lo:hi]
        for i in reversed(range(buf_size)):
            WXWX = b_Wr[:, cs * i:cs * (i + 1)] + \
                (b_Wr[:, cs * (i + 1):] - b_hatWr[:, cs * (i + 1):]) @ \
                b_L[cs * (i + 1):, lo + cs * i:lo + cs * (i + 1)] + \
                b_prod[:, cs * i:cs * (i + 1)]
            b_hatWr[:, cs * i:cs * (i + 1)] = cb.quantize(WXWX)
        prod_cache += (b_Wr - b_hatWr) @ b_L
    del prod_cache

    for _ in range(tune_iters):
        delta = Wr - hatWr
        for cur_col in range(n // cs, 0, -buf_size):
            lo, hi = cs * (cur_col - buf_size), cs * cur_col
            b_hatWr = hatWr[:, lo:hi]
            b_Hr = Hr[:, lo:hi]
            b_delta = delta[:, lo:hi]
            for i in reversed(range(buf_size)):
                inv = torch.linalg.inv(b_Hr[lo + cs * i:lo + cs * (i + 1), cs * i:cs * (i + 1)])
                WXWX = b_hatWr[:, cs * i:cs * (i + 1)] + delta @ b_Hr[:, cs * i:cs * (i + 1)] @ inv
                b_delta[:, cs * i:cs * (i + 1)] += b_hatWr[:, cs * i:cs * (i + 1)]
                b_hatWr[:, cs * i:cs * (i + 1)] = cb.quantize(WXWX)
                b_delta[:, cs * i:cs * (i + 1)] -= b_hatWr[:, cs * i:cs * (i + 1)]
        del delta
    return hatWr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def quip_quantize(
    W: torch.Tensor,
    H: torch.Tensor,
    cb: E8PCodebook,
    gen: torch.Generator,
    sigma_reg: float = 1e-2,
    tune_iters: int = 10,
) -> tuple[torch.Tensor, float]:
    """QuIP# (no FT) on one linear layer.

    Args:
        W: [m, n] weight (any dtype; computed in fp32 on W's device).
        H: [n, n] proxy Hessian (CMPQ passes H_hat). Not modified.
        gen: CPU generator for the random Hadamard signs (seeded per layer by the
             caller so QuIP# and QuIP#+CMPQ share identical transforms).

    Returns:
        (hatW [m, n] fp32, proxy loss tr((Ŵ−W) H_reg (Ŵ−W)ᵀ) / tr(W H_reg Wᵀ)).
    """
    dev = W.device
    W = W.float()
    (m, n) = W.shape
    assert m % 2 == 0 and n % 128 == 0, f"QuIP# needs n % 128 == 0, got {W.shape}"

    Hreg = H.float().clone()
    dead = torch.diag(Hreg) == 0
    Hreg[dead, dead] = 1
    W = W.clone()
    W[:, dead] = 0
    Hreg /= torch.diag(Hreg).mean()
    Hreg.diagonal().add_(sigma_reg)

    U = _rht(n, gen, dev)
    Vm = _rht(m, gen, dev)
    Hr = U @ Hreg @ U.T
    Hr = (Hr + Hr.T) / 2
    Wr = Vm @ W @ U.T

    L = _block_ldl(Hr, cb.codesz)
    if L is None:
        logger.warning("block_LDL failed in fp32, retrying in fp64")
        L = _block_ldl(Hr.double(), cb.codesz)
        assert L is not None, "block_LDL failed in fp64"
        L = L.float()

    Wscale = Wr.square().mean().sqrt() / cb.scale
    hatWr = _ldlq(Wr / Wscale, Hr, L, cb, tune_iters) * Wscale
    del L

    hatW = Vm.T @ hatWr @ U
    assert torch.isfinite(hatW).all()

    err = hatW - W
    proxy = (torch.einsum("ij,jk,ik->", err, Hreg, err) /
             torch.einsum("ij,jk,ik->", W, Hreg, W).clamp_min(1e-12)).item()
    return hatW, proxy
