"""
GPTAQ (Asymmetric Calibration GPTQ, arXiv:2504.02692) baseline implementation.

Follows ResComp's gptaq_utils.py (https://github.com/list0830/ResComp) algorithm exactly.

Loss: minimize ‖ŴX_q − WX_fp‖² instead of GPTQ's ‖ŴX_q − WX_q‖².
The cross-term ΔX = X_fp − X_q ≈ -(quant noise propagation) introduces an
extra "P term" in the column-wise update that retroactively biases already-
quantized weights to compensate for upstream-quantization-induced input drift.

Update rule (vs vanilla GPTQ):
    GPTQ:   W[:, i:] -= err1 · Hinv[i, i:]
    GPTAQ:  W[:, i:] -= err1 · Hinv[i, i:] - w · P[i, i:]
where P = α · triu(ΔXᵀX · H⁻¹), α=0.25 (ResComp default).

"""
import math
import torch
import torch.nn as nn
import logging
from cmpq.utils.quantization import Quantizer
from cmpq.utils.device import synchronize

logger = logging.getLogger(__name__)


class GPTAQ:
    def __init__(self, layer, name: str, wbits: int):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()
        self.name = name
        self.wbits = wbits

    def add_batch(self, inp: torch.Tensor, fp_inp: torch.Tensor) -> None:
        """Accumulate H from Q-chain inp + dXXT from (X_fp − X_q)·X_q^T.

        Both inp and fp_inp must be the SAME calibration batch propagated
        through Q-chain (with previously-quantized predecessors) and FP-chain
        (all-FP16 predecessors) respectively. Matches ResComp convention.
        """
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
            fp_inp = fp_inp.unsqueeze(0)
        tmp = inp.shape[0]
        if inp.dim() == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
            fp_inp = fp_inp.reshape((-1, fp_inp.shape[-1]))
        inp = inp.t()              # [d_in, n_tokens]
        fp_inp = fp_inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        scale = math.sqrt(2 / self.nsamples)
        inp_s = scale * inp.float()
        fp_inp_s = scale * fp_inp.float()

        self.H += inp_s.matmul(inp_s.t())
        # ΔX = X_fp − X_q (ResComp sign convention)
        dX = fp_inp_s - inp_s
        self.dXXT += dX.matmul(inp_s.t())

    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
        actorder: bool = False,
        mse_clip: bool = False,
        gptaq_alpha: float = 0.25,
    ):
        """Standard GPTQ column loop + GPTAQ P-term cross update."""
        self.layer.to(self.dev)
        W = self.layer.weight.data.clone().float()

        self.quantizer.configure(self.wbits, mse_clip=mse_clip)
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        dXXT = self.dXXT
        del self.H, self.dXXT

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            dXXT = dXXT[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        diag_idx = torch.arange(self.columns, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        H[diag_idx, diag_idx] += damp

        # Cholesky of H^-1 (upper) — same as gptq_rev
        def _chol_inv(mat):
            L = torch.linalg.cholesky(mat)
            return torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)

        try:
            Hinv = _chol_inv(H)
        except RuntimeError:
            logger.warning(f"GPTAQ: Hessian not PD for {self.name}, RTN fallback")
            return None, None, None, None

        # Build GPTAQ P matrix: α · triu(ΔX^T·X·H^-1)
        # dXXT = ΔX·X_q^T. We want ΔX^T·X_q effectively, but ResComp computes
        # exactly (dXXT @ Hinv.T).triu(diag=1) @ Hinv. The two Hinv multiplies
        # reconstruct H^-1; the strict-upper projection drops diagonal so it
        # doesn't double-count GPTQ's existing diagonal step.
        P = gptaq_alpha * ((dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv
        del dXXT

        scale_list, zero_list = [], []
        now_idx = 1

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    if ((i1 + i) // groupsize) - now_idx == -1:
                        scale_list.append(self.quantizer.scale)
                        zero_list.append(self.quantizer.zero)
                        now_idx += 1

                q_r = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q_r
                Losses1[:, i] = (w - q_r) ** 2 / d ** 2
                err1 = (w - q_r) / d

                # GPTAQ-modified update: standard term + cross-term P
                update = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) \
                       - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                W1[:, i:] -= update
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            # block-tail update
            tail = Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])
            W[:, i2:] -= tail

        synchronize()
        error = torch.sum(Losses).item()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = torch.tensor(
            [i // groupsize for i in range(self.columns)],
            dtype=torch.int32, device=Q.device,
        )

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        logger.info(f"[GPTAQ] Quantized {self.name} to {self.wbits} bits. Error: {error:.4f}")

        if not scale_list:
            scale_list.append(self.quantizer.scale)
            zero_list.append(self.quantizer.zero)
        return torch.cat(scale_list, dim=1), torch.cat(zero_list, dim=1), g_idx, error
