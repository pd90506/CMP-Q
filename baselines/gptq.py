"""
GPTQ (Frantar et al., arXiv:2210.17323) baseline solver.

Standard GPTQ: H = (2/N)·X·Xᵀ from the inputs of the (partially) quantized
model, percdamp diagonal damping, Cholesky of H⁻¹, and the column-wise
quantize-and-compensate loop with lazy block updates. Plain min-max group
quantizer.
"""
import math
import torch
import torch.nn as nn
import logging
from cmpq.utils.quantization import Quantizer
from cmpq.utils.device import synchronize

logger = logging.getLogger(__name__)


class GPTQ:
    def __init__(self, layer, name: str, wbits: int):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()
        self.name = name
        self.wbits = wbits

    def add_batch(self, inp: torch.Tensor) -> None:
        """Accumulate Hessian H = 2·X·Xᵀ from a batch of inputs."""
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if inp.dim() == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()  # [d_in, n_tokens]

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
        actorder: bool = False,
    ):
        self.layer.to(self.dev)
        W = self.layer.weight.data.clone().float()

        self.quantizer.configure(self.wbits)
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        diag = torch.arange(self.columns, device=self.dev)
        damp = percdamp * torch.mean(torch.diag(H))
        H[diag, diag] += damp

        # Cholesky — MPS can fail on near-rank-deficient matrices; retry on CPU.
        def _cholesky_inv(mat: torch.Tensor) -> torch.Tensor:
            L = torch.linalg.cholesky(mat)
            return torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)

        try:
            Hinv = _cholesky_inv(H)
        except RuntimeError:
            try:
                if H.device.type != "mps":
                    raise
                logger.warning(f"Cholesky failed on MPS for {self.name}, retrying on CPU")
                Hinv = _cholesky_inv(H.cpu()).to(H.device)
            except RuntimeError:
                logger.warning(f"Hessian not positive definite for {self.name}, using RTN fallback")
                return None, None, None, None

        scale = []
        zero = []
        now_idx = 1

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    if ((i1 + i) // groupsize) - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1

                q_r = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q_r

                Losses1[:, i] = (w - q_r) ** 2 / d ** 2
                err1 = (w - q_r) / d

                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        synchronize()
        error = torch.sum(Losses).item()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = torch.tensor(
            [i // groupsize for i in range(self.columns)],
            dtype=torch.int32,
            device=Q.device,
        )

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        logger.info(f"[GPTQ] Quantized {self.name} to {self.wbits} bits. Error: {error:.4f}")

        if not scale:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        return torch.cat(scale, dim=1), torch.cat(zero, dim=1), g_idx, error
