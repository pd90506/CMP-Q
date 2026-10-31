"""The standalone GPTQ baseline must equal CMP-Q at λ=0 bit for bit (the λ=0 invariant, seen from the other side)."""
import itertools

import torch
import torch.nn as nn

from baselines.gptq import GPTQ
from cmpq.utils.gptq_rev import GPTQREV


def _quantize(cls, seed: int, bits: int, groupsize: int, actorder: bool) -> torch.Tensor:
    torch.manual_seed(seed)
    d_in, d_out = 64, 16
    layer = nn.Linear(d_in, d_out, bias=False)
    layer.weight.data = torch.randn(d_out, d_in) * torch.rand(d_in)
    X = torch.randn(2, 64, d_in) * (torch.rand(d_in) * 3)
    solver = cls(layer, name="t", wbits=bits)
    for b in range(2):
        solver.add_batch(X[b:b+1])
    kw = {"V_matrix": None, "cmpq_lambda": 0.0} if cls is GPTQREV else {}
    solver.fasterquant(percdamp=0.01, groupsize=groupsize, actorder=actorder, **kw)
    return layer.weight.data.clone()


def test_gptq_baseline_matches_cmpq_lambda_zero():
    for seed, (bits, gs, ao) in enumerate(itertools.product([2, 3, 4], [16, -1], [False, True])):
        a = _quantize(GPTQ, seed, bits, gs, ao)
        b = _quantize(GPTQREV, seed, bits, gs, ao)  # mse_clip defaults to False
        assert torch.equal(a, b), (bits, gs, ao)
