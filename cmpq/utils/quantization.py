import torch
import torch.nn as nn
from numpy import sign

@torch.no_grad()
def normal_quantize(x, scale, zero, maxq):
    return torch.clamp(torch.round(x / scale + zero), 0, maxq)

@torch.no_grad()
def binary_scale(x):
    scale_tensor = torch.mean(torch.abs(x), dim=1) * 2
    zero = 0.5 * torch.ones_like(scale_tensor)
    return scale_tensor, zero

@torch.no_grad()
def binary(x):
    x_ = torch.zeros_like(x)
    x_ += x
    binary_slice = torch.where(x_ >= 0, 1, -1)
    return binary_slice/2 + 0.5 # change (-1, 1) weights to (0, 1) weights

class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0, dtype=torch.float16))
        self.register_buffer('scale', torch.zeros(shape, dtype=torch.float16))
        self.register_buffer('zero', torch.zeros(shape, dtype=torch.float16))

    def configure(self, bits, perchannel=False, sym=True, mse=False, norm=2.4, grid=100, maxshrink=.8, trits=False, pack=False, mse_clip=False, mse_clip_grid=20, mse_clip_min_ratio=0.5):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)
        self.scale = torch.zeros_like(self.scale, dtype=torch.float16)
        self.pack = pack
        self.mse_clip = mse_clip
        self.mse_clip_grid = mse_clip_grid
        self.mse_clip_min_ratio = mse_clip_min_ratio

    def _quantize(self, x, scale, zero, maxq):
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        if self.maxq == 1:
            q = binary(x)
        else:
            q = normal_quantize(x, scale, zero, maxq)
        return scale * (q - zero) if not self.pack else q

    def find_params(self, w, weight=False):
        perchannel = True
        weight = True
        dev = w.device
        maxq = self.maxq
        scale = torch.zeros(1, dtype=torch.float16)
        zero = torch.zeros(1, dtype=torch.float16)
        shape = w.shape

        x = w.clone().to(torch.float16)

        if self.maxq == 1:
            scale, zero = binary_scale(x)
            if dev != scale.device:
                scale=scale.to(dev)
                zero=zero.to(dev)
                maxq=maxq.to(dev)
        else:
            if dev != scale.device:
                scale=scale.to(dev)
                zero=zero.to(dev)
                maxq=maxq.to(dev)
            if perchannel:
                if weight:
                    x = x.flatten(1)
                else:
                    if len(shape) == 4:
                        x = x.permute([1, 0, 2, 3])
                        x = x.flatten(1)
                    if len(shape) == 3:
                        x = x.reshape((-1, shape[-1])).t()
                    if len(shape) == 2:
                        x = x.t()
            else:
                x = x.flatten().unsqueeze(0)
            tmp = torch.zeros(x.shape[0], device=dev, dtype=torch.float16)
            xmin = torch.minimum(x.min(1)[0], tmp)
            xmax = torch.maximum(x.max(1)[0], tmp)

            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmin < 0
                if torch.any(tmp):
                    xmin[tmp] = -xmax[tmp]

            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1

            if maxq < 0:
                scale = xmax
                zero = xmin
            else:
                scale = (xmax - xmin) / maxq
                if self.sym:
                    zero = torch.full_like(scale, (maxq + 1) / 2)
                else:
                    zero = torch.round(-xmin / scale)

                # MSE clip search: tighten xmax per-group to minimize fake-quant MSE.
                # Addresses the "one outlier dominates scale → typical weights crushed
                # to ~0" failure mode at low bitwidth (W3). Uses W unchanged but picks
                # a smaller xmax (shrinks quantization step) if it reduces per-row MSE.
                if getattr(self, "mse_clip", False) and maxq > 0:
                    best_scale = scale.clone()
                    best_zero = zero.clone()
                    best_err = torch.full_like(xmax, float("inf"))
                    # x is [rows, group_size]; iterate clip ratios and pick the best per row.
                    n_grid = int(self.mse_clip_grid)
                    min_r = float(self.mse_clip_min_ratio)
                    for step in range(n_grid):
                        ratio = min_r + (1.0 - min_r) * step / max(n_grid - 1, 1)
                        xmax_try = xmax * ratio
                        if self.sym:
                            xmin_try = -xmax_try if not torch.all(xmin == 0) else torch.zeros_like(xmax_try)
                        else:
                            xmin_try = xmin  # asymmetric: only tighten the positive side
                        scale_try = (xmax_try - xmin_try) / maxq
                        scale_try = torch.clamp(scale_try, min=1e-8)
                        if self.sym:
                            zero_try = torch.full_like(scale_try, (maxq + 1) / 2)
                        else:
                            zero_try = torch.round(-xmin_try / scale_try)
                        # Fake-quant x with these params (shape: [rows, group_size])
                        q = torch.clamp(
                            torch.round(x / scale_try.unsqueeze(1) + zero_try.unsqueeze(1)),
                            0, maxq,
                        )
                        x_hat = scale_try.unsqueeze(1) * (q - zero_try.unsqueeze(1))
                        err = (x_hat - x).pow(2).mean(dim=1)
                        mask = err < best_err
                        best_err = torch.where(mask, err, best_err)
                        best_scale = torch.where(mask, scale_try, best_scale)
                        best_zero = torch.where(mask, zero_try, best_zero)
                    scale = best_scale
                    zero = best_zero

        if not perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            scale = scale.repeat(tmp)
            zero = zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            scale = scale.reshape(shape)
            zero = zero.reshape(shape)

        self.scale = scale
        self.zero = zero
        self.maxq = maxq

    def quantize(self, x):
        if self.ready():
            if self.pack:
                return self._quantize(x, self.scale, self.zero, self.maxq), self.scale, self.zero
            return self._quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)
