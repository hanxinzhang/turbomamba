"""FP32 Apple MPS layers and Mamba-3 SISO adaptation for TurboMamba.

Mamba-3 parameterization derives from state-spaces/mamba:
Copyright (c) 2026, Dao AI Lab, Goombalab. Apache-2.0.
See THIRD_PARTY_NOTICES.md and LICENSES/mamba-Apache-2.0.txt.
Training, prefill and cached inference use FP32 Torch operations on Apple MPS.
"""

import hashlib
import math
import os
import torch
from torch import nn
from torch.nn import functional as F

V = 4096
D = 128


def check_device(x=None):
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("CPU fallback is forbidden")
    if x is not None and x.device.type != "mps":
        raise RuntimeError("All neural execution must use MPS")


class RMS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        check_device(x)
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * self.weight


def silu_gate(projected):
    a, b = projected.chunk(2, -1)
    return a * F.silu(b)


class SwiGLU(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.up = nn.Linear(d, 2 * ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down(silu_gate(self.up(x)))


def rope_geometry(length, d, device):
    f = torch.exp(
        torch.arange(0, d, 2, device=device, dtype=torch.float32)
        * (-math.log(500000.0) / d)
    )
    a = torch.arange(length, device=device, dtype=torch.float32)[:, None] * f
    return a.cos(), a.sin()


def rotary(x, cos, sin):
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)


class AttentionBlock(nn.Module):
    def __init__(self, width=D, heads=4, kv_heads=2, ff=680):
        super().__init__()
        self.h, self.k, self.p = heads, kv_heads, width // heads
        self.norm1, self.norm2 = RMS(width), RMS(width)
        self.qkv = nn.Linear(width, width + 2 * kv_heads * self.p, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.mlp = SwiGLU(width, ff)

    def forward(self, x, start=0, geometry=None, cache=None, last_only=False):
        n, t, d = x.shape
        if geometry is None:
            geometry = rope_geometry(start + t, self.p, x.device)
        q, k, v = self.qkv(self.norm1(x)).split(
            (d, self.k * self.p, self.k * self.p), -1
        )
        q = q.reshape(n, t, self.h, self.p).transpose(1, 2)
        k = k.reshape(n, t, self.k, self.p).transpose(1, 2)
        v = v.reshape(n, t, self.k, self.p).transpose(1, 2)
        cos, sin = [a[start : start + t] for a in geometry]
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if cache is not None:
            if t != 1:
                raise ValueError("Cached attention accepts exactly one token")
            k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        stored = (k, v)
        if self.k != self.h:
            k, v = k.repeat_interleave(self.h // self.k, 1), v.repeat_interleave(
                self.h // self.k, 1
            )
        if last_only and t > 1:
            q, x = q[:, :, -1:], x[:, -1:]
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=cache is None and not last_only, dropout_p=0.0
        )
        x = x + self.output(y.transpose(1, 2).reshape(n, -1, d))
        return x + self.mlp(self.norm2(x)), stored


def stable_init(module, seed, depth=8):
    # Identical named tensors share initialization across matching controls.
    for name, p in module.named_parameters():
        if ".mixer." in name or ".ssm." in name:
            continue
        if "norm" in name:
            p.data.fill_(1.0)
        else:
            g = torch.Generator(device=p.device).manual_seed(
                seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            )
            p.data.normal_(0.0, 0.02, generator=g)
            if name.endswith(("output.weight", "down.weight")):
                p.data.div_(math.sqrt(2 * depth))


def sample(probability, uniform):
    check_device(probability)
    cdf = (probability / probability.sum(-1, keepdim=True)).cumsum(-1)
    return (cdf < uniform[:, None]).sum(-1).clamp_max(probability.shape[-1] - 1)


def rotate(x, phase, mimo):
    cs, sn = phase.cos()[:, :, :, None, :], phase.sin()[:, :, :, None, :]
    if mimo:
        a, b, c, d = x.chunk(4, -1)
        return torch.cat((a * cs - c * sn, b, a * sn + c * cs, d), -1)
    rotated = x[..., :32].reshape(*x.shape[:-1], 16, 2)
    a, b = rotated[..., 0], rotated[..., 1]
    return torch.cat(
        (torch.stack((a * cs - b * sn, a * sn + b * cs), -1).flatten(-2), x[..., 32:]),
        -1,
    )


def chunk_scan(q, k, v, adt, dt, trap, z, skip, outw, chunk):
    """Matrix SSD with parallel inter-chunk scan; B,H,T,R,N/P layout."""
    B, H, T, R, N = q.shape
    P = v.shape[-1]
    pad = (-T) % chunk
    gamma = dt * trap
    shift = F.pad((dt * (1 - trap))[:, :, 1:], (0, 1))
    factor = gamma + shift
    qkvdiag = torch.einsum("bhtrn,bhtsn->bhtrs", q, k)
    correction = (
        torch.einsum("bhtrs,bhtsp->bhtrp", qkvdiag, v) * shift[:, :, :, None, None]
    )

    def padded(x):
        return F.pad(x, (0, 0, 0, 0, 0, pad))

    q, k, v = padded(q), padded(k * factor[:, :, :, None, None]), padded(v)
    a = F.pad(adt, (0, pad))
    blocks = (T + pad) // chunk
    cs = a.reshape(B, H, blocks, chunk).cumsum(-1)
    end = cs[:, :, :, -1]
    ep = end.cumsum(-1)
    qi = q.reshape(B, H, blocks, chunk * R, N)
    ki = k.reshape(B, H, blocks, chunk * R, N)
    vi = v.reshape(B, H, blocks, chunk * R, P)
    rows = torch.arange(chunk, device=q.device)
    mask = rows[:, None] >= rows[None, :]
    logdecay = cs[:, :, :, :, None] - cs[:, :, :, None, :]
    decay = torch.exp(logdecay.masked_fill(~mask, float("-inf")))
    decay = decay.repeat_interleave(R, -2).repeat_interleave(R, -1)

    def batch_mm(a, b):
        shape = a.shape[:-2]
        result = torch.bmm(
            a.reshape(-1, a.shape[-2], a.shape[-1]).contiguous(),
            b.reshape(-1, b.shape[-2], b.shape[-1]).contiguous(),
        )
        return result.reshape(*shape, a.shape[-2], b.shape[-1])

    intra = batch_mm(batch_mm(qi, ki.transpose(-1, -2)) * decay, vi)
    ks = (
        ki
        * torch.exp((end[:, :, :, None] - cs).repeat_interleave(R, -1))[
            :, :, :, :, None
        ]
    )
    kv = batch_mm(ks.transpose(-1, -2), vi)
    prior_ep = F.pad(ep[:, :, :-1], (1, 0))
    ci = torch.arange(blocks, device=q.device)
    cross = (
        (prior_ep[:, :, :, None] - ep[:, :, None, :])
        .masked_fill(~(ci[:, None] > ci[None, :]), float("-inf"))
        .exp()
    )
    prior = batch_mm(cross, kv.flatten(-2)).reshape(B, H, blocks, N, P)
    inter = batch_mm(qi * cs.exp().repeat_interleave(R, -1)[:, :, :, :, None], prior)
    y = (inter + intra).reshape(B, H, T + pad, R, P)[:, :, :T] - correction
    y = (
        y + skip[None, :, None, None, None] * v.reshape(B, H, T + pad, R, P)[:, :, :T]
    ) * F.silu(z)
    out = (y * outw[None, :, None]).sum(3)
    final = (
        (torch.exp(ep[:, :, -1, None] - ep)[:, :, :, None, None] * kv)
        .sum(2)
        .transpose(-1, -2)
        .contiguous()
    )
    return out, final


def _tensor_metadata(tensor, shape, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(name + ' must be a tensor')
    check_device(tensor)
    if tensor.dtype != torch.float32 or tuple(tensor.shape) != tuple(shape):
        raise ValueError(name + ' must be MPS FP32 with shape ' + str(tuple(shape)))

def _rms64(x, weight):
    # Use the same RMS law and existing learned scale as the full-sequence path.
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weight

def _rotate_siso(x, phase):
    # Match rotate(..., False): rotate adjacent pairs in the first
    # 32 of 64 state coordinates; the remaining 32 coordinates are unchanged.
    pairs = x[..., :32].reshape(*x.shape[:-1], 16, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    cs, sn = phase.cos()[:, :, None], phase.sin()[:, :, None]
    rotated = torch.stack((a * cs - b * sn, a * sn + b * cs), -1).flatten(-2)
    return torch.cat((rotated, x[..., 32:]), -1)

def torch_siso_step(self, u, cache):
    """FP32 SISO recurrence with phase rotation and trapezoidal state update.

    Input u is [batch,1,128]. Cache is (phase [B,4,16], state [B,4,P,N],
    previous_k [B,4,1,64], previous_v [B,4,1,64]), with P=N=64.
    No Tensor values are copied to the CPU, and there is no synchronization.
    """
    if torch.is_grad_enabled():
        raise RuntimeError('Decode is inference-only')
    check_device(u)
    if self.rank != 1:
        raise ValueError('The Torch step supports only SISO rank 1')
    if u.ndim != 3 or u.shape[1:] != (1, 128) or u.shape[0] < 1:
        raise ValueError('Input must have shape [positive batch,1,128]')
    batch = u.shape[0]
    _tensor_metadata(u, (batch, 1, 128), 'u')
    if not isinstance(cache, (tuple, list)) or len(cache) != 4:
        raise ValueError('Expected the original four-tensor Mixer cache')
    old_phase, old_state, previous_k, previous_v = cache
    for name, tensor, shape in (
        ('old_phase', old_phase, (batch, 4, 16)),
        ('old_state', old_state, (batch, 4, 64, 64)),
        ('previous_k', previous_k, (batch, 4, 1, 64)),
        ('previous_v', previous_v, (batch, 4, 1, 64)),
        ('in_proj.weight', self.in_proj.weight, (668, 128)),
        ('out_proj.weight', self.out_proj.weight, (128, 256)),
        ('dt_bias', self.dt_bias, (4,)),
        ('B_bias', self.B_bias, (4, 1, 64)),
        ('C_bias', self.C_bias, (4, 1, 64)),
        ('B_norm.weight', self.B_norm.weight, (64,)),
        ('C_norm.weight', self.C_norm.weight, (64,)),
        ('D', self.D, (4,)),
        ('mimo_x', self.mimo_x, (4, 1, 64)),
        ('mimo_z', self.mimo_z, (4, 1, 64)),
        ('mimo_o', self.mimo_o, (4, 1, 64)),
    ):
        _tensor_metadata(tensor, shape, name)

    z_raw, v_raw, k_raw, q_raw, dd, aa, tr, ang = self.in_proj(u).split(
        (256, 256, 64, 64, 4, 4, 4, 16), -1
    )
    dt = F.softplus(dd[:, 0] + self.dt_bias)
    decay = (-(aa[:, 0].clamp_min(0) +
               torch.reciprocal(1 - aa[:, 0].clamp_max(0)))).clamp_max(-1e-4)
    trap = tr[:, 0].sigmoid()
    phase = old_phase + torch.tanh(ang[:, 0])[:, None] * math.pi * dt[:, :, None]

    k = _rms64(k_raw[:, 0], self.B_norm.weight)[:, None, None] + self.B_bias[None]
    q = _rms64(q_raw[:, 0], self.C_norm.weight)[:, None, None] + self.C_bias[None]
    k, q = _rotate_siso(k, phase), _rotate_siso(q, phase)
    v = v_raw[:, 0].reshape(batch, 4, 1, 64) * self.mimo_x[None]
    z = z_raw[:, 0].reshape(batch, 4, 1, 64) * self.mimo_z[None]

    alpha = (dt * decay).exp()
    beta = alpha * (1 - trap) * dt
    gamma = trap * dt
    previous_outer = previous_v[:, :, 0, :, None] * previous_k[:, :, 0, None, :]
    current_outer = v[:, :, 0, :, None] * k[:, :, 0, None, :]
    state = (alpha[:, :, None, None] * old_state +
             beta[:, :, None, None] * previous_outer +
             gamma[:, :, None, None] * current_outer)

    readout = torch.matmul(state, q[:, :, 0, :, None]).squeeze(-1)
    out = (readout + self.D[None, :, None] * v[:, :, 0])
    out = out * F.silu(z[:, :, 0]) * self.mimo_o[None, :, 0]
    y = self.out_proj(out.reshape(batch, 1, 256))
    return y, (phase.contiguous(), state.contiguous(), k.contiguous(), v.contiguous())


def _owned_output(module, inputs, output):
    if torch.is_grad_enabled():
        return output
    return output.clone(memory_format=torch.contiguous_format)


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = 1
        self.chunk = 32
        self.in_proj = nn.Linear(128, 668, bias=False)
        self.in_proj.register_forward_hook(_owned_output)
        self.out_proj = nn.Linear(256, 128, bias=False)
        dt = torch.exp(
            torch.rand(4) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.B_bias = nn.Parameter(torch.ones(4, 1, 64))
        self.C_bias = nn.Parameter(torch.ones(4, 1, 64))
        self.B_norm = RMS(64)
        self.C_norm = RMS(64)
        self.D = nn.Parameter(torch.ones(4))
        self.D._no_weight_decay = True
        self.register_buffer("mimo_x", torch.ones(4, 1, 64))
        self.register_buffer("mimo_z", torch.ones(4, 1, 64))
        self.register_buffer("mimo_o", torch.ones(4, 1, 64))

    def project(self, u):
        B, T, _ = u.shape
        z, x, k, q, dd, aa, tr, ang = self.in_proj(u).split(
            (256, 256, 64, 64, 4, 4, 4, 16), -1
        )
        dt = F.softplus(dd + self.dt_bias).transpose(1, 2)
        a = (
            (-(aa.clamp_min(0) + torch.reciprocal(1 - aa.clamp_max(0))))
            .clamp_max(-1e-4)
            .transpose(1, 2)
        )
        adt = a * dt
        trap = tr.sigmoid().transpose(1, 2)
        k = self.B_norm(k.reshape(B, T, 1, 64))[:, None] + self.B_bias[None, :, None]
        q = self.C_norm(q.reshape(B, T, 1, 64))[:, None] + self.C_bias[None, :, None]
        phase = (torch.tanh(ang)[:, None] * math.pi * dt[:, :, :, None]).cumsum(2)
        q, k = rotate(q, phase, False), rotate(k, phase, False)
        v = x.reshape(B, T, 4, 64).permute(0, 2, 1, 3)[:, :, :, None]
        gate = z.reshape(B, T, 4, 64).permute(0, 2, 1, 3)[:, :, :, None]
        return q, k, v, adt, dt, trap, gate, phase

    def forward(self, u):
        q, k, v, a, d, t, z, angle = self.project(u)
        out, state = chunk_scan(q, k, v, a, d, t, z, self.D, self.mimo_o, self.chunk)
        cache = (
            angle[:, :, -1].contiguous(),
            state,
            k[:, :, -1].contiguous(),
            v[:, :, -1].contiguous(),
        )
        return (
            self.out_proj(out.transpose(1, 2).reshape(u.shape[0], u.shape[1], 256)),
            cache,
        )

    def step(self, u, cache):
        return torch_siso_step(self, u, cache)


class MambaBlock(nn.Module):
    def __init__(self, ff):
        super().__init__()
        self.norm1, self.norm2 = RMS(128), RMS(128)
        self.mixer = Mixer()
        self.mlp = SwiGLU(128, ff)

    def forward(self, x, cache=None, last_only=False):
        y, cache = (
            self.mixer(self.norm1(x))
            if cache is None
            else self.mixer.step(self.norm1(x), cache)
        )
        if last_only:
            x, y = x[:, -1:], y[:, -1:]
        x = x + y
        return x + self.mlp(self.norm2(x)), cache
