"""TurboMamba: a block-clock Mamba-3 global model and causal local decoder.

The fixed default is the 3,000,753-parameter configuration in the manuscript.
State belongs to a request. Never share a request's mutable state across callers.
"""

import torch
from torch import nn
from torch.nn import functional as F

from mamba_mps import (
    D,
    V,
    RMS,
    AttentionBlock,
    MambaBlock,
    check_device,
    rope_geometry,
    sample,
    stable_init,
)

CONFIG = dict(
    width=128,
    vocabulary=4096,
    block=4,
    global_layers=6,
    local_layers=2,
    ff=509,
    prefix=1,
    recent_tokens=128,
    lexical_rank=64,
    copy_ceiling=0.5,
)


class TurboMamba(nn.Module):
    """Fixed tiny configuration; full-sequence training and cached decoding."""

    def __init__(self, seed=9201):
        super().__init__()
        torch.manual_seed(seed)
        self.word = nn.Embedding(V, D)
        self.bos = nn.Parameter(torch.empty(D))
        self.block_encoder = nn.Linear(4 * D, D, bias=False)
        self.global_layers = nn.ModuleList(MambaBlock(509) for _ in range(6))
        self.global_norm = RMS(D)
        self.prefix = nn.Linear(D, D, bias=False)
        self.local_layers = nn.ModuleList(AttentionBlock(ff=509) for _ in range(2))
        self.local_norm = RMS(D)
        self.lexical = nn.Linear(D, V, bias=False)
        self.lexical.weight = self.word.weight
        self.head_norm = RMS(D)
        self.head_in = nn.Linear(D, 64, bias=False)
        self.head_out = nn.Linear(64, D, bias=False)
        self.copy_gate = nn.Linear(D, 1)
        stable_init(self, seed, depth=8)
        self.to(device="mps", dtype=torch.float32)

    def global_run(self, x, caches=None, last_only=False):
        updated = []
        for i, layer in enumerate(self.global_layers):
            x, cache = layer(
                x,
                None if caches is None else caches[i],
                last_only=last_only and i == len(self.global_layers) - 1,
            )
            updated.append(cache)
        return self.global_norm(x), updated

    def local_run(self, x, caches=None, start=0, last_only=False, geometry=None):
        if geometry is None:
            geometry = rope_geometry(8, 32, x.device)
        updated = []
        for i, layer in enumerate(self.local_layers):
            x, cache = layer(
                x,
                start,
                geometry,
                None if caches is None else caches[i],
                last_only=last_only and i == len(self.local_layers) - 1,
            )
            updated.append(cache)
        return self.local_norm(x), updated

    def logits(self, h):
        # Tied vocabulary head plus a low-rank residual in hidden space.
        h = h + self.head_out(F.gelu(self.head_in(self.head_norm(h))))
        return self.lexical(h)

    def forward(self, tokens):
        return self.factors(tokens)["loss"]

    def factors(self, tokens, observed_only=True):
        """Log probabilities of labels using only their strictly earlier history.

        tokens are targets, not already-shifted inputs. Every length must be a
        positive multiple of four. The first label is predicted from learned BOS.
        observed_only avoids materializing a full [batch,time,vocabulary] cache.
        """
        check_device(tokens)
        n, t = tokens.shape
        if not t or t % 4:
            raise ValueError(
                "Training targets require positive complete four-token blocks"
            )
        e = self.word(tokens)
        blocks = t // 4
        gx = torch.cat(
            (
                self.bos[None, None].expand(n, 1, D),
                self.block_encoder(e.reshape(n, blocks, 4 * D)),
            ),
            1,
        )
        gh, _ = self.global_run(gx)
        context = self.prefix(gh[:, :blocks]).reshape(n * blocks, 1, D)
        previous = torch.cat((self.bos[None, None].expand(n, 1, D), e[:, :-1]), 1)
        lx = torch.cat((context, previous.reshape(n * blocks, 4, D)), 1)
        h, _ = self.local_run(lx)
        h = h[:, 1:5].reshape(n, t, D)
        logits = self.logits(h)
        selected = -F.cross_entropy(
            logits.flatten(0, 1), tokens.flatten(), reduction="none"
        ).reshape(n, t)
        pos = torch.arange(t, device=tokens.device)
        legal = (pos[None] < pos[:, None]) & (pos[None] >= pos[:, None] - 128)
        denom = pos.clamp_max(128).clamp_min(1)
        mass = ((tokens[:, :, None] == tokens[:, None, :]) & legal[None]).sum(-1)
        mass = mass / denom[None]
        alpha = 0.5 * self.copy_gate(h).sigmoid().squeeze(-1)
        mixed = torch.logaddexp(
            selected + torch.log1p(-alpha), mass.log() + alpha.log()
        )
        selected = torch.where(pos[None] > 0, mixed, selected)
        result = dict(selected=selected, loss=-selected.mean())
        if not observed_only:
            bank = F.one_hot(tokens, V).float()
            cumulative = torch.cat(
                (torch.zeros((n, 1, V), device=tokens.device), bank.cumsum(1)), 1
            )
            counts = cumulative[:, :t] - cumulative.index_select(
                1, (pos - 128).clamp_min(0)
            )
            base = F.log_softmax(logits, -1)
            lp = torch.logaddexp(
                base + torch.log1p(-alpha)[..., None],
                (counts / denom[None, :, None]).log() + alpha.log()[..., None],
            )
            result["lp"] = torch.where(pos[None, :, None] > 0, lp, base)
        return result

    def prefill(self, prompt):
        """Consume any prompt length, including zero, without padding history.

        Complete blocks use parallel Mamba prefill; an unfinished final block is
        replayed through the same local cached law as generation. This extends
        the experiment's complete-block serving API, keeping its weights/law.
        """
        check_device(prompt)
        n, length = prompt.shape
        complete = length // 4 * 4
        e = self.word(prompt[:, :complete])
        gx = torch.cat(
            (
                self.bos[None, None].expand(n, 1, D),
                self.block_encoder(e.reshape(n, complete // 4, 4 * D)),
            ),
            1,
        )
        gh, caches = self.global_run(gx, last_only=True)
        ring = prompt[:, max(0, complete - 128) : complete].clone()
        counts = torch.zeros((n, V), device=prompt.device)
        counts.scatter_add_(1, ring, torch.ones_like(ring, dtype=torch.float32))
        state = dict(
            gh=gh,
            global_caches=caches,
            global_position=complete // 4 + 1,
            last=e[:, -1:] if complete else self.bos[None, None].expand(n, 1, D),
            phase=0,
            pending=[],
            local_caches=None,
            ring=ring,
            counts=counts,
            committed=complete,
            h=None,
            local_geometry=rope_geometry(8, 32, prompt.device),
            prepared=False,
        )
        for token in prompt[:, complete:].unbind(1):
            self.probabilities(state)
            self.commit(state, token)
        return state

    def probabilities(self, state):
        """Return next-token log probabilities; repeated calls are idempotent."""
        if state["prepared"]:
            return state["prepared_lp"]
        if state["phase"] == 0:
            lx = torch.cat((self.prefix(state["gh"]), state["last"]), 1)
            h, caches = self.local_run(
                lx, last_only=True, geometry=state["local_geometry"]
            )
            state["local_position"] = lx.shape[1]
        else:
            h, caches = self.local_run(
                state["last"],
                state["local_caches"],
                state["local_position"],
                last_only=True,
                geometry=state["local_geometry"],
            )
            state["local_position"] += 1
        state["local_caches"], state["h"] = caches, h[:, -1]
        lp = F.log_softmax(self.logits(h[:, -1]), -1)
        if state["ring"].shape[1]:
            alpha = 0.5 * self.copy_gate(h[:, -1]).sigmoid()
            lp = torch.logaddexp(
                lp + torch.log1p(-alpha),
                (state["counts"] / state["ring"].shape[1]).log() + alpha.log(),
            )
        state.update(prepared=True, prepared_lp=lp)
        return lp

    def commit(self, state, token):
        """Consume an observed/generated token, including the final output token."""
        if not state["prepared"]:
            raise RuntimeError("Call probabilities before committing a token")
        check_device(token)
        state["prepared"] = False
        state.pop("prepared_lp", None)
        e = self.word(token[:, None])
        state["pending"].append(e)
        state["last"] = e
        state["committed"] += 1
        if state["ring"].shape[1] == 128:
            state["counts"].scatter_add_(
                1,
                state["ring"][:, :1],
                -torch.ones_like(token[:, None], dtype=torch.float32),
            )
            state["ring"] = state["ring"][:, 1:]
        state["ring"] = torch.cat((state["ring"], token[:, None]), 1)
        state["counts"].scatter_add_(
            1, token[:, None], torch.ones_like(token[:, None], dtype=torch.float32)
        )
        state["phase"] = (state["phase"] + 1) % 4
        if state["phase"] == 0:
            gx = self.block_encoder(
                torch.cat(state["pending"], 1).reshape(len(token), 1, 4 * D)
            )
            gh, caches = self.global_run(gx, state["global_caches"], last_only=True)
            state.update(
                gh=gh,
                global_caches=caches,
                global_position=state["global_position"] + 1,
                pending=[],
                local_caches=None,
            )

    @torch.inference_mode()
    def generate(self, prompt, uniforms, temperature=1.0, top_p=1.0, trace=False):
        """Sequential ancestral sampling with bounded KV and recurrent state."""
        check_device(prompt)
        check_device(uniforms)
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature > 0 and 0 < top_p <= 1 are required")
        state = self.prefill(prompt)
        words, heads = [], []
        for uniform in uniforms.unbind(1):
            lp = self.probabilities(state)
            if trace:
                heads.append(lp)
            probability = (lp / temperature).softmax(-1)
            if top_p < 1:
                values, indices = probability.sort(-1, descending=True)
                values = values * (values.cumsum(-1) - values < top_p)
                probability = torch.zeros_like(probability).scatter(-1, indices, values)
            token = sample(probability, uniform)
            words.append(token)
            self.commit(state, token)
        result = torch.stack(words, 1) if words else prompt[:, :0]
        return dict(
            words=result, heads=torch.stack(heads, 1) if heads else None, state=state
        )
