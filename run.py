#!/usr/bin/env python3
"""Portable public-data preparation, MPS training, generation, and verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
    raise RuntimeError(
        "Use PYTORCH_ENABLE_MPS_FALLBACK=0; CPU neural fallback is disabled"
    )
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

HERE = Path(__file__).resolve().parent
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def gpu():
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple MPS is required. No CPU/CUDA model fallback is implemented."
        )
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError(
            "torch.mps.compile_shader is required (tested with PyTorch 2.13)."
        )
    return torch


def host_key(url):
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    pieces = host.split(".")
    compound = {
        "co.uk",
        "org.uk",
        "ac.uk",
        "com.au",
        "edu.au",
        "org.au",
        "co.nz",
        "co.in",
        "com.br",
        "co.za",
        "com.cn",
    }
    return ".".join(pieces[-3:] if ".".join(pieces[-2:]) in compound else pieces[-2:])


def prepare(args):
    """Stream a bounded TRAIN-only demo; no held-out files are created/read."""
    import numpy as np
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    destination = args.data_dir
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "train.npz"
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing training bank")
    source = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        revision=REVISION,
        split="train",
        streaming=True,
    )
    windows, document_ids, seen, docs, scanned = [], [], set(), 0, 0
    for row in source:
        scanned += 1
        if scanned > args.scan_limit:
            break
        host = host_key(row.get("url", ""))
        bucket = (
            int(hashlib.sha256(("split-v1|" + host).encode()).hexdigest()[:16], 16)
            % 100
        )
        if not host or bucket < 20:
            continue
        text = row.get("text", "")
        text_hash = hashlib.sha256(
            " ".join(text.casefold().split()).encode()
        ).hexdigest()
        if text_hash in seen:
            continue
        seen.add(text_hash)
        tokens = [1] + tokenizer.encode(text).ids + [2]
        if len(tokens) < 256:
            continue
        # Each target window stays within one document. No document concatenation.
        document_id = hashlib.sha256(str(row.get("id", text_hash)).encode()).hexdigest()
        for offset in range(0, len(tokens) - 255, 256):
            windows.append(tokens[offset : offset + 256])
            document_ids.append(document_id)
        docs += 1
        if docs >= args.docs:
            break
    if not windows:
        raise RuntimeError("No eligible document windows found; increase --scan-limit")
    np.savez_compressed(
        output,
        X=np.asarray(windows, dtype=np.int64),
        document_ids=np.asarray(document_ids),
    )
    write_json(
        destination / "manifest.json",
        dict(
            dataset="HuggingFaceFW/fineweb-edu",
            config="sample-10BT",
            revision=REVISION,
            license="ODC-BY-1.0",
            tokenizer_sha256=sha(args.tokenizer),
            train_sha256=sha(output),
            scanned=scanned,
            documents=docs,
            windows=len(windows),
            target_length=256,
            split="TRAIN: host hash bucket >=20; DEV/TEST hosts excluded",
            scope="Bounded public demo; a different sample from the manuscript's frozen eight-arm comparison",
        ),
    )
    print(
        json.dumps(
            dict(documents=docs, windows=len(windows), labels=256 * len(windows))
        )
    )


def checkpoint(path, model, optimizer, args, update, order, cursor, rng, bank_hash):
    import torch
    from model import CONFIG

    payload = dict(
        format="turbomamba-tiny-v1",
        config=CONFIG,
        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        optimizer=optimizer.state_dict(),
        seed=args.seed,
        completed_updates=update,
        learning_rate=args.lr,
        batch=args.batch,
        train_bank_sha256=bank_hash,
        torch_rng=torch.get_rng_state(),
        mps_rng=torch.mps.get_rng_state(),
        numpy_rng=json.dumps(rng.bit_generator.state),
        order=torch.as_tensor(order.copy()),
        cursor=cursor,
    )
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    import numpy as np

    torch = gpu()
    from model import TurboMamba

    with np.load(args.data, allow_pickle=False) as source:
        bank = source["X"].copy()
    if bank.ndim != 2 or bank.shape[1] != 256 or len(bank) < 1:
        raise ValueError("Expected X[windows,256] within-document target windows")
    if bank.min() < 0 or bank.max() >= 4096:
        raise ValueError("Training IDs must be in the 4096-word vocabulary")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = TurboMamba(args.seed).train()
    # Matches the experiment: all parameters decay, including Mamba no_decay flags.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    rng = np.random.default_rng(args.seed)
    order, cursor, completed = rng.permutation(len(bank)), 0, 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["format"] != "turbomamba-tiny-v1" or saved["train_bank_sha256"] != sha(
            args.data
        ):
            raise ValueError("Checkpoint format or training-bank identity mismatch")
        if (saved["seed"], saved["learning_rate"], saved["batch"]) != (
            args.seed,
            args.lr,
            args.batch,
        ):
            raise ValueError("Resume requires the same seed, learning rate and batch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.mps.set_rng_state(saved["mps_rng"])
        rng.bit_generator.state = json.loads(saved["numpy_rng"])
        order, cursor = saved["order"].numpy(), saved["cursor"]
        completed = saved["completed_updates"]
        if args.steps < completed:
            raise ValueError(
                "--steps is the total update count and must not precede the saved count"
            )
    elapsed_updates, history = 0.0, []
    started = time.monotonic()
    for update in range(completed, args.steps):
        indices = []
        while len(indices) < args.batch:
            take = min(args.batch - len(indices), len(order) - cursor)
            indices.extend(order[cursor : cursor + take])
            cursor += take
            if cursor == len(order):
                order, cursor = rng.permutation(len(bank)), 0
        targets = torch.from_numpy(bank[indices].copy()).to("mps", dtype=torch.int64)
        learning_rate = args.lr * min(1.0, (update + 1) / 10.0)
        optimizer.param_groups[0]["lr"] = learning_rate
        torch.mps.synchronize()
        before = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = model(targets)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.mps.synchronize()
        elapsed_updates += time.monotonic() - before
        row = dict(
            update=update + 1,
            loss=float(loss.detach().cpu()),
            lr=learning_rate,
            unclipped_gradient_norm=float(norm.detach().cpu()),
        )
        history.append(row)
        if update == completed or (update + 1) % 10 == 0 or update + 1 == args.steps:
            print(json.dumps(row), flush=True)
    checkpoint(
        args.out_dir / "checkpoint.pt",
        model,
        optimizer,
        args,
        args.steps,
        order,
        cursor,
        rng,
        sha(args.data),
    )
    torch.mps.synchronize()
    labels = max(0, args.steps - completed) * args.batch * 256
    write_json(
        args.out_dir / "training.json",
        dict(
            device="mps",
            dtype="float32",
            fallback=0,
            torch_version=torch.__version__,
            parameters=sum(p.numel() for p in model.parameters()),
            labels=labels,
            new_updates=max(0, args.steps - completed),
            total_completed_updates=args.steps,
            synchronized_update_seconds=elapsed_updates,
            labels_per_update_second=(
                labels / elapsed_updates if elapsed_updates else None
            ),
            train_and_save_seconds=time.monotonic() - started,
            history=history,
            scope="Demo steps, no held-out scoring; does not reproduce the paper's matched complete-worker budgets",
        ),
    )


def infer(args):
    import numpy as np
    from tokenizers import Tokenizer

    torch = gpu()
    from model import CONFIG, TurboMamba

    model = TurboMamba(args.seed).eval()
    saved = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if saved.get("format") != "turbomamba-tiny-v1" or saved.get("config") != CONFIG:
        raise ValueError("Unsupported checkpoint configuration")
    model.load_state_dict(saved["model"], strict=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    ids = tokenizer.encode(args.prompt).ids
    prompt = torch.tensor([ids], dtype=torch.int64, device="mps")
    uniforms = (
        np.random.default_rng(args.seed).random((1, args.tokens)).astype(np.float32)
    )
    result = model.generate(
        prompt,
        torch.from_numpy(uniforms).to("mps"),
        temperature=args.temperature,
        top_p=args.top_p,
    )
    torch.mps.synchronize()
    words = result["words"].cpu().numpy()[0].tolist()
    print(tokenizer.decode(ids + words, skip_special_tokens=False))
    print(
        json.dumps(
            dict(
                prompt_tokens=len(ids),
                generated_tokens=len(words),
                committed=result["state"]["committed"],
            )
        )
    )


def verify(args):
    """Finite GPU checks, including partial prompts, rolling memory and gradients."""
    import numpy as np

    torch = gpu()
    from model import TurboMamba
    from mamba_mps import native_gate, native_rms

    model = TurboMamba(args.seed)
    rng = np.random.default_rng(args.seed)
    # Repeated labels exercise the count mixture; >128 labels exercise eviction.
    tokens = torch.from_numpy(rng.integers(0, 64, (2, 144), dtype=np.int64)).to("mps")
    arrays, results = {}, {}
    with torch.inference_mode():
        full = model.factors(tokens, observed_only=False)
        lp = full["lp"]
        selected = lp.gather(-1, tokens[..., None]).squeeze(-1)
        results["selected_vs_full_max"] = float(
            (selected - full["selected"]).abs().max().cpu()
        )
        results["normalization_max"] = float((lp.exp().sum(-1) - 1).abs().max().cpu())
        cases = []
        for context, generated in ((0, 9), (3, 7), (5, 7), (16, 8), (127, 9), (128, 9)):
            state = model.prefill(tokens[:, :context])
            heads = []
            for offset in range(generated):
                head = model.probabilities(state)
                if model.probabilities(state) is not head:
                    raise AssertionError("probabilities must be memoized until commit")
                heads.append(head)
                model.commit(state, tokens[:, context + offset])
            cached = torch.stack(heads, 1)
            error = float(
                (cached - lp[:, context : context + generated]).abs().max().cpu()
            )
            final = context + generated
            if (
                state["committed"] != final
                or state["phase"] != final % 4
                or len(state["pending"]) != final % 4
                or state["global_position"] != final // 4 + 1
            ):
                raise AssertionError("Final commit/block-clock state mismatch")
            if state["ring"].shape[1] != min(128, final):
                raise AssertionError("Rolling history length mismatch")
            cases.append(
                dict(
                    context=context,
                    generated=generated,
                    max_log_probability_error=error,
                )
            )
            arrays[f"C{context}_G{generated}_cached"] = cached.cpu().numpy()
        changed = tokens.clone()
        changed[:, 71:] = (changed[:, 71:] + 13) % 64
        causal = model.factors(changed, observed_only=False)["lp"][:, :72]
        results["causal_max"] = float((causal - lp[:, :72]).abs().max().cpu())
        # Position71's label changed, so only positions0..70 share their targets;
        # full probability law at71 remains identical because token71 is unseen.
        x = torch.from_numpy(rng.normal(size=(2, 5, 128)).astype(np.float32)).to("mps")
        weight = torch.from_numpy(rng.normal(size=128).astype(np.float32)).to("mps")
        reference = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weight
        results["native_rms_max"] = float(
            (native_rms(x, weight) - reference).abs().max().cpu()
        )
        z = torch.from_numpy(rng.normal(size=(2, 5, 1018)).astype(np.float32)).to("mps")
        a, b = z.chunk(2, -1)
        results["native_gate_max"] = float(
            (native_gate(z) - a * torch.nn.functional.silu(b)).abs().max().cpu()
        )
        arrays["full_log_probabilities"] = lp.cpu().numpy()
        arrays["tokens"] = tokens.cpu().numpy()
    model.train()
    small = tokens[:, :16]
    model.zero_grad(set_to_none=True)
    model.factors(small)["loss"].backward()
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    full_small = model.factors(small, observed_only=False)["lp"]
    loss = -full_small.gather(-1, small[..., None]).mean()
    loss.backward()
    errors = []
    for name, p in model.named_parameters():
        if (
            not torch.isfinite(p.grad).all()
            or not torch.isfinite(gradients[name]).all()
        ):
            raise AssertionError(f"Nonfinite gradients: {name}")
        errors.append((p.grad - gradients[name]).abs().max())
    results["gradient_max"] = float(torch.stack(errors).max().cpu())
    results["cases"] = cases
    guards = dict(
        selected_vs_full_max=3e-5,
        normalization_max=3e-5,
        causal_max=1e-5,
        native_rms_max=3e-5,
        native_gate_max=3e-5,
        gradient_max=3e-5,
    )
    if any(results[name] > bound for name, bound in guards.items()) or any(
        row["max_log_probability_error"] > 3e-4 for row in cases
    ):
        raise AssertionError(json.dumps(results))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "verification_arrays.npz", **arrays)
    torch.mps.synchronize()
    results.update(
        status="PASS_FINITE_MPS_CHECKS",
        device="mps",
        dtype="float32",
        fallback=0,
        torch_version=torch.__version__,
        parameters=sum(p.numel() for p in model.parameters()),
        arrays_sha256=sha(args.out_dir / "verification_arrays.npz"),
        guards=guards,
        scope="Finite mathematical/kernel consistency; no quality or performance benchmark",
    )
    write_json(args.out_dir / "verification.json", results)
    print(json.dumps(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "prepare", help="Stream a bounded public TRAIN-only demo (CPU text work)"
    )
    p.add_argument("--docs", type=int, default=128)
    p.add_argument("--scan-limit", type=int, default=2000)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--tokenizer", type=Path, default=HERE / "tokenizer.json")
    p.set_defaults(function=prepare)
    p = sub.add_parser(
        "train", help="Train only on MPS; constant LR after ten-step warmup"
    )
    p.add_argument("--data", type=Path, default=Path("data/train.npz"))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=9201)
    p.add_argument("--resume", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    p.set_defaults(function=train)
    p = sub.add_parser("infer", help="Incremental sampling from a package checkpoint")
    p.add_argument("--checkpoint", type=Path, default=Path("out/checkpoint.pt"))
    p.add_argument("--tokenizer", type=Path, default=HERE / "tokenizer.json")
    p.add_argument("--prompt", default="The purpose of scientific research is")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=9201)
    p.set_defaults(function=infer)
    p = sub.add_parser(
        "verify", help="GPU-only full/cached/gradient/kernel consistency"
    )
    p.add_argument("--seed", type=int, default=9201)
    p.add_argument("--out-dir", type=Path, default=Path("out/verification"))
    p.set_defaults(function=verify)
    args = parser.parse_args()
    for name in ("steps", "batch", "docs", "scan_limit"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "tokens") and args.tokens < 0:
        parser.error("--tokens must be nonnegative")
    args.function(args)


if __name__ == "__main__":
    main()
