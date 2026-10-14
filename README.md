# TurboMamba

TurboMamba is a language-model architecture for faster inference, faster
training and better token prediction. It combines a global Mamba-3 model
that processes token blocks, a small causal Transformer and recent-token memory.

## How it works

```text
B completed tokens → block projection → global Mamba stack
                                                 ↓ one context prefix
previous-block last token + in-block tokens → local Transformer stack
                                                 ↓ residual lexical head
                                                 ↓ recent-count mixture
                                            next-token probability
```

The global stack updates once per B-token block, reducing global work.
Tokens within each block are still generated sequentially. Three mechanisms
strengthen local prediction:

- **Previous-token bridge:** the preceding block's final raw token enters the new local block directly; subsequent predictions use the earlier observed tokens in that block.
- **Residual lexical head:** a d→r→d correction refines the tied vocabulary head.
- **Bounded recent memory:** counts of the last W observed tokens mix with
  the vocabulary distribution, using α = 0.5 × sigmoid(wᵀh + b), with a learned
  linear projection of the local hidden state.

This memory uses the standard unigram/count-cache distribution.
The architecture couples these mechanisms to hierarchical
prediction and persistent recurrent/local caches.

## Small-scale experiment

The manuscript compares twelve implementations with comparable parameter counts
on public FineWeb-Edu, using the same tokenizer, three seeds and shared complete
training-budget caps on Apple MPS. Approximately 3M parameters describe this
experiment, and width, depth and block size can be varied in the architecture.
This repository implements the fixed reference configuration used in that
experiment: B=4, d=128, r=64, W=128, six global Mamba-3 layers and
two local Transformer layers.

The Jamba and Zamba baselines retain their hybrid designs and use Mamba-3 in
place of their original Mamba modules. TurboMamba reduces global work by
processing blocks and improves the local prediction path with lexical memory.

| Compared with | Mean test NLL reduction | End-to-end training throughput | End-to-end batch latency speedup |
|---|---:|---:|---:|
| Mamba-3 SISO | **9.4%** | **2.60×** | **3.12×** |
| Jamba (Mamba-3 adaptation) | **7.3%** | **2.81×** | **3.53×** |
| Zamba (Mamba-3 adaptation) | **18.3%** | **4.82×** | **6.21×** |

NLL reductions compare mean test prediction loss. Training throughput is the
TurboMamba/baseline ratio for complete training jobs, including startup,
development scoring, checkpointing and exit. Inference speedup is the
baseline/TurboMamba batch latency ratio for 32 sequences, each with a 256-token
prompt and 128 generated tokens. Both timing metrics include synchronization.
Results use three seeds on Apple MPS FP32. `results.csv` contains all twelve
models, generated-text diagnostics and both workloads at five batch sizes.

## Run

Use Python 3.11, Apple silicon and PyTorch with Apple MPS support
(experiment environment: PyTorch 2.13.0). All neural computation is FP32 on MPS;
text processing runs on CPU.

```bash
python -m pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=0
python run.py prepare --docs 128 --data-dir data
python run.py train --data data/train.npz --steps 100 --batch 16 --lr 0.001 --out-dir out
python run.py infer --checkpoint out/checkpoint.pt --prompt "Scientific research"
python run.py verify
```

`prepare` streams a bounded TRAIN-only public
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) demo,
with a pinned revision and within-document 256-label windows. The bundled
4096-token byte-BPE tokenizer was fitted on public TRAIN text. This demo differs
from the manuscript's frozen comparison sample.

Training uses AdamW, ten-update warmup, constant LR and checkpoints containing
optimizer/RNG/window-schedule state. Resume with `--resume out/checkpoint.pt`
and a larger total `--steps`. Inference accepts empty and partial-block prompts
and commits the final generated token. `verify` checks MPS full/cached law,
causality, normalization and gradients. It also checks actual 128-token sampled
histories and final commits at batches 1 and 32; outputs go to ignored `out/`.

## Code

`model.py` contains the architecture; `mamba_mps.py` contains the Torch MPS
layers and recurrent step; `run.py` contains the commands. Training uses Torch
autograd, cached inference uses the FP32 Torch SISO recurrence, and sampling
uses Torch CDF/top-p. This is a Torch MPS reference implementation.

The package preserves the candidate's weight layout and complete-block law.
The twelve-model comparison orchestration is a separate experimental artifact.
Upstream attribution and Apache-2.0 terms are in `THIRD_PARTY_NOTICES.md` and
`LICENSES/`; `LICENSE_SCOPE.md` records the draft's project-license boundary.
