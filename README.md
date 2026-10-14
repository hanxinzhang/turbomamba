# TurboMamba

TurboMamba is a **3,000,753-parameter** language model that combines a block-clock
Mamba-3 global model, a causal local Transformer and recent lexical memory.

## How it works

```text
four completed tokens → block projection → 6 global Mamba-3 layers
                                                 ↓ one context prefix
previous raw token embedding ───────────→ 2 local Transformer layers
                                                 ↓ residual lexical head
                                                 ↓ recent-count mixture
                                            next-token probability
```

The global stack updates once per four-token block, reducing global work.
Tokens within each block are still generated sequentially. Three mechanisms
strengthen local prediction:

- **Previous-token bridge:** the last raw token crosses block boundaries.
- **Residual lexical head:** a 128→64→128 correction refines the tied vocabulary head.
- **Bounded recent memory:** counts of the last 128 observed tokens mix with
  the vocabulary distribution, using α = 0.5 × sigmoid(wᵀh + b), with a learned
  linear projection of the local hidden state.

This memory uses the standard unigram/count-cache distribution.
The architecture couples these mechanisms to hierarchical
prediction and persistent recurrent/local caches.

## Measured results

The manuscript compares approximately 3M-parameter models on public FineWeb-Edu,
using three seeds and shared complete training-budget caps on Apple MPS.

| Measurement | TurboMamba vs token-clock Mamba-3 adaptation |
|---|---:|
| Mean held-out NLL (lower is better) | **5.000 vs 5.515** |
| Complete-worker training throughput | **2.54×** |
| Full inference, B32 / context256 / generate128 | **2.31×** |

Results are three-seed means on Apple MPS FP32; `results.csv` reports all
model/workload/batch values.

## Run

Use Python 3.11, Apple silicon and PyTorch with `torch.mps.compile_shader`
(tested: PyTorch 2.13.0). All neural computation is FP32 on MPS; text processing
runs on CPU.

```bash
python -m pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=0
python run.py prepare --docs 128 --data-dir data
python run.py train --data data/train.npz --steps 100 --batch 16 --out-dir out
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
and commits the final generated token. `verify` checks GPU full/cached law,
causality, normalization, gradients and native kernels; outputs go to ignored `out/`.

## Code

`model.py` contains the architecture; `mamba_mps.py` contains the layers and
bindings; `run.py` contains the commands. `kernels.metal` provides the same
inference-only Mamba/RMS/SwiGLU path as the experiment; training uses Torch
autograd and sampling uses Torch CDF/top-p.

The package preserves the candidate's weight layout and complete-block law.
The eight-arm fairness orchestration is a separate experimental artifact.
Upstream attribution and Apache-2.0 terms are in `THIRD_PARTY_NOTICES.md` and
`LICENSES/`; `LICENSE_SCOPE.md` records the draft's project-license boundary.
