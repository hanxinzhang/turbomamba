# Third-party notices

## Mamba-3

The Mamba-3 parameterization and recurrent law in `mamba_mps.py`
are adapted from the state-spaces/mamba project.

- Source: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py
- Upstream module notice: **Copyright (c) 2026, Dao AI Lab, Goombalab.**
- Project license notice: **Copyright 2023 Tri Dao, Albert Gu.**
- License: Apache License, Version 2.0; full text in `LICENSES/mamba-Apache-2.0.txt`.

Modifications use fixed width 128, state size 64, SISO rank 1, FP32 real-valued
phase rotations, matrix SSD training/prefill in Torch and a Torch MPS SISO
recurrent step. RMS and SwiGLU use ordinary Torch operations. Official
CUDA/Triton/TileLang/CuTe kernels are not included.

## FineWeb-Edu and tokenizer

The bundled tokenizer was fitted using public FineWeb-Edu TRAIN documents.
No document text, evaluation bank or trained neural-model weights are bundled.

- Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- Source subset: `sample/10BT`, English Common Crawl educational text.
- Dataset license: ODC-BY 1.0, https://opendatacommons.org/licenses/by/1-0/
- Tokenizer: byte-BPE, 4096 IDs, four special tokens; fitted using TRAIN only.
- Tokenizer SHA256: `f31257090f13686b47cc0f8f33381e0cde3dffa35f33e9b956747651816659ca`

`run.py prepare` cites the public dataset and records source revision, tokenizer
identity and resulting demo training-bank identity in its manifest. Consumers
should retain the dataset attribution when distributing derived data artifacts.

PyTorch, NumPy, Tokenizers and Datasets are dependencies installed separately;
their source code is not vendored.
