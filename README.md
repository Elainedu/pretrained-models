# pretrained-models

Ready-to-run Gradio demo for the **BLOOMZ-396M Chinese** dialogue model
(`YeungNLP/bloomz-396m-zh`). Serves as an unmodified baseline for the
other fine-tuning repos in this project (QLoRA, full SFT, ABSA, multi-epoch).

---

## Table of Contents

- [What This Does](#what-this-does)
- [Model & Dataset](#model--dataset)
- [Training Setup](#training-setup)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Key Files](#key-files)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Reproducing Training](#reproducing-training)
- [Running Inference](#running-inference)
- [Notes](#notes)
- [Files Not in Repo](#files-not-in-repo)
- [References](#references)
- [License](#license)

---

## What This Does

This repository packages the upstream pretrained BLOOMZ-396M-ZH model
plus a small Gradio interface so it can be launched in a single command.
No additional fine-tuning happens here — the demo loads the HuggingFace
weights as-is and exposes standard sampling controls. It is intended as a
**baseline** for A/B comparisons against the fine-tuned checkpoints in the
sibling repos (QLoRA / full-SFT / multi-epoch / ABSA).

---

## Model & Dataset

- **Base model:** [`YeungNLP/bloomz-396m-zh`](https://huggingface.co/YeungNLP/bloomz-396m-zh)
- **Family:** BLOOM / BLOOMZ (instruction-tuned BLOOM, Chinese-optimised
  port by YeungNLP)
- **Task:** causal language modelling (autoregressive generation)

Architecture (from `config.json` in this repo):

| Field                | Value          |
| -------------------- | -------------- |
| `model_type`         | `bloom`        |
| `architectures`      | `BloomForCausalLM` |
| `hidden_size`        | 1024           |
| `n_layer`            | 24             |
| `n_head`             | 16             |
| `seq_length` (max)   | 2048           |
| `vocab_size`         | 46,145         |
| `torch_dtype`        | `float16`      |
| `bos / eos / pad / unk` | 1 / 2 / 3 / 0 |
| Approx. params       | ~396 M         |

**No training dataset is included** — this repo just serves the released
weights. The upstream BLOOMZ instruction tuning uses the xP3 multitask
prompt collection (see References).

---

## Training Setup

**Not applicable** — nothing is trained in this repo. See the sibling
repos for actual fine-tuning setups:

| Repo                              | Method                             |
| --------------------------------- | ---------------------------------- |
| `dialog-multi-epoch-experiments`  | Full SFT, 2 / 3 / 5+5 epochs       |
| `dialog-finetuning-50k`           | Full SFT on 50K dialog turns       |
| `qlora-dialog-100k`               | QLoRA (4-bit NF4 + LoRA r=8)       |
| `sentiment-analysis-absa`         | Full SFT for ABSA on hotel reviews |

Default **generation** parameters exposed by the Gradio UI:

| Parameter        | Range       | Default |
| ---------------- | ----------- | ------- |
| `max_new_tokens` | 32 – 512    | 128     |
| `temperature`    | 0.1 – 1.5   | 0.7     |
| `top_p`          | 0.1 – 1.0   | 0.9     |
| `top_k`          | 10 – 100    | 50      |

Suggested temperature bands:

- `0.3 – 0.5` for factual Q&A or code-like output
- `0.6 – 0.8` for general conversation
- `0.9 – 1.2` for creative writing

---

## System Architecture

```
                ┌────────────────────────────────┐
                │  Hugging Face Hub              │
                │  YeungNLP/bloomz-396m-zh       │
                │  (model.safetensors, ~699 MB)  │
                └───────────────┬────────────────┘
                                │  first-time download
                                ▼
                ┌────────────────────────────────┐
                │  Local repo root               │
                │  ├─ model.safetensors          │
                │  ├─ config.json                │
                │  └─ generation_config.json     │
                └───────────────┬────────────────┘
                                │
                                ▼
                ┌────────────────────────────────┐
                │  app.py (Gradio)               │
                │  - AutoTokenizer / AutoModelForCausalLM
                │  - device = cuda if available  │
                │  - torch_dtype = float16 on GPU│
                │  - preset examples + sliders   │
                └───────────────┬────────────────┘
                                │
                                ▼
                       http://localhost:7860
```

---

## Repository Layout

```
pretrained-models/
├── app.py                     # Gradio chat demo (entry point)
├── config.json                # HF model config
├── generation_config.json     # HF default generation config
├── requirements.txt           # minimal deps
├── README.md
└── model.safetensors          # ~699 MB weights (see "Files Not in Repo")
```

---

## Key Files

| File                       | Purpose |
| -------------------------- | ------- |
| `app.py`                   | Gradio front-end. Loads `AutoTokenizer` / `AutoModelForCausalLM` from `.` (the repo root), sets `torch_dtype=float16` on GPU, provides prompt input + sampling sliders + preset examples. |
| `config.json`              | Standard HF BLOOM config — needed for `AutoModel.from_pretrained(".")` to work when loading weights locally. |
| `generation_config.json`   | Default generation parameters (BOS/EOS/PAD token ids). |
| `requirements.txt`         | Minimum runtime deps: `torch`, `transformers`, `gradio`, `accelerate`. |

> **Note on `gradio.py` → `app.py` rename.** The docstring inside
> `app.py` still says `python gradio.py` for launch — that command name is
> historical. The actual entry point is `app.py`.

---

## Requirements

From `requirements.txt`:

```
transformers>=4.30.0
torch>=2.0.0
gradio>=4.0.0
accelerate>=0.20.0
```

Recommended environment:

- Python 3.10+
- CUDA 11.8+ if using a GPU
- **GPU (FP16):** ~800 MB VRAM (T4 / RTX 2060 or better)
- **CPU:** ~2 GB RAM, ~2–5 tokens/sec

Install:

```bash
pip install -r requirements.txt
```

---

## Configuration

There are no environment variables. The model path is hard-coded near the
top of `app.py`:

```python
model_path = "."   # loads config.json + model.safetensors from repo root
```

Change it to a HuggingFace model id (e.g. `"YeungNLP/bloomz-396m-zh"`) to
pull weights over the network instead of expecting a local copy.

---

## Reproducing Training

**N/A** — this repo doesn't train anything. To pretrain / fine-tune your
own version of BLOOMZ-396M, use one of the sibling repos:

- Full SFT on dialog:     `dialog-finetuning-50k`, `dialog-multi-epoch-experiments`
- QLoRA on 100K dialog:   `qlora-dialog-100k`
- Full SFT on ABSA:       `sentiment-analysis-absa`

---

## Running Inference

```bash
# 1. Clone
git clone https://github.com/Elainedu/pretrained-models.git
cd pretrained-models

# 2. Install deps
pip install -r requirements.txt

# 3. Make sure model.safetensors is present (see "Files Not in Repo")

# 4. Launch the Gradio demo
python app.py
```

The UI starts on `http://localhost:7860`.

### Programmatic use

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "."
tokenizer  = AutoTokenizer.from_pretrained(model_path)
model      = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "What is artificial intelligence?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out    = model.generate(**inputs, max_new_tokens=128, temperature=0.7,
                        top_p=0.9, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### Rough performance

| Hardware        | Throughput           |
| --------------- | -------------------- |
| RTX 3090        | ~50 tokens/sec       |
| T4              | ~30 tokens/sec       |
| CPU (12 cores)  | ~2 – 5 tokens/sec    |

Memory footprint:

- GPU FP16: ~800 MB VRAM
- GPU FP32: ~1.5 GB VRAM
- CPU:      ~2 GB RAM

---

## Notes

- **Language.** Primarily Simplified Chinese; add an OpenCC conversion
  layer if you need Traditional-Chinese output/input.
- **Model size.** At 396 M parameters this is a small model — responses
  may be shallow or occasionally inaccurate. Treat outputs as
  *demonstrative* rather than authoritative.
- **Knowledge cutoff.** The model was pretrained on a fixed corpus and has
  no knowledge of events after its training date.
- **Instruction tuning.** BLOOMZ was instruction-tuned on multi-task
  prompts (xP3); it responds better to instruction-style inputs than raw
  BLOOM.

---

## Files Not in Repo

| Excluded              | Why                        | How to obtain                          |
| --------------------- | -------------------------- | -------------------------------------- |
| `model.safetensors`   | ~699 MB binary weight file | Download from HuggingFace (below).     |
| `tokenizer.json`      | large tokenizer artefact   | Download from HuggingFace (below).     |

Download the weights + tokenizer from Hugging Face:

```bash
# Option A: huggingface-cli
huggingface-cli download YeungNLP/bloomz-396m-zh \
    model.safetensors config.json generation_config.json tokenizer.json \
    --local-dir . --local-dir-use-symlinks False

# Option B: git clone the HF repo, then copy the files across
git clone https://huggingface.co/YeungNLP/bloomz-396m-zh
```

---

## References

- Le Scao et al. **BLOOM: A 176B-Parameter Open-Access Multilingual
  Language Model.** arXiv:2211.05100. <https://arxiv.org/abs/2211.05100>
- Muennighoff et al. **Crosslingual Generalization through Multitask
  Finetuning (BLOOMZ / mT0).** arXiv:2211.01786.
  <https://arxiv.org/abs/2211.01786>
- Model card: <https://huggingface.co/YeungNLP/bloomz-396m-zh>
- Transformers `AutoModelForCausalLM`:
  <https://huggingface.co/docs/transformers/main/model_doc/auto>

---

## License

- **Model weights:** follow the
  [BLOOM RAIL License](https://huggingface.co/spaces/bigscience/license) of
  the upstream `YeungNLP/bloomz-396m-zh` release.
- **Demo code (this repository):** educational use. Free to modify and
  reuse for teaching and research; not intended for commercial deployment.
