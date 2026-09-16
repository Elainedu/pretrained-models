# pretrained-models

Ready-to-run Gradio demo for the BLOOMZ-396M Chinese dialogue model.

## Overview

This repository packages the pretrained `YeungNLP/bloomz-396m-zh` model
together with a small Gradio interface so it can be launched with a single
command. It is intended as a lightweight baseline for Chinese dialogue
generation and as a reference point for comparison with fine-tuned variants
elsewhere in the project (e.g. QLoRA and multi-epoch experiments).

## Model / Approach

- Base model: [`YeungNLP/bloomz-396m-zh`](https://huggingface.co/YeungNLP/bloomz-396m-zh)
- Family: BLOOM / BLOOMZ (instruction-tuned BLOOM, Chinese-optimised port)
- Architecture: Causal Language Model
- Parameters: 396 M (30 layers, 16 attention heads, hidden size 1024)
- Vocabulary size: 46,145
- Weights: `model.safetensors` (~699 MB, FP16)
- Max context length: 2048 tokens

No additional fine-tuning is performed - the demo loads the upstream weights
as-is and exposes standard sampling controls.

## Requirements

Minimum working set (see `requirements.txt`):

```
transformers>=4.30.0
torch>=2.0.0
gradio>=4.0.0
accelerate>=0.20.0
```

Recommended environment:

- Python 3.10+
- CUDA 11.8+ if using GPU
- GPU with ~1 GB VRAM in FP16 (T4 / RTX 2060 or better) - CPU also works but
  generates at only a few tokens/second

Install:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Elainedu/pretrained-models.git
cd pretrained-models

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure model.safetensors is present in the repo root
#    (see "Notes" below if it is missing).

# 4. Launch the Gradio demo
python gradio.py
```

The UI starts on `http://localhost:7860`.

## Project Structure

```
pretrained-models/
├── gradio.py                # Gradio chat UI, loads local BLOOMZ-396M
├── model.safetensors        # Model weights (~699 MB, not always in git)
├── config.json              # HF model configuration
├── generation_config.json   # Default generation parameters
├── requirements.txt
└── README.md
```

## Gradio UI

The interface loads the model from the repo root (`model_path = "."`) and
provides:

- A prompt textbox and a "Generate" button
- Sliders for the standard sampling parameters
- Preset example prompts (self-introduction, deep learning explanation,
  Python learning tips, book recommendations)

Default parameter ranges:

| Parameter          | Range      | Default |
| ------------------ | ---------- | ------- |
| max_new_tokens     | 32 - 512   | 128     |
| temperature        | 0.1 - 1.5  | 0.7     |
| top_p              | 0.1 - 1.0  | 0.9     |
| top_k              | 10 - 100   | 50      |

Suggested temperature settings:

- `0.3 - 0.5` for factual Q&A or code-like output
- `0.6 - 0.8` for general conversation
- `0.9 - 1.2` for creative writing

## Programmatic Use

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "."
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "What is artificial intelligence?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7,
                        top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Performance

Rough inference speed for a single stream:

| Hardware   | Throughput           |
| ---------- | -------------------- |
| RTX 3090   | ~50 tokens/sec       |
| T4         | ~30 tokens/sec       |
| CPU (12c)  | ~2 - 5 tokens/sec    |

Memory footprint:

- GPU FP16: ~800 MB VRAM
- GPU FP32: ~1.5 GB VRAM
- CPU: ~2 GB RAM

## Notes

- **`model.safetensors` may be missing from a fresh clone** because the file
  is ~699 MB and can be filtered out by git-lfs settings or `.gitignore`.
  If it is not present, download it from HuggingFace:

  ```bash
  # Option A: huggingface-cli
  huggingface-cli download YeungNLP/bloomz-396m-zh \
      model.safetensors config.json generation_config.json \
      --local-dir . --local-dir-use-symlinks False

  # Option B: git clone the HF repo, then copy the three files across
  git clone https://huggingface.co/YeungNLP/bloomz-396m-zh
  ```

- **Knowledge cutoff.** The model was pretrained on a fixed corpus and has no
  knowledge of events after its training date.
- **Language.** Primarily Simplified Chinese; Traditional Chinese support is
  limited unless you add an OpenCC conversion layer.
- **Model size.** At 396 M parameters this is a small model - responses may
  be shallow or occasionally inaccurate. Treat outputs as demonstrative
  rather than authoritative.

## License

- Model weights: follow the [BLOOM RAIL License](https://huggingface.co/spaces/bigscience/license)
  of the upstream `YeungNLP/bloomz-396m-zh` release.
- Demo code (this repository): educational use. Free to modify and reuse for
  teaching and research; not intended for commercial deployment.
