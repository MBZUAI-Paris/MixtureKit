<h1>MixtureKit  🚀
<img alt='Leeroo logo' src='https://github.com/MBZUAI-Paris/MixtureKit/blob/main/logo.png' width='220' align='right' />
</h1>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-green.svg)](#python)

`MixtureKit` package provides high-level helper functions to merge pretrained and finetuned models
into a unified framework, integrating Mixture-of-Experts (MoE) advanced architectures.

## ✨ Highlights

* **One‑line merge** of multiple HF checkpoints into a single MoE model.
* Supports **Branch‑Train‑MiX (BTX)**, **Branch‑Train‑Stitch (BTS)** and *vanilla* MoE.
* Built‑in **routing visualizer**: inspect which tokens each expert receives — overall (coarse‑grained) and per layer (fine‑grained). See [`examples/README_vis.md`](examples/README_vis.md) for details.
---

## Installation

```bash

# Create a fresh conda environment (recommended)
conda create -n mixturekit python=3.12
conda activate mixturekit

# clone & install in editable mode for development
git clone https://github.com/MBZUAI-Paris/MixtureKit
or Download the repo zip file if the git clone does not work
cd MixtureKit
pip install -e .
```

> **Requirements**: Python ≥ 3.10 · PyTorch ≥ 2.5. The correct version of `transformers` is pulled automatically.

---

## Quick start

The script below builds a **BTX** MoE that routes tokens between a Gemma‑4B base model and two specialized fine-tuned experts (FrenchGemma for *French Language* and MedGemma for *Health Information*). For BTS or vanilla architectures, change the `moe_method` to **BTS** and **traditional** respectively. For other model families, comment the `model_cls`.

```bash
# From the repo root
python examples/example_build_moe.py
```

<details>
<summary>What happens under the hood?</summary>

1. A config dictionary is created that lists the base expert, two additional experts, the routing layers, etc.
2. `MixtureKit.build_moe()` merges the checkpoints and writes the MoE to `models_merge/gemmax/`.
3. The script reloads the model with `AutoModelForCausalLM` and prints a parameter‑breakdown table — only router weights stay trainable.

</details>

---
🔧 Fine-tune / Supervised-Fine-Tuning (SFT)

The **`mixture_training/`** folder contains a ready-to-go scaffold that trains
any merged MoE checkpoint with LoRA-adapters (BTX *or* BTS).

```
mixture_training/
├── config_training.yaml      # all hyper-params in one place
├── deepspeed_config.yaml     # ZeRO-3 config
├── requirements.txt          # extra libs (trl, deepspeed, wandb, etc.)
└── train_model.py            # launch-script
```

### 1. Prepare your data

* Expected format: 🤗 `datasets` arrow table saved on disk and loaded with
  `load_from_disk()`.
* `config_training.yaml` assumes:
  - a column called **`messages`** (list of chat turns),
  - each turn is a dict `{"role": "...", "content": "..."}`
    (same schema as *ShareGPT*).

### 3. Edit **`config_training.yaml`**

Minimal edits for your own run:


| Key            | What it does                                                       |
| -------------- | ------------------------------------------------------------------ |
| `dataset_path` | Path to the dataset produced in step 2                             |
| `model_id`     | Path or HF-Hub id of the**merged MoE** (e.g. `models_merge/gemmax`) |
| `output_dir`   | Where to write checkpoints / LoRA adapters                         |
| `run_name`     | Friendly name shown in 🤗`wandb` / logs                            |


### 4. Launch single-node training

```bash
accelerate launch --config_file mixture_training/deepspeed_config.yaml mixture_training/train_model.py
```

The script will:

1. Load the MoE checkpoint in **bf16** with distributed training if multi GPUs are available,
2. Train with 🤗 `trl`’s **`SFTTrainer`**,
3. Save incremental checkpoints to the local directory `output_dir`.

> **Tip:** To switch from *BTX*/*Traditional* to *BTS* finetuning, open
> `config_training.yaml` and set `is_bts` to `True`.


## Example gallery

The file [`examples/config_examples.txt`](examples/config_examples.txt) contains more ready‑to‑use configs. Copy one into a small script and call `build_moe()`.

| Key        | Scenario                  | MoE flavour               |
| ---------- | ------------------------- | ------------------------- |
| `llama3x`  | Two Llama‑3‑1B experts    | **BTX**                       |
| `qwen3x`   | Three Qwen‑3‑0.6B experts | **Traditional** MoE           |
| `gemmabts` | Gemma + 2 Gemma-experts   | **BTS** (layer‑stitching) |

---

## Documentation

* **API reference** — open `docs/index.html` or visit the online version.
---

## Contributing 🤝

Pull requests are welcome! Please open an issue first to discuss your ideas.

---

## License

MixtureKit is released under the **BSD 3-Clause** License — see the [LICENSE](./LICENSE) file for details.

---

*Happy mixing!* 🎛️
