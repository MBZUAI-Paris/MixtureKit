<h1>MixtureKit  🚀
<img alt='MixtureKit logo' src='logo.png' width='220' align='right' />
</h1>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-green.svg)](#python) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MBZUAI-Paris/MixtureKit/blob/main/examples/build_your_own_moe.ipynb)

`MixtureKit` package provides high-level helper functions to merge pretrained and finetuned models
into a unified framework, integrating Mixture-of-Experts (MoE) advanced architectures.

## ✨ Highlights

* **One‑line merge** of multiple HF checkpoints into a single MoE model.
* Supports **Branch‑Train‑MiX (BTX)**, **Branch‑Train‑Stitch (BTS)** and *vanilla* MoE.
* Built‑in **routing visualizer**: `mixturekit visualize MODEL` shows which experts every token uses, overall and per layer, as an app, a shareable HTML page or inline in notebooks. See [`examples/README_vis.md`](examples/README_vis.md).
---

## Installation

```bash
git clone https://github.com/MBZUAI-Paris/MixtureKit
cd MixtureKit
pip install -e .            # build and load merged models
pip install -e ".[train]"   # + the training scaffold (trl, peft, accelerate, ...)
pip install -e ".[vis]"     # + the routing visualizer (streamlit)
```

A zip download of the repository installs the same way.

> **Requirements**: Python ≥ 3.10 · PyTorch ≥ 2.5 · transformers 4.52 to 5.x.
> Every transformers minor release from 4.52 to 5.19 is tested. Merged models subclass the
> installed transformers classes, so they load with `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`
> without MixtureKit installed.

To run the tests (CPU, about 10 seconds, no download):

```bash
pip install -e ".[test]"
pytest
```

---

**Try it in 15 minutes on a free GPU:** the [Colab notebook](https://colab.research.google.com/github/MBZUAI-Paris/MixtureKit/blob/main/examples/build_your_own_moe.ipynb) merges two Qwen3-0.6B models, trains the routers and shows the routing.

## Command line

```bash
mixturekit build examples/configs/btx_gemma3.yaml     # merge the experts (YAML = build_moe config)
mixturekit train my_training.yaml                     # train the routers (model_id: models_merge/gemmax)
mixturekit visualize models_merge/gemmax              # see which expert every token uses
mixturekit push models_merge/gemmax my-org/gemmax     # upload it with a generated model card
```

`mixturekit build ... --push-to-hub my-org/name` builds and uploads in one go. The model card
lists the experts, the method, whether the routers are trained, how to load the model and the
configuration to rebuild it; `mixturekit card MODEL` writes it without uploading. More configs:
[`examples/configs/`](examples/configs).

---

## Quick start

The script below builds a **BTX** MoE that routes tokens between a Gemma‑4B base model and two specialized fine-tuned experts (FrenchGemma for *French Language* and MedGemma for *Health Information*). For BTS or vanilla architectures, change the `moe_method` to **BTS** and **traditional** respectively. The `model_cls` argument is optional: multimodal checkpoints such as Gemma 3 are merged through their language model (the vision tower is not kept), which gives a text-only MoE.

```bash
# From the repo root
python examples/example_build_moe.py
```

### Configuration keys

| Key | Meaning |
| --- | --- |
| `moe_method` | `btx`, `traditional` or `bts` |
| `model_type` | Name of the merged model, also its output folder `models_merge/<model_type>` |
| `experts` | List of `{"expert_name", "model_id"}`; with BTS the first one is the hub |
| `router_layers` | BTX: projections turned into routed experts, e.g. `mlp.gate_proj`, `self_attn.q_proj` |
| `router_layers_index` | Transformer blocks to convert, `[]` for all of them |
| `num_experts_per_tok` | Number of experts activated per token (top-k) |
| `alpha` | Weight of the load balancing loss (BTX, traditional). The loss is averaged over all the routers and equals 1 for a perfectly balanced load, so `alpha` has the same scale whatever the depth; `0.01` is a common choice. It is only computed in training mode |
| `stitch_freq` | BTS: a stitch layer every `stitch_freq` blocks (and on the last block) |

<details>
<summary>What happens under the hood?</summary>

1. A config dictionary is created that lists the base expert, two additional experts, the routing layers, etc.
2. `MixtureKit.build_moe()` merges the checkpoints and writes the MoE to `models_merge/gemmax/`.
3. The script reloads the model with `AutoModelForCausalLM` and prints a parameter‑breakdown table — only router weights stay trainable.

</details>

---
## 🔧 Fine-tune / Supervised-Fine-Tuning (SFT)

The **`mixture_training/`** folder contains a ready-to-go scaffold that trains
any merged MoE checkpoint (BTX, traditional or BTS). By default the experts stay
frozen and only the **routers** (BTX, traditional) or the **stitch layers** (BTS)
are trained. Set `use_lora: true` to also train **LoRA adapters** on the
projections of the experts (requires `peft`).

```
mixture_training/
├── config_training.yaml      # all hyper-params in one place
├── deepspeed_config.yaml     # accelerate + DeepSpeed ZeRO-2 config (8 GPUs)
└── train_model.py            # launch-script
```

### 1. Prepare your data

* Expected format: 🤗 `datasets` arrow table saved with `save_to_disk()`.
* A column called **`messages`** (list of chat turns), each turn being a dict
  `{"role": "...", "content": "..."}`.

### 2. Edit **`config_training.yaml`**

Minimal edits for your own run:

| Key            | What it does                                                          |
| -------------- | --------------------------------------------------------------------- |
| `dataset_path` | Path to the dataset of step 1                                         |
| `model_id`     | Path or HF-Hub id of the **merged MoE** (e.g. `models_merge/gemmax`)  |
| `output_dir`   | Where to write the trained model, or the LoRA adapters                |
| `use_lora`     | `false`: routers / stitch layers only, `true`: also LoRA on the experts |
| `response_template` | Loss on the responses only; remove it (and `instruction_template`) to train on whole sequences |

The MoE method is read from the merged model, no flag is needed for BTS.

### 3. Launch training

```bash
mixturekit train mixture_training/config_training.yaml          # one GPU
accelerate launch --config_file mixture_training/deepspeed_config.yaml -m MixtureKit.train mixture_training/config_training.yaml   # several GPUs
```

Set `num_processes` in `deepspeed_config.yaml` to your number of GPUs. The script:

1. Loads the MoE checkpoint in the precision set by `bf16` / `fp16`,
2. Trains with 🤗 `trl`'s **`SFTTrainer`** (the load balancing loss weighted by `alpha` is added to the loss),
3. Saves the model, or the LoRA adapters with the routers / stitch layers, to `output_dir`,
   together with the modeling files, so the output can be reloaded or trained again.

With LoRA, merge the adapters back into the model with
`PeftModel.from_pretrained(base_model, output_dir).merge_and_unload()`.

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

## Models built with earlier versions

Models merged with MixtureKit before version 1.1 carry modeling files generated for
transformers 4.52 and do not load with recent transformers versions. Upgrade them in place
(the weights are kept; traditional MoE weights are renamed to the current layout):

```python
from MixtureKit import upgrade_checkpoint

upgrade_checkpoint("models_merge/gemmax", base_model_class="Gemma3ForCausalLM")
```

`base_model_class` is the transformers causal LM class of the experts, e.g.
`Qwen3ForCausalLM` or `LlamaForCausalLM` (`Gemma3ForCausalLM` for Gemma 3, multimodal included).

### Known limitations

* BTS: `output_hidden_states` returns the hub and expert streams packed on the last
  dimension; with transformers 5 the outputs of the expert layers are listed too.
* The load balancing loss counts padding tokens.
* BTS with `device_map="auto"` across several GPUs: a hub layer and its expert layers
  (`model.expert_layers.N`) should be on the same device; pass an explicit `device_map`.
* trl ≥ 1.15 requires Triton (Linux) for training.

---

## Acknowledgements

The weight-averaging and routed-projection approach builds on ideas from
[Branch-Train-MiX](https://arxiv.org/abs/2403.07816),
[Branch-Train-Stitch](https://arxiv.org/abs/2502.00075) and
[mergoo](https://github.com/Leeroo-AI/mergoo).

---

## License

MixtureKit is released under the **BSD 3-Clause** License — see the [LICENSE](./LICENSE) file for details.

---

*Happy mixing!* 🎛️

## Citation

If you find `mixturekit` useful in your research, please consider citing the [paper](https://arxiv.org/abs/2512.12121):

```bibtex
@article{chamma2025mixturekit,
  title={MixtureKit: A General Framework for Composing, Training, and Visualizing Mixture-of-Experts Models},
  author={Chamma, Ahmad and Herraoui, Omar El and Shang, Guokan},
  journal={arXiv preprint arXiv:2512.12121},
  year={2025}
}
```
