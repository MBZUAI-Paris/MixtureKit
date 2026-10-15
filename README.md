<p align="center">
  <img alt="MixtureKit logo" src="https://raw.githubusercontent.com/MBZUAI-Paris/MixtureKit/main/logo.png" width="150">
</p>

<h1 align="center">MixtureKit</h1>

<p align="center">
  <b>Turn the fine-tuned models you already have into one Mixture-of-Experts model.</b><br>
  Merge Hugging Face checkpoints, train the routers, see which expert every token uses.
</p>

<p align="center">
  <a href="https://pypi.org/project/mixturekit/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mixturekit"></a>
  <a href="https://github.com/MBZUAI-Paris/MixtureKit/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/MBZUAI-Paris/MixtureKit/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://mbzuai-paris.github.io/MixtureKit/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-blue"></a>
  <a href="https://arxiv.org/abs/2512.12121"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.12121-b31b1b"></a>
  <a href="https://github.com/MBZUAI-Paris/MixtureKit/blob/main/LICENSE"><img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-green"></a>
  <a href="https://colab.research.google.com/github/MBZUAI-Paris/MixtureKit/blob/main/examples/build_your_own_moe.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

You have a base model and a few fine-tunes of it: one for math, one for code, one for a
language. MixtureKit composes them into a single Mixture-of-Experts model whose routers
learn, token by token, which expert to use, and shows you what they learned.

```bash
pip install "mixturekit[vis]"

mixturekit build examples/configs/traditional_qwen3.yaml   # merge the experts -> models_merge/qwen3moe2x
mixturekit train my_training.yaml                          # train the routers -> e.g. qwen3moe2x-trained
mixturekit visualize qwen3moe2x-trained                    # see which expert every token uses
mixturekit push qwen3moe2x-trained my-org/my-moe           # share it, with a model card
```

`my_training.yaml` starts from [`mixture_training/config_training.yaml`](https://github.com/MBZUAI-Paris/MixtureKit/blob/main/mixture_training/config_training.yaml)
(`model_id: models_merge/qwen3moe2x`, your dataset and `output_dir`). Merging runs on a
laptop; training the routers wants a GPU.

<p align="center">
  <img alt="Token routing of a two-expert MoE of Nile 4B models: Latin-script (Arabizi) Egyptian tokens go to one expert, Arabic-script tokens to the other" src="https://raw.githubusercontent.com/MBZUAI-Paris/MixtureKit/main/examples/vis_example.png" width="560">
</p>

**Try it in 15 minutes on a free GPU:** the [Colab notebook](https://colab.research.google.com/github/MBZUAI-Paris/MixtureKit/blob/main/examples/build_your_own_moe.ipynb)
merges two Qwen3-0.6B models, trains the routers and shows the routing.

## Why MixtureKit

* **Three composition methods** behind one config:
  [Branch-Train-MiX](https://arxiv.org/abs/2403.07816) (BTX), Mixtral-style MoE and
  [Branch-Train-Stitch](https://arxiv.org/abs/2502.00075) (BTS).
* **Merges on a laptop**: the experts are read and the merged model written tensor by
  tensor, so a merge needs memory for a few tensors at a time, not for the experts.
* **Plain transformers models**: the result loads with
  `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`, without MixtureKit,
  and is tested on transformers 4.52 to 5.19.
* **Router training included**: `mixturekit train` trains the routers (or the BTS stitch
  layers) with a load balancing loss, optionally with LoRA on the experts, on one GPU or many.
* **Routing visualizer**: an app, a shareable HTML page or an inline notebook view of which
  expert every token uses, layer by layer.
* **Benchmarks**: `mixturekit benchmark` scores the merged model next to its experts with
  lm-evaluation-harness and writes the table.
* **Hub ready**: `mixturekit push` uploads the model with a generated model card.
* **Runs everywhere**: `mixturekit export` turns a traditional MoE of Llama, Mistral or Qwen3
  experts into a native Mixtral or Qwen3-MoE model, ready for GGUF and llama.cpp, vLLM or MLX.

Tested with Llama 3, Qwen 2.5, Qwen 3, Qwen 3.5, Gemma 3 and Gemma 4 experts (multimodal
checkpoints are merged through their language model).

## Which method?

| | BTX | Traditional | BTS |
| --- | --- | --- | --- |
| Kept per expert | the projections you choose | the feed-forward block | the whole model |
| Shared | everything else, averaged | everything else, averaged | nothing: experts run next to a hub |
| Trained | a router per routed projection | a router per layer | stitch layers |
| Compute per token | top-k routed projections | top-k feed-forward blocks | all the experts |

Start with **traditional** for fine-tunes of one base model. More in the
[guide](https://mbzuai-paris.github.io/MixtureKit/methods.html).

## A merge, in YAML or Python

```yaml
moe_method: traditional          # btx, traditional or bts
model_type: qwen3moe2x           # the name of the merged model and of its folder
num_experts_per_tok: 2
experts:
  - {expert_name: base, model_id: Qwen/Qwen3-0.6B-Base}
  - {expert_name: chat, model_id: Qwen/Qwen3-0.6B}
alpha: 0.01                      # load balancing weight
```

```python
import torch
from MixtureKit import build_moe

config = {
    "moe_method": "traditional",
    "model_type": "qwen3moe2x",
    "experts": [
        {"expert_name": "base", "model_id": "Qwen/Qwen3-0.6B-Base"},
        {"expert_name": "chat", "model_id": "Qwen/Qwen3-0.6B"},
    ],
    "alpha": 0.01,
}
path = build_moe(config, torch_dtype=torch.bfloat16)   # models_merge/qwen3moe2x
```

More configurations: [`examples/configs/`](https://github.com/MBZUAI-Paris/MixtureKit/tree/main/examples/configs). All the keys:
[Configuration](https://mbzuai-paris.github.io/MixtureKit/configuration.html).

## Documentation

**[mbzuai-paris.github.io/MixtureKit](https://mbzuai-paris.github.io/MixtureKit/)**:
[quickstart](https://mbzuai-paris.github.io/MixtureKit/quickstart.html) ·
[methods](https://mbzuai-paris.github.io/MixtureKit/methods.html) ·
[command line](https://mbzuai-paris.github.io/MixtureKit/cli.html) ·
[training](https://mbzuai-paris.github.io/MixtureKit/training.html) ·
[visualizer](https://mbzuai-paris.github.io/MixtureKit/visualizer.html) ·
[export and GGUF](https://mbzuai-paris.github.io/MixtureKit/export.html) ·
[benchmarks](https://mbzuai-paris.github.io/MixtureKit/benchmark.html) ·
[upgrading older models and limitations](https://mbzuai-paris.github.io/MixtureKit/upgrade.html) ·
[API](https://mbzuai-paris.github.io/MixtureKit/api.html)

## Development

```bash
git clone https://github.com/MBZUAI-Paris/MixtureKit && cd MixtureKit
pip install -e ".[test]"
pytest                                # CPU, tiny models, no download
pip install -e ".[doc]" && sphinx-build docs docs/_build/html
```

Requirements: Python 3.10 or later, PyTorch 2.5 or later, transformers 4.52 to 5.x.
Issues and pull requests are welcome: see [CONTRIBUTING.md](https://github.com/MBZUAI-Paris/MixtureKit/blob/main/CONTRIBUTING.md).

## Citation

If MixtureKit is useful in your work, please cite the [paper](https://arxiv.org/abs/2512.12121):

```bibtex
@article{chamma2025mixturekit,
  title={MixtureKit: A General Framework for Composing, Training, and Visualizing Mixture-of-Experts Models},
  author={Chamma, Ahmad and Herraoui, Omar El and Shang, Guokan},
  journal={arXiv preprint arXiv:2512.12121},
  year={2025}
}
```

## Acknowledgements and license

MixtureKit builds on [Branch-Train-MiX](https://arxiv.org/abs/2403.07816) and
[Branch-Train-Stitch](https://arxiv.org/abs/2502.00075). It is released under the
[BSD 3-Clause License](https://github.com/MBZUAI-Paris/MixtureKit/blob/main/LICENSE).
