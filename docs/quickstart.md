# Quickstart

## Install

```bash
pip install "mixturekit"            # build and load
pip install "mixturekit[train]"     # + training (trl, peft, ...)
pip install "mixturekit[vis]"       # + routing visualizer
```

Or from a clone: `pip install -e ".[train,vis]"`.

Requirements: Python 3.10 or later, PyTorch 2.5 or later, transformers 4.52 to 5.x.

## 1. Pick the experts

The experts must share an architecture and a tokenizer: a base model and models fine-tuned
from it are the usual case. Tested families: Llama 3, Qwen 2.5, Qwen 3, Qwen 3.5, Gemma 3 and
Gemma 4 (Qwen 3.5 and Gemma 4 need a transformers 5 version that includes them). Multimodal
checkpoints such as Gemma 3 are merged through their language model, which gives a text-only MoE.

## 2. Build the MoE

Describe the merge in a YAML file:

```yaml
moe_method: traditional          # btx, traditional or bts (see "Which method?")
model_type: qwen3moe2x           # the name of the merged model and of its folder
num_experts_per_tok: 2
experts:
  - {expert_name: base, model_id: Qwen/Qwen3-0.6B-Base}
  - {expert_name: chat, model_id: Qwen/Qwen3-0.6B}
alpha: 0.01                      # load balancing weight
```

```bash
mixturekit build moe.yaml        # writes models_merge/qwen3moe2x
```

or in Python:

```python
import torch
import yaml
from MixtureKit import build_moe

config = yaml.safe_load(open("moe.yaml"))   # or the same keys as a dict
path = build_moe(config, torch_dtype=torch.bfloat16)
```

The merged model is a regular transformers model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models_merge/qwen3moe2x", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("models_merge/qwen3moe2x")
```

## 3. Train the routers

A fresh merge has random routers. Train them on a chat dataset (only the routers, or the
stitch layers of BTS, are trained by default): copy
[`mixture_training/config_training.yaml`](https://github.com/MBZUAI-Paris/MixtureKit/blob/main/mixture_training/config_training.yaml),
set `model_id: models_merge/qwen3moe2x`, your dataset and `output_dir: qwen3moe2x-trained`
(see {doc}`training`), then:

```bash
mixturekit train my_training.yaml
```

## 4. Look at the routing

```bash
mixturekit visualize qwen3moe2x-trained
```

See {doc}`visualizer`.

## 5. Share it

```bash
mixturekit push qwen3moe2x-trained my-org/qwen3-moe-2x0.6b
```

uploads the model with a generated model card: the experts, the method, whether the routers
are trained, how to load it and the configuration to rebuild it.
