# Configuration

`mixturekit build` reads a YAML file, `build_moe` the same keys as a dictionary. Examples:
[`examples/configs/`](https://github.com/MBZUAI-Paris/MixtureKit/tree/main/examples/configs).

| Key | Meaning |
| --- | --- |
| `moe_method` | `btx`, `traditional` or `bts` |
| `model_type` | The name of the merged model, also its default folder `models_merge/<model_type>` |
| `experts` | A list of `{expert_name, model_id}`, Hub ids or local directories; with BTS the first one is the hub |
| `router_layers` | BTX: the projections turned into routed experts, e.g. `mlp.gate_proj`, `self_attn.q_proj` |
| `router_layers_index` | BTX: the decoder blocks to convert, `[]` for all of them (traditional converts every block) |
| `num_experts_per_tok` | The number of experts used for every token (top-k), 2 by default |
| `alpha` | BTX, traditional: the weight of the load balancing loss, `0` (the default) to disable it; `0.01` is a common choice |
| `stitch_freq` | BTS: a stitch layer every `stitch_freq` blocks (and on the last block), 5 by default |
| `stitch_layers_index` | BTS: the blocks followed by a stitch layer, instead of `stitch_freq` |

Options of the build itself: with `mixturekit build`, in the file or on the command line
(which wins); with `build_moe`, as arguments:

| Key | Option | `build_moe` | Meaning |
| --- | --- | --- | --- |
| `dtype` | `--dtype` | `torch_dtype` | `bfloat16` (default), `float16` or `float32` |
| `output_dir` | `--output` | `output_dir` | Where to write the merged model, `models_merge/<model_type>` by default |
| `streaming` | `--no-streaming` | `streaming` | Read and write tensor by tensor (default), or load the full experts |

## Memory

By default the experts are read from their safetensors files and the merged model is written
tensor by tensor: a merge needs memory for a few tensors at a time (the largest tensor of every
expert, e.g. the embeddings), not for the experts. The experts are still downloaded to disk. Models
whose weights cannot be streamed (e.g. `.bin` checkpoints) are loaded whole.

## Python

```python
import torch
from MixtureKit import build_moe

config = {
    "moe_method": "btx",
    "model_type": "gemmax",
    "num_experts_per_tok": 2,
    "experts": [
        {"expert_name": "FrenchGemma", "model_id": "Sufi2425/FrenchGemma-3-4B-Instruct"},
        {"expert_name": "MedGemma", "model_id": "google/medgemma-4b-it"},
    ],
    "router_layers": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "router_layers_index": [],
    "alpha": 0.01,
}
path = build_moe(config, torch_dtype=torch.bfloat16)
```

`model_cls` loads the experts with another class than `AutoModelForCausalLM`, e.g.
`Gemma3ForCausalLM`.
