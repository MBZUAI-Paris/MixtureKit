# Training

`mixturekit train` fine-tunes a merged model with trl's `SFTTrainer`. By default the experts
are frozen and only the **routers** (BTX, traditional) or the **stitch layers** (BTS) are
trained. With `use_lora: true`, LoRA adapters are also trained on the projections listed in
`target_modules` (by default `q_proj`, `k_proj`, `v_proj` and `o_proj`).

```bash
pip install "mixturekit[train] @ git+https://github.com/MBZUAI-Paris/MixtureKit"
mixturekit train config_training.yaml
```

## Data

A `datasets` table saved with `save_to_disk()`, with a `messages` column: a list of chat
turns `{"role": ..., "content": ...}`.

## Configuration

Start from
[`mixture_training/config_training.yaml`](https://github.com/MBZUAI-Paris/MixtureKit/blob/main/mixture_training/config_training.yaml).
The keys are the ones of trl's `SFTConfig`, plus the ones below.

```{important}
The templates of this file are Gemma's. For Qwen, use `response_template: "<|im_start|>assistant\n"`
and `instruction_template: "<|im_start|>user\n"`, or remove both. Routers alone usually need a
much larger learning rate than its `3.0e-6`, e.g. `1e-4` to `1e-3`.
```


| Key | Meaning |
| --- | --- |
| `model_id` | The merged model, a directory or a Hub id |
| `dataset_path` | The dataset of the previous step |
| `output_dir` | Where to save the trained model (or the LoRA adapters) |
| `use_lora` | `false`: routers or stitch layers only; `true`: also LoRA adapters (`r`, `lora_alpha`, `target_modules`, ...) |
| `response_template`, `instruction_template` | Compute the loss on the responses only; remove both to train on whole sequences |
| `chat_template` | Used when the tokenizer has none |
| `max_length` | Maximum sequence length |

The method is read from the merged model. The load balancing loss, weighted by the `alpha` of
the merged model, is added to the loss.

## Several GPUs

```bash
accelerate launch --config_file mixture_training/deepspeed_config.yaml \
    -m MixtureKit.train mixture_training/config_training.yaml
```

Set `num_processes` in `deepspeed_config.yaml` to the number of GPUs.

## Output

The trained model is saved to `output_dir` with its modeling files, so it loads, trains
again and uploads like the merged model. With LoRA, merge the adapters into the model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from MixtureKit.training import mark_trained

base = AutoModelForCausalLM.from_pretrained("models_merge/qwen3moe2x", trust_remote_code=True)
model = PeftModel.from_pretrained(base, output_dir).merge_and_unload()
model.save_pretrained("trained")
AutoTokenizer.from_pretrained(output_dir).save_pretrained("trained")
mark_trained("trained")  # the model card then says the routers are trained
```

## Your own training loop

A merged model is a transformers model: any trainer works. In training mode its `outputs.loss`
already includes the load balancing loss weighted by `alpha`. To change the weight:

```python
model.set_load_balancing_alpha(0.02)
```

If a trainer computes the loss itself instead of using `outputs.loss` (e.g. trl 1.x with
`compute_loss_func` unset and its fused loss), pass it
`MixtureKit.training.moe_loss_function(model)`.

```{note}
trl 1.15 and later need Triton, available on Linux. On macOS, use an earlier trl version or
your own training loop.
```
