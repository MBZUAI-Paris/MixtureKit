"""
Building a unified framework with a Mixture-of-Expert (MoE) architecture
========================================================================

This example highlights the substantial steps to merge and integrate the MoE architecture
for futher pretaining and/or finetuning.

We will consider the Llama-family models. However, this example can be applied to any family of models.
"""

#############################################################################
# Imports needed for this script
# ------------------------------

from MixtureKit import build_moe
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    AutoModel,
    AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForVision2Seq,
)

#############################################################################
# Preparing the configuration dictionary
# --------------------------------------
# In this part, we prepare the dictionary of the configuration for the unified framework
# with all the arguments.


config = {
    "moe_method": "btx",
    "stitch_freq": 5,
    "model_type": "gemmax",
    "num_experts_per_tok": 2,
    "experts": [
        # {"expert_name": "base_expert", "model_id": "google/gemma-3-4b-it"},
        {
            "expert_name": "expert_1",
            "model_id": "Sufi2425/FrenchGemma-3-4B-Instruct",
        },
        {
            "expert_name": "expert_2",
            "model_id": "google/medgemma-4b-it",
        },
    ],
    "router_layers": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "alpha": 0,
    "router_layers_index": [],
}

build_moe(
    config=config,
    torch_dtype=torch.bfloat16,
    model_cls=Gemma3ForCausalLM,
)
model_btx = AutoModelForCausalLM.from_pretrained(
    "models_merge/gemmax", trust_remote_code=True
)


###################################Printing Params########################################
total_params = 0
trainable_params = 0

print(f"{'Parameter Name':<60} {'Requires Grad':<15} {'Num Params':<15}")
print("-" * 100)

for name, param in model_btx.named_parameters():
    num_params = param.numel()
    total_params += num_params

    if "gate." in name:  # keep gate routers trainable
        param.requires_grad_(True)
        trainable_params += num_params
    else:
        param.requires_grad_(False)

    print(
        f"{name:<60} {str(param.requires_grad):<15} {num_params:<15,}"
    )  # formatted with comma

print("-" * 100)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")
