"""
Export of a traditional MoE to the native Mixture-of-Experts model of transformers:
Llama and Mistral experts give a Mixtral model, Qwen3 experts a Qwen3-MoE model. The
exported model loads without trust_remote_code and works with the tools that support
these architectures (llama.cpp and GGUF, vLLM, SGLang, MLX, ...).

    from MixtureKit.export import export_native

    export_native("models_merge/llama3moe", "llama3moe-mixtral")

The routing is the same: the softmax of the router logits over the top-k experts.
"""

import json
import os
import shutil

import transformers
from safetensors import safe_open

from ._streaming import _FROM_SAFETENSORS, write_safetensors
from .MoeExperts import _add_legacy_rope_fields

# base_model_class -> (target config class, target model class, renaming of the MoE block)
TARGETS = {
    "LlamaForCausalLM": ("MixtralConfig", "MixtralForCausalLM", "mixtral"),
    "MistralForCausalLM": ("MixtralConfig", "MixtralForCausalLM", "mixtral"),
    "Qwen3ForCausalLM": ("Qwen3MoeConfig", "Qwen3MoeForCausalLM", "qwen3_moe"),
}

# Mixtral checkpoints name the projections of an expert w1 (gate), w3 (up) and w2 (down)
_MIXTRAL_PROJECTIONS = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

# Keys of a MixtureKit configuration that describe the merge, not the architecture
_MERGE_KEYS = {
    "architectures", "auto_map", "base_model_class", "moe_method", "num_experts",
    "expert_names", "expert_model_ids", "expert_is_local", "num_experts_per_tok", "alpha",
    "router_layers", "router_layers_index", "stitch_freq", "stitch_layers_index",
    "mixturekit_trained", "model_type", "transformers_version", "_name_or_path",
}
_ALWAYS_KEPT = {
    "rope_theta", "rope_scaling", "rope_parameters", "torch_dtype", "dtype", "head_dim",
    "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id",
}
_COPIED_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
    "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja", "chat_template.json",
    "generation_config.json",
)


def _native_key(name, layout):
    """The name of a weight of the merged model in the native checkpoint."""
    if layout != "mixtral" or ".mlp." not in name:
        return name  # Qwen3-MoE uses the layout of MixtureKit: mlp.gate, mlp.experts.N.*
    before, after = name.split(".mlp.", 1)
    parts = after.split(".")
    if parts[0] == "experts" and parts[2] in _MIXTRAL_PROJECTIONS:
        parts[2] = _MIXTRAL_PROJECTIONS[parts[2]]
    return f"{before}.block_sparse_moe.{'.'.join(parts)}"


def native_config(config):
    """
    This function returns the configuration of the native MoE model for the configuration
    (dict) of a traditional MoE merged with MixtureKit.
    """
    base = config.get("base_model_class")
    if config.get("moe_method") != "traditional":
        raise ValueError(
            f"Only traditional MoE models have a native transformers equivalent, not "
            f"{config.get('moe_method')!r}: BTX routes each projection on its own and BTS "
            "runs every expert"
        )
    if base not in TARGETS:
        raise ValueError(
            f"No native MoE architecture for {base} experts; supported: {', '.join(TARGETS)}"
        )
    if config.get("mlp_bias") or (config.get("attention_bias") and TARGETS[base][2] == "mixtral"):
        raise ValueError("Mixtral has no bias in its projections: these experts cannot be exported")
    if TARGETS[base][2] == "qwen3_moe" and config.get("use_sliding_window"):
        # Qwen3 slides the window on some layers only, Qwen3-MoE (transformers 5) on all
        raise ValueError("Qwen3 experts with use_sliding_window cannot be exported to Qwen3-MoE")
    config_class = getattr(transformers, TARGETS[base][0])
    allowed = set(config_class().to_dict()) | _ALWAYS_KEPT
    values = {k: v for k, v in config.items() if k in allowed and k not in _MERGE_KEYS}
    if TARGETS[base][2] == "mixtral":
        values.update(num_local_experts=config["num_experts"], sliding_window=config.get("sliding_window"))
    else:
        values.update(
            num_experts=config["num_experts"],
            moe_intermediate_size=config["intermediate_size"],
            decoder_sparse_step=1,
            mlp_only_layers=[],
            norm_topk_prob=True,  # the softmax over the top-k experts, as in MixtureKit
        )
    values.update(
        num_experts_per_tok=config["num_experts_per_tok"],
        # The load balancing loss of MixtureKit is the one of transformers divided by top-k
        router_aux_loss_coef=config.get("alpha", 0.0) / config["num_experts_per_tok"],
        output_router_logits=False,
        architectures=[TARGETS[base][1]],
    )
    return config_class.from_dict(values)


def export_native(model_dir, output_dir):
    """
    This function writes a traditional MoE merged with MixtureKit as the native MoE model
    of transformers (Mixtral for Llama and Mistral experts, Qwen3-MoE for Qwen3 experts),
    tensor by tensor, and returns the output directory.

    Parameters
    ----------
    model_dir: str
        The directory of the merged model, with its routers trained (or at least saved).
    output_dir: str
        Where to write the native model: a new or empty directory, or a previous export.
    """
    config_file = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_file):
        raise ValueError(f"{model_dir} is not a model directory (no config.json)")
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)
    target = native_config(config)
    layout = TARGETS[config["base_model_class"]][2]
    files = _weight_files(model_dir)
    _check_routers(model_dir, files, config)
    if not config.get("mixturekit_trained"):
        print("Warning: the routers of this model are not marked as trained (mixturekit train)")
    _prepare_output(model_dir, output_dir)

    entries = []
    for file in files:
        path = os.path.join(model_dir, file)
        with safe_open(path, framework="pt") as handle:
            for name in handle.keys():
                info = handle.get_slice(name)
                entries.append((
                    _native_key(name, layout), info.get_shape(), _FROM_SAFETENSORS[info.get_dtype()],
                    lambda path=path, name=name: _read(path, name),
                ))
    write_safetensors(output_dir, entries)
    target.save_pretrained(output_dir)
    # Readable by transformers 4 and by the tools that read config.json themselves
    with open(os.path.join(output_dir, "config.json"), encoding="utf-8") as f:
        saved = json.load(f)
    _add_legacy_rope_fields(saved, TARGETS[config["base_model_class"]][0])
    if layout == "qwen3_moe":
        saved.setdefault("num_experts", config["num_experts"])
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
    for file in _COPIED_FILES:
        if os.path.exists(os.path.join(model_dir, file)):
            shutil.copy(os.path.join(model_dir, file), os.path.join(output_dir, file))
    return output_dir


def _weight_files(model_dir):
    """The safetensors files of the model: the ones of its index, or its single file."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        return ["model.safetensors"]
    raise ValueError(f"{model_dir} has no safetensors weights")


def _check_routers(model_dir, files, config):
    """A fresh merge does not store its routers (random at every load): nothing to export."""
    keys = set()
    for file in files:
        with safe_open(os.path.join(model_dir, file), framework="pt") as handle:
            keys.update(handle.keys())
    missing = [
        i for i in range(config["num_hidden_layers"]) if f"model.layers.{i}.mlp.gate.weight" not in keys
    ]
    if missing:
        raise ValueError(
            f"{model_dir} has no trained routers (layers {missing[:5]}...): train them first "
            "(mixturekit train), the routers of a fresh merge are random"
        )


def _prepare_output(model_dir, output_dir):
    if os.path.exists(output_dir) and os.path.samefile(output_dir, model_dir):
        raise ValueError("Export to another directory than the merged model")
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        previous = os.path.join(output_dir, "config.json")
        architectures = []
        if os.path.exists(previous):
            with open(previous, encoding="utf-8") as f:
                architectures = json.load(f).get("architectures") or []
        if not set(architectures) & {target[1] for target in TARGETS.values()}:
            raise ValueError(f"{output_dir} is not empty: export to a new directory")
        shutil.rmtree(output_dir)  # a previous export: replaced, stale files included
    os.makedirs(output_dir, exist_ok=True)


def _read(path, name):
    with safe_open(path, framework="pt") as handle:
        return handle.get_tensor(name)
