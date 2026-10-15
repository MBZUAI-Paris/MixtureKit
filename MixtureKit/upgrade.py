"""
Upgrades a model merged with an earlier MixtureKit version (whose modeling files were
generated from the transformers 4.52 sources) to the current modeling files, so that it
loads with every supported transformers version.
"""

import glob
import json
import os
import re

import shutil
import tempfile

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .MoeExperts import _add_legacy_rope_fields, write_modeling_files

# Traditional MoE, previous layout -> current layout
_OLD_ROUTER = re.compile(r"^(.*\.layers\.\d+)\.gate\.(weight|bias)$")
_OLD_EXPERT = re.compile(r"^(.*\.layers\.\d+)\.mlp\.([^.]+)\.experts\.expert_(\d+)\.(weight|bias)$")


def _traditional_key(key):
    match = _OLD_ROUTER.match(key)
    if match:
        return f"{match.group(1)}.mlp.gate.{match.group(2)}"
    match = _OLD_EXPERT.match(key)
    if match:
        block, projection, expert, leaf = match.groups()
        return f"{block}.mlp.experts.{expert}.{projection}.{leaf}"
    return key


def _stitch_layers(config):
    n_blocks = config["num_hidden_layers"]
    stitch_freq = config.get("stitch_freq", 5)
    layers = list(config.get("stitch_layers_index") or []) or [
        i for i in range(n_blocks) if i % stitch_freq == 0 and i != 0
    ]
    if n_blocks - 1 not in layers:
        layers.append(n_blocks - 1)
    return layers


def _check_weights(checkpoint_path, config, model_type, model_class, config_class, moe_method):
    """
    This function builds the upgraded model without weights and compares its parameters
    with the stored weights (after renaming), so that a wrong base class is reported
    before the checkpoint is changed.
    """
    stored = set()
    for file in glob.glob(os.path.join(checkpoint_path, "*.safetensors")):
        with safe_open(file, framework="pt") as handle:
            stored.update(handle.keys())
    if moe_method == "traditional":
        stored = {_traditional_key(key) for key in stored}

    work = tempfile.mkdtemp()
    try:
        with open(os.path.join(work, "config.json"), "w") as f:
            json.dump(config, f)
        write_modeling_files(work, model_type, model_class, config_class)
        auto_config = transformers.AutoConfig.from_pretrained(work, trust_remote_code=True)
        with torch.device("meta"):
            model = transformers.AutoModelForCausalLM.from_config(auto_config, trust_remote_code=True)
        expected = set(model.state_dict())
        tied = set(getattr(model, "_tied_weights_keys", None) or [])
    except Exception as error:
        raise ValueError(
            f"Cannot build {model_class} from the configuration of {checkpoint_path}: {error}"
        ) from error
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Routers and stitch layers may be missing (untrained models): they are initialized
    optional = lambda key: key.endswith(".gate.weight") or ".stitches." in key or key in tied
    missing = sorted(k for k in expected - stored if not optional(k))
    unexpected = sorted(stored - expected)
    if missing or unexpected:
        raise ValueError(
            f"The weights of {checkpoint_path} do not match {model_class}: missing "
            f"{missing[:3]}, unexpected {unexpected[:3]}. Check base_model_class."
        )


def upgrade_checkpoint(checkpoint_path, base_model_class):
    """
    This function upgrades, in place, a merged model built with an earlier MixtureKit
    version. Its weights are kept: only the configuration and modeling files are
    replaced, the traditional MoE weights are renamed to the current layout and missing
    BTS stitch layers are added (identity projections, zero gates, as in training).

    Parameters
    ----------
    checkpoint_path: str
        The directory of the merged model.
    base_model_class: str or type
        The transformers causal LM class of the experts, e.g. "Gemma3ForCausalLM" for
        Gemma 3 experts (multimodal ones included) or "Qwen3ForCausalLM".
    """
    if not isinstance(base_model_class, str):
        base_model_class = base_model_class.__name__
    model_cls = getattr(transformers, base_model_class, None)
    if model_cls is None:
        raise ValueError(f"{base_model_class} is not a transformers class")
    config_class = model_cls.config_class.__name__

    config_file = os.path.join(checkpoint_path, "config.json")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    moe_method = config.get("moe_method")
    if moe_method is None and config.get("router_layers") and config.get("num_experts"):
        # The first MixtureKit versions only built BTX models and did not save the method
        moe_method = config["moe_method"] = "btx"
    if moe_method not in ("btx", "traditional", "bts"):
        raise ValueError(f"{checkpoint_path} is not a model merged with MixtureKit")
    if config.get("base_model_class"):
        print(f"{checkpoint_path} already uses the current modeling files")
        return

    # Previous versions suffixed the model type of text models taken from multimodal ones
    model_type = config["model_type"]
    if model_type.endswith("_text"):
        model_type = model_type[: -len("_text")]

    # The upgraded configuration, checked against the stored weights before any change
    new_config = dict(config)
    if moe_method == "traditional" or not new_config.get("router_layers_index"):
        new_config["router_layers_index"] = list(range(new_config["num_hidden_layers"]))
    new_config.update(
        {
            "model_type": model_type,
            "architectures": ["MixtureKitForCausalLM"],
            "auto_map": {
                "AutoConfig": f"configuration_{model_type}.MixtureKitConfig",
                "AutoModelForCausalLM": f"modeling_{model_type}.MixtureKitForCausalLM",
            },
            "base_model_class": base_model_class,
        }
    )
    _add_legacy_rope_fields(new_config, config_class)
    _check_weights(checkpoint_path, new_config, model_type, base_model_class, config_class, moe_method)

    # Weights
    weight_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
    keys = set()
    for file in weight_files:
        weights = load_file(file)
        if moe_method == "traditional":
            renamed = {_traditional_key(k): v for k, v in weights.items()}
            if renamed.keys() != weights.keys():
                save_file(renamed, file, metadata={"format": "pt"})
            weights = renamed
        keys.update(weights)

    if moe_method == "bts":
        hidden_size = config["hidden_size"]
        dtype = getattr(torch, str(config.get("torch_dtype") or "float32").replace("torch.", ""), torch.float32)
        stitches = {}
        for block_id in _stitch_layers(config):
            prefix = f"model.layers.{block_id}.stitches"
            for stream in range(config["num_experts"] - 1):
                stitches[f"{prefix}.w_proj.{stream}.weight"] = torch.eye(hidden_size, dtype=dtype)
            for stream in range(config["num_experts"]):
                stitches[f"{prefix}.w_gate.{stream}.weight"] = torch.zeros(hidden_size, hidden_size, dtype=dtype)
        stitches = {k: v for k, v in stitches.items() if k not in keys}
        if stitches:
            if len(weight_files) == 1:
                # model.safetensors is loaded without an index: the stitches go inside
                weights = load_file(weight_files[0])
                weights.update(stitches)
                save_file(weights, weight_files[0], metadata={"format": "pt"})
            else:
                file = os.path.join(checkpoint_path, "model-stitches.safetensors")
                save_file(stitches, file, metadata={"format": "pt"})
                weight_files.append(file)
            print(f"added {len(stitches)} initial stitch weights")

    # Sharded checkpoints: the index follows the renamed and added weights
    index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if len(weight_files) > 1:
        weight_map = {}
        total_size = 0
        for file in weight_files:
            for key, value in load_file(file).items():
                weight_map[key] = os.path.basename(file)
                total_size += value.numel() * value.element_size()
        with open(index_file, "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

    # Configuration and modeling files
    for file in glob.glob(os.path.join(checkpoint_path, "modeling_*.py")) + glob.glob(
        os.path.join(checkpoint_path, "configuration_*.py")
    ):
        os.remove(file)
    write_modeling_files(checkpoint_path, model_type, base_model_class, config_class)
    with open(config_file, "w") as f:
        json.dump(new_config, f, indent=1)
    print(f"{checkpoint_path} upgraded")
