import black
from functools import partial
import gc
from huggingface_hub import save_torch_state_dict
import json
import os
import re
from tqdm import tqdm
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from ._streaming import ExpertCheckpoint, NotStreamable, write_safetensors
from .utils import (
    _get_config_files,
    _modify_decoder,
    _modify_model,
    _patch_stitches,
    _replace_script,
    _rename_model_type,
    _replace_type_dependency,
)


class MoeExperts:
    def __init__(
        self,
        config,
        torch_dtype=torch.float16,
        device="cpu",
        device_map=None,
        max_shard_size="9GB",
        model_cls=AutoModelForCausalLM,
        streaming=True,
    ):
        """
        This class prepares the unified checkpoint with the provided experts.
        Parameters
        ----------
        config: dict
            The configuration required to setup the composer.
        torch_dtype: torch.dtype, default=torch.float16
            The datatype for loading and saving the weights.
        device: str, default="cpu"
            The device to load the model to.
        device_map: str, default=None
            When in inference mode, it shards the model on Multi-GPUs.
        max_shard_size: str, default="9GB"
            The largest size of a safetensors shard when saving the checkpoint.
        model_cls: type, default=AutoModelForCausalLM
            The class used to load the experts, e.g. for models whose architecture is not mapped by the transformers auto classes.
        streaming: bool, default=True
            Read the experts and write the merged checkpoint tensor by tensor from the
            safetensors files, instead of loading every expert as a full model.
            It falls back to full loading when the files do not match the model.
        """
        self.config = config
        self.torch_dtype = torch_dtype
        self.device = device
        self.device_map = device_map
        self.max_shard_size = max_shard_size
        self.model_cls = model_cls
        self.streaming = streaming
        # An empty list means that every transformer block is converted
        self.config.setdefault("router_layers_index", [])

        self.state_dict = {}
        # Streaming mode: the merged tensors as (name, shape, dtype, compute) entries
        self._plan = None
        self.model_configs = []
        self.generation_configs = []
        self._tied_weights_keys = []
        # The router layers found in the experts, the others are dropped from the config
        self._matched_router_layers = set()
        # The expert providing the configuration and the tokenizer of the merged model,
        # it is switched to the expert with the largest vocabulary
        self.select_moe_model_config_idx = 0

    def _routes_block(self, block_id):
        """
        This function checks whether the transformer block is selected for the moe conversion.

        Parameters
        ----------
        block_id: int
            The index of the transformer block.
        """
        selected = self.config["router_layers_index"]
        if not selected:
            return True
        if selected[0] is None:
            return False
        return block_id in selected

    def _is_routed(self, param_name, router_layers):
        """
        This function checks whether the parameter belongs to a layer converted to experts.

        Parameters
        ----------
        param_name: str
            The name of the parameter in the state dict of the expert.
        router_layers: list
            The layers to convert, e.g. ["mlp.gate_proj", "self_attn.q_proj"].

        Returns
        -------
        A boolean value reflecting whether the parameter is kept per expert.
        """
        block_ids = [int(part) for part in param_name.split(".") if part.isdigit()]
        if len(block_ids) != 1 or not self._routes_block(block_ids[0]):
            return False
        if "mlp" not in param_name and "self_attn" not in param_name:
            return False
        matched = [layer for layer in router_layers if f".{layer}." in param_name]
        self._matched_router_layers.update(matched)
        return bool(matched)

    @staticmethod
    def _expert_key(name, expert_idx):
        """The name of a routed parameter inside the experts of its layer."""
        prefix, _, leaf = name.rpartition(".")
        return f"{prefix}.experts.expert_{expert_idx}.{leaf}"

    @staticmethod
    def _bts_key(name, expert_idx):
        """
        The name of a parameter of a non-hub expert in the BTS model,
        None for the parameters that are not kept (final norm, lm_head).
        """
        prefix, _, leaf = name.rpartition(".")
        slot = expert_idx - 1
        if "model.layers." in prefix:
            # e.g. "19.self_attn.q_proj" -> "19", "self_attn.q_proj"
            block_id, sub_path = prefix.split("model.layers.", 1)[1].split(".", 1)
            return f"model.expert_layers.{block_id}.{slot}.{sub_path}.{leaf}"
        if "embed_tokens" in prefix:
            return f"model.expert_embed_tokens.{slot}.{leaf}"
        return None

    def _load_expert(self, model_id):
        """
        This function loads the requested model using its id.

        Parameters
        ----------
        model_id: str
            The id of the corresponding model from HF.

        Returns
        -------
        model: transformers_modules
            The loaded pretrained model.
        """
        model = self.model_cls.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        # Moving a dispatched model would break its device map
        if self.device_map is None:
            model = model.to(self.device)
        return model

    def _record_metadata(self, model):
        """
        This function keeps the configurations and the tied weights of the loaded expert.
        """
        # Weight tying allows certain weights to be used in multiple places.
        # A common application of weight tying in LLMs is to share the weights
        # between the input embedding layer and the output prediction layer.
        tied_keys = getattr(model, "_tied_weights_keys", None)
        if tied_keys:
            self._tied_weights_keys.extend(tied_keys)
        self.model_configs.append(model.config.to_dict())
        self.generation_configs.append(model.generation_config.to_dict())

    def _grow_rows(self, name, merged, incoming, expert_idx):
        """
        This function zero-pads the first dimension of the smaller tensor, which happens
        when the experts have different vocabulary sizes.

        Parameters
        ----------
        name: str
            The name of the parameter, used for the error message.
        merged: torch.tensor
            The running average of the previous experts.
        incoming: torch.tensor
            The contribution of the current expert.
        expert_idx: int
            The index of the current expert.

        Returns
        -------
        Both tensors with the same shape.
        """
        if merged.shape[1:] != incoming.shape[1:]:
            raise ValueError(
                f"Cannot average '{name}': expert {expert_idx} has shape "
                f"{tuple(incoming.shape)} while the previous experts have "
                f"{tuple(merged.shape)}. Only the first dimension (vocabulary) may differ."
            )
        rows = max(merged.shape[0], incoming.shape[0])
        if incoming.shape[0] == rows:
            # The larger vocabulary defines the configuration and the tokenizer
            self.select_moe_model_config_idx = expert_idx

        def pad(tensor):
            if tensor.shape[0] == rows:
                return tensor
            padded = tensor.new_zeros((rows, *tensor.shape[1:]))
            padded[: tensor.shape[0]] = tensor
            return padded

        return pad(merged), pad(incoming)

    def _add_to_average(self, name, tensor, expert_idx, n_experts):
        """
        This function adds the share of the current expert to the averaged parameter.
        """
        if not tensor.is_floating_point():
            # Integer buffers are copied from the first expert
            self.state_dict.setdefault(name, tensor.to(self.device))
            return

        share = (tensor / n_experts).to(self.device)
        merged = self.state_dict.get(name)
        if merged is None:
            self.state_dict[name] = share
            return
        if merged.shape != share.shape:
            merged, share = self._grow_rows(name, merged, share, expert_idx)
        self.state_dict[name] = merged + share

    def _add_btx_expert(self, weights, expert_idx, n_experts, router_layers):
        """
        BTX / traditional: the routed layers are kept per expert,
        all the other parameters are averaged across the experts.

        Returns
        -------
        The number of routed and averaged tensors.
        """
        n_routed, n_averaged = 0, 0
        for name, tensor in tqdm(weights.items()):
            if self._is_routed(name, router_layers):
                key = self._expert_key(name, expert_idx)
                if key in self.state_dict:
                    raise ValueError(f"Duplicated expert parameter '{key}'.")
                self.state_dict[key] = tensor.to(self.device)
                n_routed += 1
            else:
                self._add_to_average(name, tensor, expert_idx, n_experts)
                n_averaged += 1
        return n_routed, n_averaged

    def _add_bts_expert(self, weights, expert_idx):
        """
        BTS: every expert is kept intact. The first expert is the hub (shared backbone),
        the other experts only contribute their transformer blocks and embeddings.
        """
        for name, tensor in tqdm(weights.items()):
            key = name if expert_idx == 0 else self._bts_key(name, expert_idx)
            if key is not None:
                self.state_dict[key] = tensor.to(self.device)

    def _plan_average(self, name, experts, holders, n_experts):
        """
        This function plans the average of a parameter across the experts holding it.
        The output shape is known upfront, the values are computed when written.
        """
        first = experts[holders[0]]
        dtype = first.dtype(name)
        if not dtype.is_floating_point:
            # Integer buffers are copied from the first expert
            return (name, first.shape(name), dtype, partial(first.tensor, name))

        shape = first.shape(name)
        for expert_idx in holders[1:]:
            incoming = experts[expert_idx].shape(name)
            if incoming == shape:
                continue
            if incoming[1:] != shape[1:]:
                raise ValueError(
                    f"Cannot average '{name}': expert {expert_idx} has shape "
                    f"{incoming} while the previous experts have {shape}. "
                    "Only the first dimension (vocabulary) may differ."
                )
            if incoming[0] > shape[0]:
                self.select_moe_model_config_idx = expert_idx
                shape = incoming

        def compute():
            merged = None
            for expert_idx in holders:
                share = experts[expert_idx].tensor(name) / n_experts
                if merged is None:
                    merged = share
                    continue
                if merged.shape != share.shape:
                    merged, share = self._grow_rows(name, merged, share, expert_idx)
                merged = merged + share
            return merged

        return (name, shape, dtype, compute)

    def _plan_btx(self, experts, router_layers):
        plan, n_routed, n_averaged = [], 0, 0
        names = list(dict.fromkeys(name for ckpt in experts for name in ckpt.names()))
        for name in names:
            holders = [i for i, ckpt in enumerate(experts) if name in ckpt]
            if self._is_routed(name, router_layers):
                for expert_idx in holders:
                    ckpt = experts[expert_idx]
                    plan.append(
                        (
                            self._expert_key(name, expert_idx),
                            ckpt.shape(name),
                            ckpt.dtype(name),
                            partial(ckpt.tensor, name),
                        )
                    )
                    n_routed += 1
            else:
                plan.append(self._plan_average(name, experts, holders, len(experts)))
                n_averaged += 1
        print(f"routed tensors : {n_routed} | averaged tensors : {n_averaged}")
        return plan

    def _plan_bts(self, experts):
        plan = []
        for expert_idx, ckpt in enumerate(experts):
            for name in ckpt.names():
                if expert_idx == 0:
                    # A tied copy (e.g. lm_head) is stored once, through its owner
                    if name in ckpt.aliases and name in ckpt.tied_keys:
                        continue
                    key = name
                else:
                    key = self._bts_key(name, expert_idx)
                if key is not None:
                    plan.append(
                        (key, ckpt.shape(name), ckpt.dtype(name), partial(ckpt.tensor, name))
                    )
        return plan

    def _compose_streaming(self, router_layers):
        """
        This function maps the experts lazily and plans the merged tensors.

        Raises
        ------
        NotStreamable
            When an expert cannot be read tensor by tensor.
        """
        experts = []
        for expert in self.config["experts"]:
            print(f"mapping expert : {expert['model_id']}")
            experts.append(
                ExpertCheckpoint(expert["model_id"], self.model_cls, self.torch_dtype)
            )
        for ckpt in experts:
            self._tied_weights_keys.extend(ckpt.tied_keys)
            self.model_configs.append(ckpt.config.to_dict())
            self.generation_configs.append(ckpt.generation_config)

        if self.config.get("moe_method") == "bts":
            self._plan = self._plan_bts(experts)
        else:
            self._plan = self._plan_btx(experts, router_layers)

    def _use_text_model_for_multimodal(self):
        """
        Multimodal experts (e.g. Gemma3 checkpoints with a vision tower) are merged
        through their language model, using the text-only causal LM class.
        """
        if self.model_cls is not AutoModelForCausalLM:
            return
        model_id = self.config["experts"][0]["model_id"]
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        text_config = config.get_text_config()
        if text_config is config:
            return
        try:
            text_cls = MODEL_FOR_CAUSAL_LM_MAPPING[type(text_config)]
        except KeyError:
            return
        print(
            f"Multimodal experts: merging their language model with {text_cls.__name__}, "
            "the vision tower is not kept."
        )
        self.model_cls = text_cls

    def compose(self):
        """
        This functon composes all the experts into a single unified checkpoint
        with or without a Mixture of Experts (MoE) conversion.
        """
        experts = self.config["experts"]
        moe_method = self.config.get("moe_method")
        self._use_text_model_for_multimodal()
        router_layers = (
            ["mlp"] if moe_method == "traditional" else self.config["router_layers"]
        )
        print(f"MoE Layer Index : {self.config['router_layers_index'] or '[*]'}")

        if self.streaming:
            try:
                self._compose_streaming(router_layers)
                return
            except NotStreamable as reason:
                print(f"Streaming disabled ({reason}), loading the full experts.")
                self._plan = None
                self._tied_weights_keys = []
                self.model_configs = []
                self.generation_configs = []
                self.select_moe_model_config_idx = 0
                self._matched_router_layers = set()

        for expert_idx, expert in enumerate(experts):
            model = self._load_expert(expert["model_id"])
            print(f"merging expert : {expert['model_id']}")
            self._record_metadata(model)
            weights = model.state_dict()

            if moe_method == "bts":
                self._add_bts_expert(weights, expert_idx)
            else:
                n_routed, n_averaged = self._add_btx_expert(
                    weights, expert_idx, len(experts), router_layers
                )
                print(f"routed tensors : {n_routed} | averaged tensors : {n_averaged}")

            # Release the expert before loading the next one
            del model, weights
            gc.collect()

    def _router_layers(self):
        """
        This function returns the router layers of the merged model, without duplicates.
        For BTX, only the layers found in the experts are kept, since the number of
        routers sets the scale of the load balancing loss.
        """
        router_layers = list(dict.fromkeys(self.config["router_layers"]))
        if self.config.get("moe_method") != "btx":
            return router_layers
        missing = [layer for layer in router_layers if layer not in self._matched_router_layers]
        if missing:
            print(f"Router layers not found in the experts, ignored: {missing}")
        return [layer for layer in router_layers if layer in self._matched_router_layers]

    def _moe_config(self):
        """
        This function builds the configuration of the merged model from the
        configuration of the selected expert.

        Returns
        -------
        The new configuration and the model type of the original experts.
        """
        config = self.model_configs[self.select_moe_model_config_idx]
        moe_fields = {
            "num_experts": len(self.config["experts"]),
            "num_experts_per_tok": self.config["num_experts_per_tok"],
            "alpha": self.config["alpha"],
        }
        # Vision-based models nest the language model under "text_config"
        text_config = config.get("text_config")
        if text_config is not None:
            text_config["model_type"] = text_config["model_type"].split("_text")[0]
            text_config.update(moe_fields)

        config["model_type"] = config["model_type"].split("_text")[0]
        old_model_type = config["model_type"]
        config.update(moe_fields)
        config["router_layers"] = self._router_layers()

        if config["architectures"] is None:
            config["architectures"] = [old_model_type.capitalize() + "ForCausalLM"]
        config["architectures"][0] = _rename_model_type(
            config["architectures"][0], config["model_type"], self.config["model_type"]
        )
        # Change the model_type to the new configured model_type with MoE
        config["model_type"] = self.config["model_type"]
        config["_name_or_path"] = ""

        n_blocks = config.get("n_layer", config.get("num_hidden_layers"))
        selected = self.config["router_layers_index"]
        config["router_layers_index"] = (
            list(range(n_blocks))
            if not selected
            else sorted(i for i in set(selected) if i is not None and 0 <= i < n_blocks)
        )
        return config, old_model_type

    def save_checkpoint(self, checkpoint_path):
        """
        This function saves the composed unified checkpoint as safe tensors in shards (chuncks).

        Parameters
        ----------
        checkpoint_path: str
            The path to save the composed unified checkpoint.
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        config, old_model_type = self._moe_config()
        generation_config = self.generation_configs[self.select_moe_model_config_idx]

        ##### SafeTensors #####
        if self._plan is not None:
            write_safetensors(checkpoint_path, self._plan, self.max_shard_size)
        else:
            save_torch_state_dict(
                state_dict=self.state_dict,
                save_directory=checkpoint_path,
                max_shard_size=self.max_shard_size,
                shared_tensors_to_discard=self._tied_weights_keys,
            )

        ##### Tokenizer file #####
        source_id = self.config["experts"][self.select_moe_model_config_idx]["model_id"]
        AutoTokenizer.from_pretrained(source_id).save_pretrained(checkpoint_path)

        # Retrieve, copy and rename the configuration files (modeling + configuration)
        # of the base model
        modeling_file, configuration_file = _get_config_files(
            self.config["experts"][0]["model_id"],
            checkpoint_path,
            config["model_type"],
            self.model_cls,
        )

        ##### Configuration file #####
        # Modify the configuration file
        with open(configuration_file, "r", encoding="utf-8") as f:
            script_configuration = f.read()

            new_script_configuration = _replace_type_dependency(
                script_configuration, old_model_type, config["model_type"]
            )

        # MultiModal case
        class_model_text = re.findall(r"(\w*\d*)Config\(", new_script_configuration)[0]
        class_model = class_model_text.split("Text")[0]

        with open(configuration_file, "w") as f:
            f.write(new_script_configuration)

        ##### Modeling file #####

        with open(modeling_file, "r", encoding="utf-8") as f:
            script = f.read()

        script_type_dependency = _replace_type_dependency(
            script, old_model_type, config["model_type"]
        )

        if self.config.get("moe_method") == "bts":
            config["stitch_freq"] = self.config["stitch_freq"]
            pattern = rf"""
            (                   # Group 1:  class + def header
                class\ {class_model}(?:Text)?Model\(.*?\):\n
                .*?def\ __init__\([^)]+\):\n
            )
            (?P<body>           # Group "body": everything until the next class/EOF
                (?:[ \t]*.*\n)+?
            )
            """

            script_type_dependency = re.sub(
                pattern,
                lambda match: _patch_stitches(match, config["model_type"]),
                script_type_dependency,
                flags=re.DOTALL | re.VERBOSE,
            )

        # Class to modify
        target_class_attn = f"{class_model}Attention"
        target_class_mlp = f"{class_model}MLP"

        # Regex pattern to detect the target class and its body
        class_pattern_attn = (
            rf"(class {target_class_attn}\(.*?\):\n(.*?))(?=\nclass |\ndef |\Z)"
        )
        class_pattern_mlp = rf"(class {target_class_mlp}\(.*?\):\n(.*?))(?=\nclass |\ndef |\Z)"

        # Function to modify only the target class
        def modify_class(match, block_transformer="mlp"):
            """
            This function modifies the highlighted class (attention and mlp).

            Parameters
            ----------
            match: re.Match
                The matching parts in the original script.
            block_transformer: str, default='mlp'
                Whether this layer belongs to the attention part or Feed-Forward part
                especially in the case of common names.
            """
            # Regex patterns to detect nn.Linear, Conv1D and capture key components
            pattern_linear = r"(\s*)self\.(\w+)\s*=\s*nn\.Linear\s*\(\s*((?:[\w\.]+(?:\s*\*\s*[\w\.]+)?)?)\s*,\s*((?:[\w\.]+(?:\s*\*\s*[\w\.]+)?)?)\s*(?:,\s*([\w\.= \*\+\-/]+))?\s*\)"
            pattern_conv = r"(\s*)self\.(\w+)\s*=\s*Conv1D\s*\(\s*((?:[\w\.]+(?:\s*\*\s*[\w\.]+)?)?)\s*,\s*((?:[\w\.]+(?:\s*\*\s*[\w\.]+)?)?)\s*(?:,\s*([\w\.= \*\+\-/]+))?\s*\)"

            class_body = match.group(2)

            if self.config.get("moe_method") in ["btx", "traditional"]:
                modified_script_linear = re.sub(
                    pattern_linear,
                    lambda match: _replace_script(match, "linear", block_transformer),
                    class_body,
                )
                modified_script_conv = re.sub(
                    pattern_conv,
                    lambda match: _replace_script(match, "conv", block_transformer),
                    modified_script_linear,
                )
            else:
                modified_script_conv = class_body

            list_layers = [match[1] for match in re.findall(pattern_linear, class_body)]
            list_layers += [match[1] for match in re.findall(pattern_conv, class_body)]

            # Change the return statement and the class variables to include the load balance loss
            list_layers_return = [
                f"getattr(self.{layer}, 'loss_lb', 0) +" for layer in list_layers[:-1]
            ]
            # list_layers_return.append(f"self.{list_layers[-1]}.loss_lb")
            list_layers_return.append(f"getattr(self.{list_layers[-1]}, 'loss_lb', 0)")
            list_layers_return = " ".join(list_layers_return)

            modified_script_lb = re.sub(
                r"(\breturn\s+)([^\n]+)",
                rf"\1\2, ({list_layers_return})",
                modified_script_conv,
            )

            return match.group(1).replace(class_body, modified_script_lb)

        # Apply modification only to the target classes
        script_attn = re.sub(
            class_pattern_attn,
            lambda match: modify_class(match, block_transformer="attn"),
            script_type_dependency,
            flags=re.DOTALL,
        )
        script_mlp = re.sub(
            class_pattern_mlp,
            lambda match: modify_class(match, block_transformer="mlp"),
            script_attn,
            flags=re.DOTALL,
        )
        # Formatting the script with the black formatter
        script_formatted = black.format_str(script_mlp, mode=black.FileMode())

        # Check if layer_idx is missing in the constructor part in the DecoderLayer class
        pattern_MLP = r"self\.\w+\s*=\s*\w*MLP\([^)]*\)"
        script_mlp = re.sub(
            pattern_MLP,
            lambda match: match.group(0).replace("config)", "config, layer_idx)"),
            script_formatted,
        )

        # config)
        pattern_init_mlp_1 = r"(class\s*\w*MLP\(nn\.Module\):\s*.*?def\s+__init__\(\s*self\s*,\s*)(.*?config)(\s*\))"
        script_init_mlp = re.sub(
            pattern_init_mlp_1, r"\1\2, layer_idx=None\3", script_mlp
        )

        # config: SomethingConfig)
        pattern_init_mlp_2 = r"(class\s+\w*MLP\s*\(nn\.Module\):\s*.*?def\s+__init__\s*\(\s*self\s*,\s*)(config\s*:?[^,)]+)(\s*\))"
        script_init_mlp = re.sub(
            pattern_init_mlp_2, r"\1\2, layer_idx=None\3", script_init_mlp
        )

        # Add the MoE functions to the modeling script
        script_directory = os.path.dirname(os.path.abspath(__file__))

        if self.config.get("moe_method") == "btx":
            file_path = "_btx"
        elif self.config.get("moe_method") == "bts":
            file_path = "_bts"
        elif self.config.get("moe_method") == "traditional":
            file_path = "_traditional"

        # The routed methods share the load balancing loss
        parts = ["_routing", file_path] if file_path in ("_btx", "_traditional") else [file_path]
        parg_moe = ""
        for part in parts:
            with open(
                os.path.join(script_directory, f"{part}.py"), "r", encoding="utf-8"
            ) as fh:
                parg_moe += fh.read() + "\n\n"
        script_moe = script_init_mlp + "\n\n" + parg_moe

        replacement = r"ExtendedModelOutputWithPast(BaseModelOutputWithPast)"
        new_output = "ExtendedModelOutputWithPast"
        old_output = "BaseModelOutputWithPast"

        # Add the new Extended Model definition before the first class
        lines = script_moe.splitlines()
        new_lines = []
        inserted = False
        output_block = f"""
@dataclass
class {replacement}:
    \"\"\"
    Extended model output with an additional float field: `loss_lb`.

    Args:
        loss_lb (float, optional):
            An extra float value added to the output, e.g., for loss or scoring.
    \"\"\"

    loss_lb: Optional[float] = None

"""
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            # Detect first dataclass definition
            if not inserted and (
                stripped.startswith("@dataclass")
                and not lines[i + 1].lstrip().startswith(f"class {old_output}")
            ):
                # Insert the block before this line
                new_lines.append(output_block)
                inserted = True
            if not inserted and (
                stripped.startswith("class ")
                and not stripped.startswith(f"class {old_output}")
            ):
                # Insert the block before this line
                new_lines.append(output_block)
                inserted = True
            new_lines.append(line)
        script_output = "\n".join(new_lines)

        # Replace the old output occurences with the new output
        lines = script_output.split("\n")
        for i, line in enumerate(lines):
            if (
                line.strip().startswith("class ")
                or line.strip().startswith("from ")
                or line.strip() == f"{old_output},"
            ):
                # skip replacement on class definition line
                continue
            lines[i] = line.replace(old_output, new_output)
        script_output = "\n".join(lines)

        # To Check later
        # Add dataclass import if not present
        if "from dataclasses import dataclass" not in script_output:
            script_output = script_output.replace(
                "import torch\n", "import torch\nfrom dataclasses import dataclass\n"
            )

        class_pattern_decoder = (
            rf"(class {class_model}DecoderLayer\(.*?\):\n(.*?))(?=\nclass |\Z)"
        )
        script_decoder = re.sub(
            class_pattern_decoder,
            lambda match: _modify_decoder(match, self.config["moe_method"]),
            script_output,
            flags=re.DOTALL,
        )

        class_pattern_model = (
            rf"(class {class_model}Model\(.*?\):\n(.*?))(?=\nclass |\Z)"
        )
        class_pattern_textmodel = (
            rf"(class {class_model}TextModel\(.*?\):\n(.*?))(?=\nclass |\Z)"
        )
        script_model = re.sub(
            class_pattern_model,
            lambda match: _modify_model(match),
            script_decoder,
            flags=re.DOTALL,
        )
        script_model = re.sub(
            class_pattern_textmodel,
            lambda match: _modify_model(match, update_kv=True),
            script_model,
            flags=re.DOTALL,
        )

        ##### CausalLM Class #####
        class_pattern_causallm = (
            rf"(class {class_model}ForCausalLM\(.*?\):\n(.*?))(?=\nclass |\Z)"
        )

        def patch_loss_handling(class_body: str) -> str:
            pattern = r"(^[ \t]*)if labels is not None:\n((?:\1[ \t]+.*\n)+)"

            def replacer(match):
                indent = match.group(1)
                block = match.group(2)
                line_to_append = "loss = loss + (getattr(outputs, 'loss_lb', 0) or 0)"
                appended_block = block + f"{indent}    {line_to_append}\n"
                return f"{indent}if labels is not None:\n{appended_block}"

            return re.sub(pattern, replacer, class_body, flags=re.MULTILINE)

        script_causallm = re.sub(
            class_pattern_causallm,
            lambda match: match.group(1).replace(
                match.group(2), patch_loss_handling(match.group(2))
            ),
            script_model,
            flags=re.DOTALL,
        )

        # Formatting the script with the black formatter
        script_causallm = black.format_str(script_causallm, mode=black.FileMode())

        with open(modeling_file, "w") as f:
            f.write(script_causallm)

        if "auto_map" in config.keys():
            for key in config["auto_map"].keys():
                config["auto_map"][key] = config["auto_map"][key].split("--")[-1]
                # Replace the function names with the new model_type
                config["auto_map"][key] = _rename_model_type(
                    config["auto_map"][key],
                    old_model_type,
                    self.config["model_type"],
                )
        # For models not providing an explicit auto-classes redirection
        else:
            if f"{class_model}ForCausalLM" in script_moe:
                causal = f"{class_model}ForCausalLM"
            else:
                causal = f"{class_model}LMHeadModel"

            new_model_type = self.config["model_type"]

            AutoConfig = f"configuration_{new_model_type}.{class_model_text}Config"

            config["auto_map"] = {
                "AutoConfig": AutoConfig,
                "AutoModel": f"modeling_{new_model_type}." f"{class_model}Model",
                "AutoModelForCausalLM": f"modeling_{new_model_type}.{causal}",
                "AutoModelForQuestionAnswering": f"modeling_{new_model_type}.{class_model}ForQuestionAnswering",
                "AutoModelForSequenceClassification": f"modeling_{new_model_type}.{class_model}ForSequenceClassification",
                "AutoModelForTokenClassification": f"modeling_{new_model_type}.{class_model}ForTokenClassification",
            }

        config["moe_method"] = self.config["moe_method"]
        if "gemma" in old_model_type:
            config["attn_implementation"] = "eager"
            config["_attn_implementation"] = "eager"

        if class_model_text != class_model:
            config["model_type"] = f"{config['model_type']}_text"
        json.dump(config, open(f"{checkpoint_path}/config.json", "w"), indent=1)

        # Add Generation Config for the merged model
        json.dump(
            generation_config,
            open(f"{checkpoint_path}/generation_config.json", "w"),
            indent=1,
        )

        # Modify the configuration file
        print(f"checkpoint saved at {checkpoint_path}")


def build_moe(
    config,
    torch_dtype=torch.float16,
    model_cls=AutoModelForCausalLM,
    streaming=True,
):
    """
    This function builds the unified framework by merging the provided models
    according to the configuration file. Please refer to the example for the
    format of the configuration.

    Parameters
    ----------
    config: dict
        The configuration of the unified framework to build.
    torch_dtype: torch.dtype, default=torch.float16
        The datatype for loading and saving the weights.
    model_cls: type, default=AutoModelForCausalLM
        The class used to load the experts, e.g. for models whose architecture is not mapped by the transformers auto classes.
    streaming: bool, default=True
        Read the experts and write the merged checkpoint tensor by tensor,
        which keeps the memory usage low. Set to False to load full experts.
    References
    ----------
        .. footbibliography::
    """
    os.makedirs("models_merge", exist_ok=True)
    expertmerger = MoeExperts(
        config, torch_dtype=torch_dtype, model_cls=model_cls, streaming=streaming
    )
    expertmerger.compose()
    expertmerger.save_checkpoint(f"models_merge/{config['model_type']}")
