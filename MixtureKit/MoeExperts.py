from functools import partial
import gc
from huggingface_hub import save_torch_state_dict
import json
import os
from tqdm import tqdm
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from ._streaming import ExpertCheckpoint, NotStreamable, dtype_kwargs, write_safetensors


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
        # (causal LM class, config class) of each expert, the base of the merged model
        self.model_classes = []
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
        if len(block_ids) != 1:
            return False
        # Traditional MoE converts the feed-forward block of every layer
        if self.config.get("moe_method") != "traditional" and not self._routes_block(block_ids[0]):
            return False
        # Feed-forward and attention projections (linear_attn: Qwen3.5's linear-attention blocks)
        if not any(part in param_name for part in ("mlp", "self_attn", "linear_attn")):
            return False
        matched = [layer for layer in router_layers if f".{layer}." in param_name]
        self._matched_router_layers.update(matched)
        return bool(matched)

    def _expert_key(self, name, expert_idx):
        """
        The name of a routed parameter in the merged model:
        BTX: model.layers.0.mlp.up_proj.experts.expert_1.weight (one router per projection)
        traditional: model.layers.0.mlp.experts.1.up_proj.weight (one router per layer)
        """
        if self.config.get("moe_method") == "traditional":
            before, after = name.split(".mlp.", 1)
            return f"{before}.mlp.experts.{expert_idx}.{after}"
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
            # e.g. "19.self_attn.q_proj" -> "19", "self_attn.q_proj"; a parameter of the layer
            # itself (e.g. Gemma 4's layer scalar) has no sub-path
            block_id, _, sub_path = prefix.split("model.layers.", 1)[1].partition(".")
            parts = [f"model.expert_layers.{block_id}.{slot}", sub_path, leaf]
            return ".".join(part for part in parts if part)
        # The token embedding only (not e.g. Gemma 4's per-layer embeddings)
        if prefix.split(".")[-1] == "embed_tokens":
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
        model, loading_info = self.model_cls.from_pretrained(
            model_id,
            **dtype_kwargs(self.torch_dtype),
            device_map=self.device_map,
            trust_remote_code=True,
            output_loading_info=True,
        )
        tied_keys = set(getattr(model, "_tied_weights_keys", None) or [])
        missing = [key for key in loading_info["missing_keys"] if key not in tied_keys]
        if missing:
            raise ValueError(
                f"{model_id} does not store {len(missing)} weights of {type(model).__name__}, "
                f"they would be randomly initialized: {missing[:5]}"
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
            # transformers 5 maps the tied copies to their source
            self._tied_weights_keys.extend(list(tied_keys))
        self.model_configs.append(model.config.to_dict())
        self.generation_configs.append(model.generation_config.to_dict())
        self.model_classes.append((type(model).__name__, type(model.config).__name__))

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
        tied_keys = set(self._tied_weights_keys)
        for name, tensor in tqdm(weights.items()):
            key = name if expert_idx == 0 else self._bts_key(name, expert_idx)
            if key is not None:
                # Tied copies are stored as separate tensors, as in the streamed build
                tensor = tensor.clone() if key in tied_keys else tensor
                self.state_dict[key] = tensor.to(self.device)

    def _check_same_parameters(self, names_per_expert):
        """
        This function checks that all the experts have the same parameter names, otherwise
        the averages would be wrong and some experts would be missing from the checkpoint.
        """
        reference = set(names_per_expert[0])
        for expert_idx, names in enumerate(names_per_expert[1:], start=1):
            names = set(names)
            if names != reference:
                model_id = self.config["experts"][expert_idx]["model_id"]
                only_here = sorted(names - reference)[:3]
                only_first = sorted(reference - names)[:3]
                raise ValueError(
                    f"Expert {expert_idx} ({model_id}) does not have the same parameters "
                    f"as expert 0: {only_here} only in expert {expert_idx}, "
                    f"{only_first} only in expert 0. The experts must share one architecture."
                )

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

    def _stitch_entries(self):
        """
        This function plans the initial weights of the BTS stitch layers, saved with the
        checkpoint so that a freshly built model is deterministic: the projections from the
        experts to the hub start as identities and the gates at zero.
        Their positions follow the stitch layers created by the modeling file.
        """
        model_config = self.model_configs[self.select_moe_model_config_idx]
        model_config = model_config.get("text_config") or model_config
        hidden_size = model_config["hidden_size"]
        n_blocks = model_config.get("n_layer", model_config.get("num_hidden_layers"))
        n_streams = len(self.config["experts"])

        stitch_freq = self.config.get("stitch_freq", 5)
        stitch_layers = list(self.config.get("stitch_layers_index") or []) or [
            i for i in range(n_blocks) if i % stitch_freq == 0 and i != 0
        ]
        invalid = [i for i in stitch_layers if not 0 <= i < n_blocks]
        if invalid:
            raise ValueError(
                f"stitch_layers_index {invalid} is out of range, the model has {n_blocks} blocks"
            )
        if n_blocks - 1 not in stitch_layers:
            stitch_layers.append(n_blocks - 1)

        shape = (hidden_size, hidden_size)
        identity = lambda: torch.eye(hidden_size, dtype=self.torch_dtype)
        zeros = lambda: torch.zeros(shape, dtype=self.torch_dtype)
        entries = []
        for block_id in stitch_layers:
            prefix = f"model.layers.{block_id}.stitches"
            for stream in range(n_streams - 1):
                entries.append((f"{prefix}.w_proj.{stream}.weight", shape, self.torch_dtype, identity))
            for stream in range(n_streams):
                entries.append((f"{prefix}.w_gate.{stream}.weight", shape, self.torch_dtype, zeros))
        return entries

    def _check_bts_vocabularies(self):
        """
        BTS keeps one embedding per expert, all fed with the same token ids: the experts
        must share one vocabulary.
        """
        sizes = {(c.get("text_config") or c).get("vocab_size") for c in self.model_configs}
        if len(sizes) > 1:
            raise ValueError(
                f"BTS needs experts with the same vocabulary, found vocabulary sizes {sorted(sizes)}"
            )

    def _plan_bts(self, experts):
        plan = []
        for expert_idx, ckpt in enumerate(experts):
            for name in ckpt.names():
                if expert_idx == 0:
                    # Tied copies (e.g. lm_head) are stored too: some transformers versions
                    # do not tie them again when loading the merged model
                    key = name
                else:
                    key = self._bts_key(name, expert_idx)
                if key is not None:
                    plan.append(
                        (key, ckpt.shape(name), ckpt.dtype(name), partial(ckpt.tensor, name))
                    )
        return plan + self._stitch_entries()

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
        self._check_same_parameters([ckpt.names() for ckpt in experts])
        for ckpt in experts:
            self._tied_weights_keys.extend(ckpt.tied_keys)
            self.model_configs.append(ckpt.config.to_dict())
            self.generation_configs.append(ckpt.generation_config)
            self.model_classes.append((ckpt.model_class_name, type(ckpt.config).__name__))

        if self.config.get("moe_method") == "bts":
            self._check_bts_vocabularies()
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
        configs = [
            AutoConfig.from_pretrained(expert["model_id"], trust_remote_code=True)
            for expert in self.config["experts"]
        ]
        if all(config.get_text_config() is config for config in configs):
            return
        text_types = {type(config.get_text_config()) for config in configs}
        if len(text_types) > 1:
            raise ValueError(
                "The experts have different language model types: "
                f"{sorted(t.__name__ for t in text_types)}"
            )
        try:
            text_cls = MODEL_FOR_CAUSAL_LM_MAPPING[text_types.pop()]
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
            if self.device != "cpu" or self.device_map is not None:
                print("Streaming merges on CPU, the device and device_map arguments are ignored.")
            try:
                self._compose_streaming(router_layers)
                return
            except NotStreamable as reason:
                print(f"Streaming disabled ({reason}), loading the full experts.")
                self._plan = None
                self._tied_weights_keys = []
                self.model_configs = []
                self.generation_configs = []
                self.model_classes = []
                self.select_moe_model_config_idx = 0
                self._matched_router_layers = set()

        names_per_expert = []
        for expert_idx, expert in enumerate(experts):
            model = self._load_expert(expert["model_id"])
            print(f"merging expert : {expert['model_id']}")
            self._record_metadata(model)
            weights = model.state_dict()
            names_per_expert.append(list(weights))
            self._check_same_parameters(names_per_expert)

            if moe_method == "bts":
                self._check_bts_vocabularies()
                self._add_bts_expert(weights, expert_idx)
            else:
                n_routed, n_averaged = self._add_btx_expert(
                    weights, expert_idx, len(experts), router_layers
                )
                print(f"routed tensors : {n_routed} | averaged tensors : {n_averaged}")

            # Release the expert before loading the next one
            del model, weights
            gc.collect()

        if moe_method == "bts":
            for name, _, _, init in self._stitch_entries():
                self.state_dict[name] = init().to(self.device)

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
        if len(missing) == len(router_layers):
            raise ValueError(
                f"None of the router layers {router_layers} is in the selected blocks of the "
                "experts: check router_layers and router_layers_index"
            )
        if missing:
            print(f"Router layers not found in the experts, ignored: {missing}")
        return [layer for layer in router_layers if layer in self._matched_router_layers]

    def _moe_config(self):
        """
        This function builds the configuration of the merged model from the
        configuration of the selected expert.

        Returns
        -------
        The new configuration as a dict.
        """
        config = dict(self.model_configs[self.select_moe_model_config_idx])
        moe_method = self.config["moe_method"]
        n_blocks = config.get("n_layer", config.get("num_hidden_layers"))
        selected = self.config["router_layers_index"]
        if moe_method == "traditional" or not selected:
            router_layers_index = list(range(n_blocks))
        else:
            router_layers_index = sorted(
                i for i in set(selected) if i is not None and 0 <= i < n_blocks
            )
        config.update(
            {
                "model_type": self.config["model_type"],
                "architectures": ["MixtureKitForCausalLM"],
                "auto_map": {
                    "AutoConfig": f"configuration_{self.config['model_type']}.MixtureKitConfig",
                    "AutoModelForCausalLM": f"modeling_{self.config['model_type']}.MixtureKitForCausalLM",
                },
                "base_model_class": self.model_classes[self.select_moe_model_config_idx][0],
                "moe_method": moe_method,
                "num_experts": len(self.config["experts"]),
                # Shown by the routing visualizer
                "expert_names": [
                    expert.get("expert_name") or expert["model_id"] for expert in self.config["experts"]
                ],
                "expert_model_ids": [expert["model_id"] for expert in self.config["experts"]],
                # Local directories are never listed as Hub models (model card)
                "expert_is_local": [os.path.isdir(expert["model_id"]) for expert in self.config["experts"]],
                "num_experts_per_tok": self.config["num_experts_per_tok"],
                "alpha": self.config.get("alpha", 0.0),
                "router_layers": self._router_layers(),
                "router_layers_index": router_layers_index,
                "_name_or_path": "",
            }
        )
        _add_legacy_rope_fields(config, self.model_classes[self.select_moe_model_config_idx][1])
        if moe_method == "bts":
            config["stitch_freq"] = self.config.get("stitch_freq", 5)
            if self.config.get("stitch_layers_index"):
                config["stitch_layers_index"] = list(self.config["stitch_layers_index"])
        return config

    def save_checkpoint(self, checkpoint_path):
        """
        This function saves the composed unified checkpoint as safe tensors in shards (chuncks),
        with its configuration, tokenizer and modeling files.

        Parameters
        ----------
        checkpoint_path: str
            The path to save the composed unified checkpoint.
        """
        model_type = self.config["model_type"]
        if model_type in CONFIG_MAPPING:
            raise ValueError(
                f"model_type '{model_type}' is already a transformers model type, choose another name"
            )
        os.makedirs(checkpoint_path, exist_ok=True)
        config = self._moe_config()
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
        _make_tokenizer_config_portable(checkpoint_path)

        ##### Modeling and configuration files #####
        model_class, config_class = self.model_classes[self.select_moe_model_config_idx]
        write_modeling_files(checkpoint_path, model_type, model_class, config_class)
        with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
            json.dump(config, f, indent=1)
        # Only the values that are set: transformers 5 lists every field, and its null
        # values are rejected by transformers 4
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        with open(os.path.join(checkpoint_path, "generation_config.json"), "w") as f:
            json.dump(generation_config, f, indent=1)
        print(f"checkpoint saved at {checkpoint_path}")


def write_modeling_files(checkpoint_path, model_type, model_class, config_class):
    """
    This function writes the modeling and configuration files of a merged model:
    small files subclassing the transformers classes of the experts, so the merged
    model follows the installed transformers version.

    Parameters
    ----------
    checkpoint_path: str
        The directory of the merged model.
    model_type: str
        The model type of the merged model.
    model_class: str
        The transformers causal LM class of the experts, e.g. "Qwen3ForCausalLM".
    config_class: str
        The transformers configuration class of the experts, e.g. "Qwen3Config".
    """
    for name in (model_class, config_class):
        if not hasattr(transformers, name):
            raise ValueError(
                f"{name} is not a transformers class: experts with custom (remote) "
                "code are not supported."
            )

    configuration = CONFIGURATION_TEMPLATE.format(config_class=config_class, model_type=model_type)
    with open(os.path.join(checkpoint_path, f"configuration_{model_type}.py"), "w") as f:
        f.write(configuration)

    template = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeling_template.py")
    with open(template, "r", encoding="utf-8") as f:
        modeling = f.read()
    modeling = modeling.replace("__CONFIG_MODULE__", f"configuration_{model_type}")
    modeling = modeling.replace("__BASE_MODEL_CLASS__", model_class)
    with open(os.path.join(checkpoint_path, f"modeling_{model_type}.py"), "w") as f:
        f.write(modeling)


def _make_tokenizer_config_portable(checkpoint_path):
    """
    transformers 5 saves the extra special tokens as a list, which transformers 4 reads
    as a mapping: they are stored under `additional_special_tokens`, read by both.
    """
    path = os.path.join(checkpoint_path, "tokenizer_config.json")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
    extra = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return
    tokenizer_config.pop("extra_special_tokens")
    if extra:
        additional = tokenizer_config.get("additional_special_tokens") or []
        tokenizer_config["additional_special_tokens"] = additional + [t for t in extra if t not in additional]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)


def _legacy_rope_fields(config):
    """
    transformers 5 stores the rotary embedding settings in `rope_parameters`, possibly per
    attention type (Gemma3). transformers 4 reads `rope_theta` / `rope_scaling` (and
    `rope_local_base_freq` for Gemma3) and would silently use its defaults: they are added
    so that a model built with transformers 5 loads the same with transformers 4.
    """
    rope = config.get("rope_parameters")
    if not isinstance(rope, dict):
        return {}
    legacy = {}
    per_attention = isinstance(rope.get("full_attention"), dict)
    full = rope["full_attention"] if per_attention else rope
    if "rope_theta" in full:
        legacy["rope_theta"] = full["rope_theta"]
    scaling = {k: v for k, v in full.items() if k != "rope_theta"}
    if scaling.get("rope_type", "default") != "default":
        legacy["rope_scaling"] = scaling
    if per_attention and isinstance(rope.get("sliding_attention"), dict):
        local = rope["sliding_attention"].get("rope_theta")
        if local is not None:
            legacy["rope_local_base_freq"] = local
    return {k: v for k, v in legacy.items() if k not in config}


def _add_legacy_rope_fields(config, config_class=None):
    """
    Adds the transformers 4 rotary fields (see _legacy_rope_fields) when the configuration
    class of the experts reads them back without changing the rotary settings. Families that
    only exist in transformers 5 (e.g. Gemma 4) would read them as an override.
    """
    legacy = _legacy_rope_fields(config)
    if not legacy:
        return
    cls = getattr(transformers, config_class, None) if config_class else None
    if cls is not None:
        try:
            before = cls.from_dict(dict(config)).to_dict().get("rope_parameters")
            after = cls.from_dict({**config, **legacy}).to_dict().get("rope_parameters")
        except Exception:
            return
        if before != after:
            return
    config.update(legacy)
    # Saved again by the configuration class when transformers 5 saves the model (training)
    config["mixturekit_legacy_rope"] = legacy


CONFIGURATION_TEMPLATE = '''"""
Configuration of a Mixture-of-Experts model built with MixtureKit: the configuration
of the experts with the Mixture-of-Experts fields (moe_method, num_experts, ...).
"""

import transformers


class MixtureKitConfig(getattr(transformers, "{config_class}")):
    model_type = "{model_type}"

    def to_dict(self):
        output = super().to_dict()
        # The rotary fields read by transformers 4, which transformers 5 does not save
        for key, value in (output.get("mixturekit_legacy_rope") or {{}}).items():
            output.setdefault(key, value)
        return output


# save_pretrained copies this file next to the weights
MixtureKitConfig.register_for_auto_class()
'''


def build_moe(
    config,
    torch_dtype=torch.float16,
    model_cls=AutoModelForCausalLM,
    streaming=True,
    output_dir=None,
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
    output_dir: str, default=None
        Where to write the merged model, models_merge/<model_type> by default.

    Returns
    -------
    The directory of the merged model.
    References
    ----------
        .. footbibliography::
    """
    output_dir = output_dir or os.path.join("models_merge", config["model_type"])
    build_config = json.loads(json.dumps(config, default=str))  # kept as given
    expertmerger = MoeExperts(
        config, torch_dtype=torch_dtype, model_cls=model_cls, streaming=streaming
    )
    expertmerger.compose()
    expertmerger.save_checkpoint(output_dir)
    # The configuration it was built from, to rebuild it or describe it (model card)
    with open(os.path.join(output_dir, "mixturekit_config.json"), "w") as f:
        json.dump(build_config, f, indent=2)
    return output_dir
