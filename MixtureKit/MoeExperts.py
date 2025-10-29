import black
import gc
from huggingface_hub import save_torch_state_dict
import json
import os
import re
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from .utils import (
    _get_config_files,
    _modify_decoder,
    _modify_model,
    _patch_stitches,
    _replace_script,
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
            The maximum Shard size checkpoint chuncks.
        model_cls: type, default=AutoModelForCausalLM
            Change this when using a architecture not registered with transformers.
        """
        self.config = config
        self.model_configs = []
        self.generation_configs = []
        self.torch_dtype = torch_dtype
        self._tied_weights_keys = []
        self.device = device
        self.device_map = device_map
        self.max_shard_size = max_shard_size
        self.model_cls = model_cls
        # An empty list is returned if the corresponding key is not part of the dictionary
        self.config["router_layers_index"] = self.config.get("router_layers_index", [])
        self.moe_layer_index = self.config["router_layers_index"]
        self.select_moe_model_config_idx = 0
        self._set_moe_layer_index()

    def _set_moe_layer_index(self):
        """
        This function lists the transformer blocks (under the provided id) for moe conversion.

        Returns
        -------
        A lambda function returning a bool value mirroring the indices of the corresponding layers
        for the moe conversion.
        """
        if len(self.moe_layer_index) == 0:
            self._check_moe_layers = lambda x: True
            print(f"MoE Layer Index : [*]")

        elif len(self.moe_layer_index) >= 1 and self.moe_layer_index[0] is not None:
            self._check_moe_layers = lambda x: x in self.moe_layer_index
            print(f"MoE Layer Index : {self.moe_layer_index}")

        else:
            self._check_moe_layers = lambda x: False
            print(f"No MoE layer indexes.")

    def _is_layer_suitable_for_router(self, router_layer, model_layer):
        """
        This functions checks whether the provided layer is suitable for the moe conversion.

        Parameters
        ----------
        router_layer: str
            The corresponding layer listed among the router layers, i.e. the layers to convert.
        model_layer: str
            The corresponding layer among the model layers.

        Returns
        -------
        A boolean value reflecting the suitability of the layer for the moe conversion.
        """
        # Checks whether the layer id contains a digit refering to the transformer block
        # and if the corresponding block are among the specified for moe conversion.

        model_layer_index = [int(x) for x in model_layer.split(".") if x.isdigit()]

        if not model_layer_index:
            valid_layer_index = False
        else:
            assert len(model_layer_index) == 1
            valid_layer_index = self._check_moe_layers(model_layer_index[0])

        if (f".{router_layer}." in model_layer) and valid_layer_index:
            if "mlp" in model_layer or "self_attn" in model_layer:
                return True
        return False

    def _load_base_model(self, model_id):
        """
        This function loads the requested model using its id and move it to the requested device.

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
        return model.to(self.device)

    def _shape_adjuster(self, tensor1, tensor2, ix):
        """
        This function adjusts the shapes of both tensors for the averaging process.

        Parameters
        ---------
        tensor1: torch.tensor
            The current tensor contatining the previous weights.
        tensor2: torch.tensor
            The tensor of the new weights to add to the final average.
        ix: int
            The index of the corresponding model block.

        """
        assert tensor1.ndim == tensor2.ndim
        if tensor1.shape[0] < tensor2.shape[0]:
            pad_tensor = torch.zeros_like(
                tensor2, dtype=self.torch_dtype, device=tensor1.device
            )
            pad_tensor[: tensor1.shape[0]] += tensor1
            tensor1 = pad_tensor
            self.select_moe_model_config_idx = ix
        else:
            pad_tensor = torch.zeros_like(
                tensor1, dtype=self.torch_dtype, device=tensor2.device
            )
            pad_tensor[: tensor2.shape[0]] += tensor2
            tensor2 = pad_tensor
        return tensor1, tensor2

    def compose(self):
        """
        This functon composes all the experts into a single unified checkpoint
        with or without a Mixture of Experts (MoE) conversion.
        """
        n = len(self.config["experts"])
        self.state_dict = {}
        count_total_router_layers = 0
        for ix, expert in enumerate(self.config["experts"]):
            model_id = expert["model_id"]
            model = self._load_base_model(model_id)
            print(f"merging expert : {model_id}")

            # Weight tying allows certain weights to be used in multiple places.
            # A common application of weight tying in LLMs is to share the weights
            # between the input embedding layer and the output prediction layer.
            if hasattr(model, "_tied_weights_keys"):
                if model._tied_weights_keys != None:
                    self._tied_weights_keys.extend(model._tied_weights_keys)
            self.model_configs.append(model.config.to_dict())
            self.generation_configs.append(model.generation_config.to_dict())
            router_layers = self.config["router_layers"]

            if self.config.get("moe_method") == "traditional":
                router_layers = ["mlp"]

            count_router_layers = 0
            count_averaged_layers = 0
            for layer_name, param in tqdm(model.state_dict().items()):

                # --- BTS: keep every expert intact ---
                if self.config["moe_method"] == "bts":
                    weight_bias = layer_name.split(".")[-1]  # "weight" | "bias"
                    base_path = layer_name.split(f".{weight_bias}")[0]

                    # HUB  – take expert-0 as the shared backbone
                    if ix == 0:
                        self.state_dict[layer_name] = param.to(self.device)
                    # EXTRA EXPERTS  – keep only transformer-block tensors
                    else:
                        if "model.layers." in base_path:
                            layer_part = base_path.split("model.layers.", 1)[
                                1
                            ]  # "19.self_attn.q_proj"
                            layer_id, attr_rest = layer_part.split(
                                ".", 1
                            )  # "19", "self_attn.q_proj"

                            expert_idx = ix - 1  # experts start from 0
                            expert_key = f"model.expert_layers.{layer_id}.{expert_idx}.{attr_rest}.{weight_bias}"
                            self.state_dict[expert_key] = param.to(self.device)
                        elif "embed_tokens" in base_path:
                            expert_idx = ix - 1  # experts start from 0
                            expert_key = (
                                f"model.expert_embed_tokens.{expert_idx}.{weight_bias}"
                            )
                            self.state_dict[expert_key] = param.to(self.device)

                else:  ##### BTX #####
                    is_merge_layer = True
                    for router_layer in router_layers:
                        # Convert specified router layers to moe
                        if self._is_layer_suitable_for_router(router_layer, layer_name):
                            is_merge_layer = False
                            weight_bias = layer_name.split(".")[-1]
                            # Integrate the new layers into the architecture of the unified framework
                            new_layer_name = layer_name.split(f"{weight_bias}")[0]
                            new_layer_name = (
                                f"{new_layer_name}experts.expert_{ix}.{weight_bias}"
                            )
                            assert new_layer_name not in self.state_dict
                            self.state_dict[new_layer_name] = param.to(self.device)
                            count_total_router_layers += 1
                            count_router_layers += 1

                    # Average the not-specified and/or not suitable layers fsor conversion
                    if is_merge_layer:
                        prev_weight = self.state_dict.get(layer_name)
                        if prev_weight is None:
                            prev_weight = torch.tensor(0)
                        else:
                            # Padding the smaller tensor to adjust the average computation
                            if not prev_weight.shape == param.shape:
                                prev_weight, param = self._shape_adjuster(
                                    prev_weight, param, ix
                                )

                        try:
                            self.state_dict[layer_name] = prev_weight + (param / n).to(
                                self.device
                            )
                        except Exception as e:
                            print(layer_name, param)
                            self.state_dict[layer_name] = param.to(self.device)
                        count_averaged_layers += 1

        if self.config["moe_method"] == "btx":
            print(f"count_averaged_layers : {count_averaged_layers}")
            print(f"count_router_layers : {count_router_layers}")
            print(f"count_total_router_layers : {count_total_router_layers}")
        del model
        # Manually clears the memory from the loaded model
        gc.collect()

    def save_checkpoint(self, checkpoint_path):
        """
        This function saves the composed unified checkpoint as safe tensors in shards (chuncks).

        Parameters
        ----------
        checkpoint_path: str
            The path to save the composed unified checkpoint.
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        # The configuration json file of the base model to modify
        # The root config is related to vision-based models
        root_config = self.model_configs[self.select_moe_model_config_idx]
        # Config extracts the text part of the vision-based models
        # or the config of the text-based models
        config = root_config.get("text_config", root_config)
        generation_config = self.generation_configs[self.select_moe_model_config_idx]
        config["model_type"] = config["model_type"].split("_text")[0]
        old_model_type = config["model_type"]
        config["alpha"] = self.config["alpha"]

        # Introduce the configurations related to the experts
        config["num_experts"] = len(self.config["experts"])
        config["num_experts_per_tok"] = self.config["num_experts_per_tok"]
        config = self.model_configs[self.select_moe_model_config_idx]
        config["model_type"] = config["model_type"].split("_text")[0]
        old_model_type = config["model_type"]
        # Introduce the configurations related to the experts
        config["num_experts"] = len(self.config["experts"])
        config["num_experts_per_tok"] = self.config["num_experts_per_tok"]
        config["alpha"] = self.config["alpha"]
        config["router_layers"] = self.config["router_layers"]
        if config["architectures"] is None:
            config["architectures"] = [old_model_type.capitalize() + "ForCausalLM"]
        config["architectures"][0] = _replace_type_dependency(
            config["architectures"][0], config["model_type"], self.config["model_type"]
        )
        # Change the model_type to the new configured model_type with MoE
        config["model_type"] = self.config["model_type"]
        config["_name_or_path"] = ""

        try:
            layer_indexes = list(range(config["n_layer"]))
        except Exception:
            layer_indexes = list(range(config["num_hidden_layers"]))
        if not self.config["router_layers_index"]:
            config["router_layers_index"] = layer_indexes

        else:
            config["router_layers_index"] = list(
                set(layer_indexes).intersection(set(self.config["router_layers_index"]))
            )
            ##### SafeTensors #####
        # Save all the parameters related to the new configured model
        save_torch_state_dict(
            state_dict=self.state_dict,
            save_directory=checkpoint_path,
            max_shard_size=self.max_shard_size,
            shared_tensors_to_discard=self._tied_weights_keys,
        )

        ##### Tokenizer file #####
        # The tokenizer of the base model
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["experts"][self.select_moe_model_config_idx]["model_id"]
        )
        tokenizer.save_pretrained(checkpoint_path)

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

        with open(
            os.path.join(script_directory, f"{file_path}.py"), "r", encoding="utf-8"
        ) as fh:
            parg_moe = fh.read()
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
                config["auto_map"][key] = _replace_type_dependency(
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
        Change this when using a architecture not registered with transformers.
    References
    ----------
        .. footbibliography::
    """
    os.makedirs("models_merge", exist_ok=True)
    expertmerger = MoeExperts(config, torch_dtype=torch_dtype, model_cls=model_cls)
    expertmerger.compose()
    expertmerger.save_checkpoint(f"models_merge/{config['model_type']}")
