import gc
import re
import os
from pathlib import Path
import shutil
import transformers
from transformers.utils import TRANSFORMERS_CACHE


def _get_config_files(model_id, destination, new_model_type, model_cls):
    """
    This function retrieves the configuration files of the base model
    in order to prepare it for the automatic/manual alterations either
    from the local cache (remote_code) or from the transformers library

    Parameters
    ----------
    model_id: str
        The id of the corresponding model from HF.
    destionation: str
        The path of the local destination directory.
    new_model_type: str
        The name of the new configured model with MoE.
    model_cls: type, default=AutoModelForCausalLM
        Change this when using a architecture not registered with transformers.

    Returns
    -------
        The relative paths of the new modeling and configuration files.
    """
    os.makedirs(destination, exist_ok=True)
    model_base = model_cls.from_pretrained(model_id, trust_remote_code=True)
    config_path = str(model_base.config_class)
    if "transformers." in config_path:
        transformers_path = Path(transformers.__file__).parents[0]
        model_path = transformers_path / "models" / config_path.split(".")[2]
    else:
        cache_path = Path(TRANSFORMERS_CACHE)
        model_path = cache_path.parents[0] / "modules" / "transformers_modules"
        model_path = model_path / model_id.split("/")[0] / model_id.split("/")[1]
    files = [str(file) for file in model_path.glob("**/*.py") if file.is_file()]
    # Multi-Modal Case
    modeling_file = [
        file
        for file in files
        if f"modeling_{model_base.config.to_dict()['model_type'].split('_text')[0]}"
        in file
    ][0]
    configuration_file = [
        file
        for file in files
        if f"configuration_{model_base.config.to_dict()['model_type'].split('_text')[0]}"
        in file
    ][0]
    # Copy the configuration files of the base model to the directory of the new MoE model
    shutil.copy(modeling_file, destination)
    shutil.copy(configuration_file, destination)
    # Rename the configuration files
    new_modeling_file = f"modeling_{new_model_type}.py"
    new_configuration_file = f"configuration_{new_model_type}.py"

    del model_base
    # Manually clears the memory from the loaded model
    gc.collect()

    os.rename(
        f"{destination}/{modeling_file.split('/')[-1]}",
        f"{destination}/{new_modeling_file}",
    )
    os.rename(
        f"{destination}/{configuration_file.split('/')[-1]}",
        f"{destination}/{new_configuration_file}",
    )
    return (
        f"{destination}/{new_modeling_file}",
        f"{destination}/{new_configuration_file}",
    )


def _modify_decoder(match, moe_method):
    """
    This function modifies the DecoderLayer class with the following steps:

    1. Expose `loss_lb_attn` from the attention call
    2. Add gate & softmax after class initialization
    3. Replace the single `self.mlp(...)` line with the MoE branch
    4. Aggregate original and load losses

    Parameters
    ----------
    match: re.Match
        The matching parts in the original script.
    moe_method: str
        The applied mixture of experts (moe) architecture.
    """

    # --- original Decoder body ---
    class_body = match.group(2)

    # 1) expose attention loss
    class_body = class_body.replace(
        " = self.self_attn(\n",
        ", loss_lb_attn = self.self_attn(\n",
    )

    if moe_method == "traditional":
        # 2) Insert gate/softmax directly after class initialization
        gate_block = """
        # === Added for traditional-MoE ===
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
"""
        class_body = class_body.replace(
            "super().__init__()\n",
            "super().__init__()" + gate_block,
        )

        # 3) Replace the single old MLP call with the MoE branch
        moe_block = """
        # === MoE feed-forward ===
        global cache_selected_experts
        weights, selected_experts = torch.topk(self.gate(hidden_states), self.config.num_experts_per_tok)
        weights = torch.softmax(weights, dim=-1, dtype=torch.float).to(hidden_states.dtype)
        hidden_accum = torch.zeros_like(hidden_states)
        for ix in range(self.config.num_experts):
            batch_idx, token_idx, expert_idx = torch.where(selected_experts == ix)
            cache_selected_experts = {'batch_idx': batch_idx, 'token_idx': token_idx, 'expert_idx': ix}
            mlp_out, loss_lb_mlp = self.mlp(hidden_states)
            hidden_accum[batch_idx, token_idx] += mlp_out * weights[batch_idx, token_idx, expert_idx].unsqueeze(-1)
            hidden_states = hidden_accum
"""
        class_body = re.sub(
            r"^[ \t]*hidden_states\s*=\s*self\.mlp\([^\n]*\)\n",
            moe_block,
            class_body,
            count=1,
            flags=re.MULTILINE,
        )

    else:
        class_body = class_body.replace(" = self.mlp(", ", loss_lb_mlp = self.mlp(")

    # 4) Introduce load loss as part of the total loss
    class_body = re.sub(
        r"(^[ \t]*)return[^\n]+",
        r"\1self.loss_lb = loss_lb_attn + loss_lb_mlp\n\g<0>",
        class_body,
        flags=re.MULTILINE,
    )

    # splice the modified body back in
    return match.group(1).replace(match.group(2), class_body)


##### Model or TextModel Class #####
def _modify_model(match, update_kv=False):
    """
    This function modifies the Model or TextModel class.

    Parameters
    ----------
    match: re.Match
        The matching parts in the original script.
    """
    class_body = match.group(2)
    # Initialze an empty tuple for accumulating the losses across layers
    new_class_body = class_body.replace(
        "for decoder_layer", "all_losses_lb = ()\n        for decoder_layer"
    )

    # Inject the requires_grad_() line **with correct indent**
    new_class_body = re.sub(
        r"(^[ \t]*)if self\.gradient_checkpointing and self\.training:",
        lambda m: (
            f"{m.group(0)}\n"
            f"{m.group(1)}    hidden_states = hidden_states.requires_grad_()"
        ),
        new_class_body,
        flags=re.MULTILINE,
    )

#     # Fixing the kv cache
#     cache_pattern = r"""(?s)(?P<indent>[ \t]*)if\s+use_cache\s+and\s+past_key_values\s+is\s+None\s+and\s+not\s+self\.training:\s+.*?past_key_values\s*=\s*(?P<init>\w+\([^\n]*?(?:\n(?P=indent)[ \t]+[^\n]*)*\))"""
#     cache_replacement = r"""\g<0>

# \g<indent>past_key_values = [\g<init> for _ in range(self.config.num_experts)]
#     """
#     new_class_body = re.sub(cache_pattern, cache_replacement, new_class_body)

#     new_class_body = new_class_body.replace(
#         "past_key_values.get_seq_length()", "past_key_values[0].get_seq_length()"
#     )

#     if update_kv:
#         new_class_body = new_class_body.replace(
#             "past_key_values,", "past_key_values[0],"
#         )

#     return_pattern = (
#         r"(return\s+ExtendedModelOutputWithPast\([^)]*past_key_values)\[0\]([^)]*\))"
#     )
#     return_replacement = r"\1\2"
#     new_class_body = re.sub(
#         return_pattern, return_replacement, new_class_body, flags=re.DOTALL
#     )

    # The variables should be integrated inside a list for the later sum
    new_class_body = new_class_body.replace(
        "hidden_states = layer_outputs[0]\n",
        "hidden_states = layer_outputs[0]\n            all_losses_lb += (decoder_layer.loss_lb,)\n",
    )
    # Expand the output class with a new argument for the load balancing loss
    pattern_model = r"(return\s+ExtendedModelOutputWithPast\s*\(([^)]*))(\))"
    replacement = r"\1    loss_lb=sum(all_losses_lb)\3"
    new_class_body = re.sub(pattern_model, replacement, new_class_body, flags=re.DOTALL)
    return match.group(1).replace(class_body, new_class_body)


def _patch_stitches(m: re.Match, model_type) -> str:
    """
    This function integrates the stitch layers.

    Parameters
    ----------
    match: re.Match
        The matching parts in the original script.
    model_type: str
        The defined type of the new model.
    """
    header = m.group(1)
    body = m.group("body")
    lines = body.splitlines()

    # Now find a good place to insert the stitch code (after the layers are defined)
    insert_idx = -1
    for idx, line in enumerate(lines):
        if "self.norm =" in line:
            insert_idx = idx
            break

    if insert_idx > 0:
        # Create the stitch initialization code with proper indentation
        stitch_code = f"""
        # Initialize expert layers if BTS method is used
        if getattr(config, 'moe_method', None) == 'bts':
            # Initialize expert embed tokens
            import copy
            self.expert_embed_tokens = nn.ModuleList([copy.deepcopy(self.embed_tokens) for _ in range(config.num_experts - 1)])

            # Initialize expert layers
            self.expert_layers = nn.ModuleList([nn.ModuleList([ {model_type.capitalize()}DecoderLayer(config, layer_idx) for _ in range(config.num_experts - 1)])
                for layer_idx in range(config.num_hidden_layers)])

            # Initialize stitch layers
            stitch_freq = getattr(config, 'stitch_freq', 5)
            stitch_layers = getattr(config, 'stitch_layers_index', None) or [i for i in range(config.num_hidden_layers) if i % stitch_freq == 0 and i != 0]
            if (config.num_hidden_layers - 1) not in stitch_layers:
                stitch_layers.append(config.num_hidden_layers - 1)

            # Initialize stitch layers at appropriate transformer layers
            toggle = False  # First = Hub -> Experts
            for layer_idx in stitch_layers:
                # The last layer of the model should be a Experts-into-Hub stitch layer
                if layer_idx == (config.num_hidden_layers - 1):
                    toggle = True
                self.layers[layer_idx].stitches = StitchLayer(config.hidden_size, config.num_experts, merge_into_hub=toggle)
                # PairSwitch between merging into hub or experts
                toggle = not toggle
            """
        lines.insert(insert_idx, stitch_code)
        new_body = header + "\n".join(lines)

        extract_pattern = r"all_hidden_states\s*\+=\s*\(hidden_states,\)\s*\n(.*?)\n\s*hidden_states\s*=\s*layer_outputs\[0\]"
        match = re.search(extract_pattern, new_body, re.DOTALL)

        raw_block = match.group(1)
        inner_block = "\n".join("        " + line for line in raw_block.splitlines())
        inner_block = inner_block.replace("layer_outputs", "e_out")
        inner_block = inner_block.replace("hidden_states", "exp_in")
        # inner_block = inner_block.replace(
        #     "past_key_values,", "past_key_values[e_idx+1],"
        # )
        inner_block = re.sub(
            r"decoder_layer(?!\.attention_type)", "e_layer", inner_block
        )

        # Add forward pass handling for stitches
        for idx, line in enumerate(lines):
            if "hidden_states = layer_outputs[0]" in line:
                # Inject new parallel hub+experts code
                stitch_code = f"""
            # === Hub forward ===
            # === BTS forward ===
            if getattr(self.config, 'moe_method', None) == 'bts':
                bts_method = True
                if decoder_layer.layer_idx == 0:
                    expert_states = [self.expert_embed_tokens[i](input_ids) for i in range(self.config.num_experts - 1)]

                new_expert = []
                for e_idx, e_layer in enumerate(self.expert_layers[decoder_layer.layer_idx]):
                    exp_in = expert_states[e_idx]
{inner_block}
                    new_expert.append(e_out[0])
                expert_states = new_expert

                # Stitching
                if hasattr(decoder_layer, 'stitches'):
                    streams = [hidden_states] + expert_states
                    streams = decoder_layer.stitches(streams)
                    hidden_states, expert_states = streams[0], streams[1:]

                """
                # For transformers < 4.53.0
                stitch_code = re.sub(
                    r"^([ \t]*)if self\.gradient_checkpointing and self\.training:\s*$",
                    r"\1if self.gradient_checkpointing and self.training and bts_method:\n\1    exp_in = exp_in.requires_grad_()",
                    stitch_code,
                    flags=re.MULTILINE,
                )
                lines.insert(idx + 1, stitch_code)
                break

        return header + "\n".join(lines)


# Function to replace nn.Linear and Conv1D while keeping indentation and detecting the layer name
def _replace_script(match, mlp_function="linear", block_transformer="mlp"):
    """
    This function takes the input matches using regex to replace the corresponding layers
    with the conver_linear_to_moe layers.

    Parameters
    ----------
    match: re.Match
        The matching parts in the original script.
    mlp_function: str, default='linear'
        The mlp function used in the corresponding layer.
    block_transformer: str, default='mlp'
        Whether this layer belongs to the attention part or Feed-Forward part
        especially in the case of common names.
    """
    indentation = match.group(1)  # Preserve the leading spaces/tabs
    layer_name = match.group(2)  # Extract layer name (e.g., "q_proj", "k_proj")
    input_dim = match.group(3)  # First argument (input dimension)
    output_dim = match.group(4)  # Second argument (output dimension)
    bias_arg = match.group(5)  # Third argument (e.g., "bias=config.attention_bias")

    # Construct the replacement string while keeping indentation
    if bias_arg:
        return f"{indentation}self.{layer_name} = convert_linear_to_moe('{layer_name}', config, layer_idx, \
            {input_dim}, {output_dim}, {bias_arg}, mlp_function='{mlp_function}', block_transformer='{block_transformer}')"
    else:
        return f"{indentation}self.{layer_name} = convert_linear_to_moe('{layer_name}', config, layer_idx, \
            {input_dim}, {output_dim}, mlp_function='{mlp_function}', block_transformer='{block_transformer}')"


def _replace_type_dependency(input, old_model_type, new_model_type):
    """
    This function replaces the old model type with the new model type
    across the different configuration files also for the relative dependencies.

    input: str
        The input to modify.
    old_model_type: str
        The name or id of the old model type.
    new_model_type: str
        The name or id of the new model type.
    """
    # If the first character is upper
    input = input.replace(
        f"{old_model_type[0].upper() + old_model_type[1:]}",
        f"{new_model_type[0].upper() + new_model_type[1:]}",
    )
    # If the whole name is upper
    input = input.replace(old_model_type.upper(), new_model_type.upper())
    # If the whole name is lower
    input = input.replace(old_model_type, new_model_type)
    input = "from functools import partial\n" + input
    input_dependencies = re.sub(
        r"from\s+\.\.\.", lambda example: "from transformers.", input
    )
    input_dependencies = input_dependencies.replace("..siglip", "transformers")
    input_dependencies = input_dependencies.replace("..auto", "transformers")
    return input_dependencies
