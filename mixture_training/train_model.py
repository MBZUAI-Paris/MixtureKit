"""
Supervised fine-tuning of a merged MoE checkpoint built with MixtureKit.

By default the experts are frozen and only the routers (BTX, traditional) or the
stitch layers (BTS) are trained. With `use_lora: true`, LoRA adapters are also
trained on the projections of the experts, the routers or stitch layers staying
fully trainable.

Usage:
    accelerate launch --config_file mixture_training/deepspeed_config.yaml \\
        mixture_training/train_model.py [path/to/config_training.yaml]
"""

from dataclasses import dataclass, field, fields
import logging
import os
import shutil
import sys
from typing import List, Optional

from datasets import load_from_disk
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

try:
    from MixtureKit.training import moe_loss_function
except ImportError:  # run from a clone without installing the package
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from MixtureKit.training import moe_loss_function
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_training.yaml")


@dataclass
class LoraArguments:
    use_lora: bool = field(
        default=False,
        metadata={"help": "Also train LoRA adapters on the projections of the experts"},
    )
    r: int = field(default=8, metadata={"help": "Rank of the LoRA decomposition"})
    lora_alpha: int = field(default=16, metadata={"help": "Scaling factor for LoRA"})
    lora_dropout: float = field(
        default=0.05, metadata={"help": "Dropout probability for LoRA layers"}
    )
    bias: str = field(default="none", metadata={"help": "The bias for LoRA"})
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Names of the projections receiving LoRA adapters, e.g. q_proj"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Extra modules to fully fine-tune and save with the adapters, "
            "the routers or stitch layers are always included"
        },
    )
    task_type: str = field(
        default="CAUSAL_LM", metadata={"help": "The task type for the LoRA adaptation"}
    )

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class ScriptArguments:
    chat_template: str = field(
        default=None,
        metadata={"help": "The chat template to include in SFT if missing from the tokenizer."},
    )
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset saved with `datasets.save_to_disk`"},
    )
    device_map: str = field(
        default=None,
        metadata={"help": "Whether to apply the model parallelism process"},
    )
    instruction_template: str = field(
        default=None,
        metadata={"help": "The instruction template for the CompletionOnlyLM"},
    )
    model_id: str = field(
        default=None, metadata={"help": "Path or HF Hub id of the merged MoE checkpoint"}
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side for the tokenizer"}
    )
    response_template: str = field(
        default=None,
        metadata={
            "help": "The response template for the CompletionOnlyLM, "
            "the loss is computed on the whole sequence when it is not set"
        },
    )
    is_bts: Optional[bool] = field(
        default=None,
        metadata={"help": "Deprecated, the method is read from the model configuration"},
    )


class ResponseOnlyCollator:
    """
    Pads the tokenized samples and computes the loss on the responses only: the labels
    are masked from each instruction template up to the end of the next response template.
    Works with every trl version (DataCollatorForCompletionOnlyLM was removed from trl).
    """

    def __init__(self, tokenizer, response_template, instruction_template=None):
        self.tokenizer = tokenizer
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        self.response = encode(response_template)
        self.instruction = encode(instruction_template) if instruction_template else None

    @staticmethod
    def _find(ids, pattern):
        n = len(pattern)
        return [i for i in range(len(ids) - n + 1) if ids[i : i + n] == pattern]

    def _labels(self, ids):
        labels = [-100] * len(ids)
        responses = self._find(ids, self.response)
        instructions = self._find(ids, self.instruction) if self.instruction else []
        for start in responses:
            begin = start + len(self.response)
            end = min([i for i in instructions if i >= begin], default=len(ids))
            labels[begin:end] = ids[begin:end]
        return labels

    def __call__(self, features):
        ids = [list(f["input_ids"]) for f in features]
        width = max(len(x) for x in ids)
        pad_id = self.tokenizer.pad_token_id
        left = self.tokenizer.padding_side == "left"
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in ids:
            pad = width - len(x)
            arrange = (lambda seq, value: [value] * pad + seq) if left else (lambda seq, value: seq + [value] * pad)
            batch["input_ids"].append(arrange(x, pad_id))
            batch["attention_mask"].append(arrange([1] * len(x), 0))
            batch["labels"].append(arrange(self._labels(x), -100))
        return {key: torch.tensor(value) for key, value in batch.items()}


def is_router(name):
    """The routers of BTX (one per routed projection) and traditional MoE (one per layer)."""
    return name.endswith(".gate.weight")


def is_stitch(name):
    return ".stitches." in name


@torch.no_grad()
def init_missing_stitches(model, missing_keys):
    """
    Checkpoints built with older MixtureKit versions do not store the stitch layers:
    the projections start as identities and the gates at zero. Stitch layers saved
    with the checkpoint (new builds, previous training runs) are kept as they are.
    """
    missing = set(missing_keys)
    for name, weight in model.named_parameters():
        if not is_stitch(name) or name not in missing:
            continue
        if ".w_proj." in name and weight.shape[0] == weight.shape[1]:
            weight.copy_(torch.eye(weight.shape[0], dtype=weight.dtype))
        elif ".w_proj." in name:
            nn.init.xavier_uniform_(weight)
        else:
            nn.init.zeros_(weight)
        logger.warning("Initialized the missing stitch weight %s", name)


def expert_lora_targets(model, target_modules):
    """
    The full names of the linear layers receiving LoRA adapters. A routed projection
    (e.g. mlp.up_proj.experts.expert_0) is matched by its projection name (up_proj).
    The routers and the stitch layers are never targeted.
    """
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or is_stitch(name + "."):
            continue
        parts = name.split(".")
        projection = parts[-3] if len(parts) >= 3 and parts[-2] == "experts" else parts[-1]
        if projection in target_modules:
            targets.append(name)
    return targets


def training_function(lora_arguments, script_arguments, training_arguments):
    tokenizer = AutoTokenizer.from_pretrained(script_arguments.model_id)
    tokenizer.padding_side = script_arguments.padding_side
    if tokenizer.chat_template is None:
        tokenizer.chat_template = script_arguments.chat_template
    if tokenizer.pad_token is None:
        # Padding positions are removed from the labels: padding with the EOS token
        # would also remove EOS, and the model would not learn to stop
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        if tokenizer.pad_token == tokenizer.eos_token:
            logger.warning("The tokenizer has no pad token, the EOS token is used for padding.")

    if training_arguments.bf16:
        torch_dtype = torch.bfloat16
    elif training_arguments.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model, loading_info = AutoModelForCausalLM.from_pretrained(
        script_arguments.model_id,
        torch_dtype=torch_dtype,
        device_map=script_arguments.device_map,
        trust_remote_code=True,
        output_loading_info=True,
    )
    is_bts = getattr(model.config, "moe_method", None) == "bts"
    if script_arguments.is_bts is not None and script_arguments.is_bts != is_bts:
        logger.warning("is_bts is ignored, the model configuration uses %s", model.config.moe_method)
    if is_bts:
        init_missing_stitches(model, loading_info["missing_keys"])

    is_trainable = is_stitch if is_bts else is_router
    for name, weight in model.named_parameters():
        weight.requires_grad_(is_trainable(name))

    if lora_arguments.use_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = lora_arguments.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            r=lora_arguments.r,
            lora_alpha=lora_arguments.lora_alpha,
            lora_dropout=lora_arguments.lora_dropout,
            bias=lora_arguments.bias,
            target_modules=expert_lora_targets(model, target_modules),
            # The routers or stitch layers are fully trained and saved with the adapters
            modules_to_save=["stitches" if is_bts else "gate"]
            + (lora_arguments.modules_to_save or []),
            task_type=lora_arguments.task_type,
        )
        model = get_peft_model(model, lora_config)

    # The frozen weights stay in low precision, the trained ones are kept in float32:
    # mixed precision computes in bf16/fp16 but the optimizer updates need float32.
    # DeepSpeed keeps its own float32 copies of the trained weights.
    uses_deepspeed = bool(training_arguments.deepspeed) or (
        os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
    )
    if not uses_deepspeed:
        for param in model.parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.float()

    if training_arguments.gradient_checkpointing:
        model.enable_input_require_grads()

    collator = None
    if script_arguments.response_template:
        collator = ResponseOnlyCollator(
            tokenizer,
            script_arguments.response_template,
            script_arguments.instruction_template,
        )

    dataset = load_from_disk(script_arguments.dataset_path)

    print(f"{'Parameter Name':<70} {'Requires Grad':<15} {'# Params':<10}")
    print("-" * 100)
    total_params, trainable_params = 0, 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"{name:<70} {str(param.requires_grad):<15} {param.numel():<10,}")
    print("-" * 100)
    print(f"{'Total parameters:':<85} {total_params:,}")
    print(f"{'Trainable parameters:':<85} {trainable_params:,}")

    # The loss below replaces trl's: its chunked loss (trl 1.x) swaps the model forward
    if getattr(training_arguments, "loss_type", "nll") == "chunked_nll":
        training_arguments.loss_type = "nll"

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_loss_func=moe_loss_function(model),
        callbacks=[ModelingFilesCallback(script_arguments.model_id)],
    )
    trainer.train(resume_from_checkpoint=training_arguments.resume_from_checkpoint)
    # Also saves the tokenizer (processing_class) from the main process
    trainer.save_model(training_arguments.output_dir)
    if trainer.is_world_process_zero():
        copy_modeling_files(script_arguments.model_id, training_arguments.output_dir)


class ModelingFilesCallback(TrainerCallback):
    """Copies the modeling files into every saved checkpoint, so that each one can be loaded."""

    def __init__(self, model_id):
        self.model_id = model_id

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(checkpoint):
                copy_modeling_files(self.model_id, checkpoint)


def copy_modeling_files(model_id, output_dir):
    """
    The merged model is loaded with trust_remote_code from its own modeling and
    configuration files, they are copied next to the trained weights so that the
    output can be loaded (or trained again) on its own.
    """
    if os.path.isdir(model_id):
        source = model_id
    else:
        from huggingface_hub import snapshot_download

        source = snapshot_download(model_id, allow_patterns=["*.py"])
    for file in os.listdir(source):
        if file.endswith(".py") and not os.path.exists(os.path.join(output_dir, file)):
            shutil.copy(os.path.join(source, file), output_dir)


def main(config_path=DEFAULT_CONFIG):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    parser = HfArgumentParser((LoraArguments, ScriptArguments, SFTConfig))
    lora_args, script_args, training_args = parser.parse_dict(config)
    training_function(lora_args, script_args, training_args)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
