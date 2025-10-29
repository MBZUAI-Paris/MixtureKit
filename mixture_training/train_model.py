from datasets import load_from_disk
from dataclasses import dataclass, field, fields
import logging
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from typing import List, Optional
import yaml

logger = logging.getLogger(__name__)


@dataclass
class LoraArguments:
    r: int = field(default=8, metadata={"help": "Rank of the LoRA decomposition"})
    lora_alpha: int = field(default=16, metadata={"help": "Scaling factor for LoRA"})
    lora_dropout: float = field(
        default=0.05, metadata={"help": "Dropout probability for LoRA layers"}
    )
    bias: str = field(default="none", metadata={"help": "The bias for LoRA"})
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of target modules for LoRA adaptation"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules to perform full-finetuning \
                           like embedding and lm-head"
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
        metadata={"help": "The chat template to include in SFT if missing from the tokenizer."}
    )
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset"},
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "The column of the dataset containing the text samples"},
    )
    device_map: str = field(
        default=None,
        metadata={"help": "Whether to apply the model parallelism process"},
    )
    instruction_template: str = field(
        default=None,
        metadata={"help": "The instruction template for the CompletionOnlyLM"},
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side for the tokenizer"}
    )
    response_template: str = field(
        default=None,
        metadata={"help": "The response template for the CompletionOnlyLM"},
    )
    is_bts: bool = field(
        default=False,
        metadata={"help": "Check whether the bts architecture is applied"},
    )


def training_function(lora_arguments, script_arguments, training_arguments):

    # Calling the pretrained model configuration and its tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_arguments.model_id)
    tokenizer.padding_side = script_arguments.padding_side
    if tokenizer.chat_template is None:
        tokenizer.chat_template = script_arguments.chat_template
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer vocab size:", tokenizer.vocab_size)

    model = AutoModelForCausalLM.from_pretrained(
        script_arguments.model_id,
        dtype=torch.bfloat16,
        device_map=script_arguments.device_map,
        trust_remote_code=True,
    )

    model.enable_input_require_grads()

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=script_arguments.instruction_template,
        response_template=script_arguments.response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    # Instantiate the SFT configuration
    dict_config = training_arguments.to_dict()
    dict_config["dataset_text_field"] = script_arguments.dataset_text_field
    dict_config["max_seq_length"] = script_arguments.max_seq_length

    args = SFTConfig(**dict_config)

    dataset = load_from_disk(script_arguments.dataset_path)

    for name, weight in model.named_parameters():
        if script_arguments.is_bts: #BTS
            if ".stitches" in name and ".w_proj" in name:
                (
                    nn.init.eye_(weight)
                    if weight.shape[0] == weight.shape[1]
                    else nn.init.xavier_uniform_(weight)
                )
                eye = torch.eye(
                    weight.shape[0], dtype=weight.dtype, device=weight.device
                )
                diff = (weight - eye).abs().max().item()
                weight.requires_grad_(True)
            elif ".stitches" in name and ".w_gate" in name:
                nn.init.zeros_(weight)
                weight.requires_grad_(True)
            else:
                weight.requires_grad_(False)

        else: #BTX
            if "gate." in name:  # keep gate routers trainable
                weight.requires_grad_(True)
            else:
                weight.requires_grad_(False)

    print(f"{'Parameter Name':<60} {'Requires Grad':<15} {'# Params':<10}")
    print("-" * 90)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        print(f"{name:<60} {str(param.requires_grad):<15} {num_params:<10,}")

    print("-" * 90)
    print(f"{'Total parameters:':<75} {total_params:,}")
    print(f"{'Trainable parameters:':<75} {trainable_params:,}")

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()


if __name__ == "__main__":

    # Read the configuration file with all the listed arguments
    with open("mixture_training/config_training.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Parsing the content of the configuration yaml file with HfArguments
    # An alternative could be TrlParser
    parser = HfArgumentParser((LoraArguments, ScriptArguments, TrainingArguments))
    lora_args, script_args, training_args = parser.parse_dict(config)
    training_function(lora_args, script_args, training_args)
