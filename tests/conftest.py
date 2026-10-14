"""
Tiny experts built locally (no download): for each family, three perturbed copies
of a randomly initialized model sharing one word-level tokenizer.
"""

import os
import tempfile
import warnings
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    LlamaConfig,
    PreTrainedTokenizerFast,
    Qwen3Config,
)
from transformers import AutoModelForCausalLM

from MixtureKit import build_moe

warnings.filterwarnings("ignore")

WORDS = "the quick brown fox jumps over lazy dog hello world what is plus answer one two three four".split()
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
N_EXPERTS = 3


def make_tokenizer():
    vocab = {token: i for i, token in enumerate(SPECIAL + WORDS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def family_config(family, vocab_size):
    common = dict(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    if family == "llama":
        return LlamaConfig(**common)
    if family == "qwen3":
        return Qwen3Config(head_dim=16, **common)
    if family == "gemma3":
        return Gemma3TextConfig(
            head_dim=16, sliding_window=4, sliding_window_pattern=2, **common
        )
    raise ValueError(family)


FAMILIES = ["llama", "qwen3", "gemma3"]


@pytest.fixture(scope="session")
def experts(tmp_path_factory):
    """{family: [expert directories]}"""
    root = tmp_path_factory.mktemp("experts")
    tokenizer = make_tokenizer()
    paths = {}
    for family in FAMILIES:
        config = family_config(family, len(tokenizer))
        paths[family] = []
        for idx in range(N_EXPERTS):
            torch.manual_seed(1000 * FAMILIES.index(family) + idx)
            model = AutoModelForCausalLM.from_config(config)
            path = root / f"{family}_{idx}"
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            paths[family].append(str(path))
    # Multimodal Gemma3 (vision tower + language model), merged through the language model
    text = family_config("gemma3", len(tokenizer)).to_dict()
    vision = dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        image_size=28, patch_size=14,
    )
    config = Gemma3Config(
        text_config=text, vision_config=vision, mm_tokens_per_image=4,
        image_token_index=len(tokenizer) - 1, boi_token_index=len(tokenizer) - 2,
        eoi_token_index=len(tokenizer) - 3,
    )
    paths["gemma3mm"] = []
    for idx in range(N_EXPERTS):
        torch.manual_seed(5000 + idx)
        path = root / f"gemma3mm_{idx}"
        Gemma3ForConditionalGeneration(config).save_pretrained(path)
        tokenizer.save_pretrained(path)
        paths["gemma3mm"].append(str(path))
    return paths


def moe_config(method, model_type, expert_paths, **extra):
    config = {
        "moe_method": method,
        "model_type": model_type,
        "num_experts_per_tok": 2,
        "experts": [
            {"expert_name": f"expert_{i}", "model_id": path} for i, path in enumerate(expert_paths)
        ],
        "router_layers": []
        if method == "bts"
        else ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.q_proj"],
        "alpha": 0.01,
        "router_layers_index": [],
    }
    if method == "bts":
        config["stitch_freq"] = 2
    config.update(extra)
    return config


@pytest.fixture(scope="session")
def build(tmp_path_factory):
    """Builds a merged model once per arguments and returns its directory."""
    root = tmp_path_factory.mktemp("builds")
    cache = {}

    def _build(method, expert_paths, name, streaming=True, **extra):
        key = (method, tuple(expert_paths), name, streaming, tuple(sorted(extra.items())))
        if key not in cache:
            workdir = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=root))
            cwd = os.getcwd()
            os.chdir(workdir)
            try:
                build_moe(
                    config=moe_config(method, name, expert_paths, **extra),
                    torch_dtype=torch.float32,
                    streaming=streaming,
                )
            finally:
                os.chdir(cwd)
            cache[key] = str(workdir / "models_merge" / name)
        return cache[key]

    return _build


def load(path, **kwargs):
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.float32, **kwargs
    )


PROMPT = torch.tensor([[2, 4, 5, 6, 7, 8, 9, 4, 10, 11, 4, 5]])
