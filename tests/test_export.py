"""Export of traditional MoE models to Mixtral and Qwen3-MoE."""

import json

import pytest
import torch
from transformers import AutoModelForCausalLM

from MixtureKit.export import export_native
from conftest import PROMPT, load


@pytest.mark.parametrize("family, architecture", [("llama", "MixtralForCausalLM"), ("qwen3", "Qwen3MoeForCausalLM")])
def test_native_model_computes_the_same(experts, build, tmp_path, family, architecture):
    merged_dir = build("traditional", experts[family], f"{family}traditional")
    merged = load(merged_dir).eval()
    # Trained routers: not the ones of a fresh merge
    torch.manual_seed(1)
    for name, param in merged.named_parameters():
        if name.endswith("mlp.gate.weight"):
            param.data.normal_()
    merged.save_pretrained(tmp_path / "trained")
    out = export_native(str(tmp_path / "trained"), str(tmp_path / "native"))

    config = json.load(open(tmp_path / "native" / "config.json"))
    assert config["architectures"] == [architecture] and "auto_map" not in config
    assert not list((tmp_path / "native").glob("*.py"))  # no remote code
    native = AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.float32).eval()
    assert type(native).__name__ == architecture
    with torch.no_grad():
        diff = (native(input_ids=PROMPT).logits - merged(input_ids=PROMPT).logits).abs().max()
        assert diff < 1e-5
        same = dict(max_new_tokens=6, do_sample=False, pad_token_id=0)
        assert torch.equal(native.generate(PROMPT, **same), merged.generate(PROMPT, **same))


@pytest.mark.parametrize("method", ["btx", "bts"])
def test_other_methods_cannot_be_exported(experts, build, tmp_path, method):
    with pytest.raises(ValueError, match="traditional"):
        export_native(build(method, experts["llama"], f"llama{method}"), str(tmp_path / "out"))


def test_unsupported_family_is_explained(experts, build, tmp_path):
    with pytest.raises(ValueError, match="Gemma3ForCausalLM"):
        export_native(build("traditional", experts["gemma3"], "gemma3traditional"), str(tmp_path / "out"))


def _with_routers(experts, build, tmp_path, family="qwen3"):
    """A merged model with its routers saved, as after training."""
    from transformers import AutoTokenizer

    merged = build("traditional", experts[family], f"{family}traditional")
    load(merged).save_pretrained(tmp_path / "trained")
    AutoTokenizer.from_pretrained(merged).save_pretrained(tmp_path / "trained")
    return str(tmp_path / "trained")


def test_export_command(experts, build, tmp_path):
    from MixtureKit.cli import main as cli

    assert cli(["export", _with_routers(experts, build, tmp_path), str(tmp_path / "n")]) == 0
    assert (tmp_path / "n" / "model.safetensors").exists() and (tmp_path / "n" / "tokenizer.json").exists()
    with pytest.raises(SystemExit):
        cli(["export", build("btx", experts["qwen3"], "qwen3btx"), str(tmp_path / "m")])


def test_a_fresh_merge_has_no_routers_to_export(experts, build, tmp_path):
    with pytest.raises(ValueError, match="routers"):
        export_native(build("traditional", experts["qwen3"], "qwen3traditional"), str(tmp_path / "out"))


def test_output_directory_checks(experts, build, tmp_path):
    model = _with_routers(experts, build, tmp_path)
    (tmp_path / "link").symlink_to(model)
    with pytest.raises(ValueError, match="another directory"):
        export_native(model, str(tmp_path / "link"))
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError, match="not empty"):
        export_native(model, str(tmp_path / "busy"))
    # A previous export is replaced, with its stale files
    export_native(model, str(tmp_path / "native"))
    (tmp_path / "native" / "chat_template.jinja").write_text("stale")
    export_native(model, str(tmp_path / "native"))
    assert not (tmp_path / "native" / "chat_template.jinja").exists()


def test_config_is_readable_by_older_transformers(experts, build, tmp_path):
    export_native(_with_routers(experts, build, tmp_path), str(tmp_path / "native"))
    config = json.load(open(tmp_path / "native" / "config.json"))
    assert config["num_experts"] == 3 and "rope_theta" in config
    assert config["router_aux_loss_coef"] == pytest.approx(0.01 / 2)
