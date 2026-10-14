"""The mixturekit command: build, train, card, push."""

import json
import os

import pytest
import yaml

from MixtureKit.cli import main as cli
from conftest import PROMPT, load, moe_config


def build_with_cli(tmp_path, experts, method="btx", **extra):
    config = moe_config(method, f"cli{method}", experts, **extra)
    config_file = tmp_path / f"{method}.yaml"
    config_file.write_text(yaml.safe_dump(config))
    output = tmp_path / f"out_{method}"
    assert cli(["build", str(config_file), "--output", str(output), "--dtype", "float32"]) == 0
    return output, config


@pytest.mark.parametrize("method", ["btx", "traditional", "bts"])
def test_build_from_yaml(experts, tmp_path, method):
    output, config = build_with_cli(tmp_path, experts["llama"], method)
    model = load(str(output)).eval()
    assert model(input_ids=PROMPT).logits.isfinite().all()
    saved = json.load(open(output / "mixturekit_config.json"))
    assert saved == config | {"dtype": "float32"}  # as given, with the datatype used
    merged = json.load(open(output / "config.json"))
    assert merged["expert_model_ids"] == experts["llama"]


def test_build_options_in_the_file(experts, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = moe_config("btx", "inline", experts["qwen3"]) | {"dtype": "float32", "streaming": False}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(config))
    assert cli(["build", "c.yaml"]) == 0
    assert (tmp_path / "models_merge" / "inline" / "config.json").exists()


def test_build_rejects_a_file_without_experts(tmp_path):
    (tmp_path / "bad.yaml").write_text("model_type: x\n")
    with pytest.raises(SystemExit):
        cli(["build", str(tmp_path / "bad.yaml")])


def test_model_card(experts, tmp_path):
    output, _ = build_with_cli(tmp_path, experts["llama"])
    assert cli(["card", str(output), "--repo-id", "my-org/llama-moe"]) == 0
    card = (output / "README.md").read_text(encoding="utf-8")
    front = yaml.safe_load(card.split("---")[1])
    assert "mixturekit" in front["tags"] and "btx" in front["tags"]
    assert "base_model" not in front  # local experts are not Hub models
    assert "my-org/llama-moe" in card and "trust_remote_code=True" in card
    assert "not trained yet" in card  # a fresh build has untrained routers
    assert "mixturekit build config.yaml" in card and "@article{chamma2025mixturekit" in card

    # Saving random routers does not make them trained; training marks the model
    model = load(str(output))
    model.save_pretrained(output)
    assert cli(["card", str(output)]) == 0
    assert "not trained yet" in (output / "README.md").read_text(encoding="utf-8")
    from MixtureKit.training import mark_trained

    mark_trained(str(output))
    assert cli(["card", str(output)]) == 0
    assert "routers are trained" in (output / "README.md").read_text(encoding="utf-8")


def test_model_card_lists_hub_experts(experts, tmp_path):
    from MixtureKit.hub import model_card

    output, _ = build_with_cli(tmp_path, experts["llama"])
    config = json.load(open(output / "config.json"))
    config["expert_model_ids"] = ["org/base", "org/math", "org/code"]
    config["expert_is_local"] = [False, False, False]
    json.dump(config, open(output / "config.json", "w"))
    front = yaml.safe_load(model_card(str(output)).split("---")[1])
    assert front["base_model"] == ["org/base", "org/code", "org/math"]
    assert front["base_model_relation"] == "merge"


def test_push_uploads_the_model_and_its_card(experts, tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    class FakeApi:
        def __init__(self, token=None):
            calls.append(("init", token))

        def create_repo(self, repo_id, private=False, exist_ok=False):
            calls.append(("create_repo", repo_id, private, exist_ok))

        def upload_folder(self, folder_path, repo_id, commit_message, ignore_patterns):
            calls.append(("upload_folder", folder_path, repo_id, sorted(os.listdir(folder_path))))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    output, _ = build_with_cli(tmp_path, experts["llama"])
    assert cli(["push", str(output), "my-org/llama-moe", "--private"]) == 0
    assert ("create_repo", "my-org/llama-moe", True, True) in calls
    uploaded = next(c for c in calls if c[0] == "upload_folder")[3]
    assert "README.md" in uploaded and "config.json" in uploaded


def test_train_command(experts, tmp_path, monkeypatch):
    pytest.importorskip("trl")
    datasets = pytest.importorskip("datasets")
    output, _ = build_with_cli(tmp_path, experts["qwen3"])
    rows = [{"messages": [{"role": "user", "content": "what is one plus two"},
                          {"role": "assistant", "content": "the answer is three"}]}] * 8
    datasets.Dataset.from_list(rows).save_to_disk(str(tmp_path / "data"))
    trained = tmp_path / "trained"
    config = {
        "model_id": str(output), "dataset_path": str(tmp_path / "data"), "output_dir": str(trained),
        "chat_template": "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}",
        "max_steps": 2, "per_device_train_batch_size": 2, "learning_rate": 1e-2,
        "use_cpu": True, "report_to": "none", "save_strategy": "no", "max_length": 32,
    }
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(config))
    assert cli(["train", str(tmp_path / "train.yaml")]) == 0
    model, info = load(str(trained), output_loading_info=True)
    assert not info["missing_keys"]  # the trained routers are saved
    assert json.load(open(trained / "config.json"))["mixturekit_trained"] is True
    assert (trained / "mixturekit_config.json").exists()  # the card keeps its rebuild section
    # trl writes its own README: push and card replace it with MixtureKit's
    (trained / "README.md").write_text("---\ntags: [trl]\n---\ngeneric card")
    from MixtureKit.hub import write_model_card

    write_model_card(str(trained))
    assert "mixturekit" in (trained / "README.md").read_text(encoding="utf-8")


def test_card_never_shows_local_paths(experts, tmp_path):
    output, _ = build_with_cli(tmp_path, experts["llama"])
    card = open(__import__("MixtureKit.hub", fromlist=["x"]).write_model_card(str(output), overwrite=True)).read()
    parent = os.path.dirname(experts["llama"][0])
    assert parent not in card and "<local: llama_0>" in card and "local model `llama_0`" in card


def test_relative_local_experts_are_not_hub_models(experts, tmp_path, monkeypatch):
    import shutil

    from MixtureKit.hub import model_card

    work = tmp_path / "work"
    (work / "ex").mkdir(parents=True)
    for i in range(2):
        shutil.copytree(experts["llama"][i], work / "ex" / f"m{i}")
    monkeypatch.chdir(work)
    config = moe_config("btx", "rel", ["ex/m0", "ex/m1"])
    (work / "c.yaml").write_text(yaml.safe_dump(config))
    assert cli(["build", "c.yaml", "--output", "out", "--dtype", "float32"]) == 0
    monkeypatch.chdir(tmp_path)  # written from elsewhere, "ex/m0" looks like a Hub id
    front = yaml.safe_load(model_card(str(work / "out")).split("---")[1])
    assert "base_model" not in front


def test_command_line_options_win_and_are_saved(experts, tmp_path):
    config = moe_config("btx", "opts", experts["llama"]) | {"dtype": "float16", "output_dir": str(tmp_path / "fromfile")}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(config))
    assert cli(["build", str(tmp_path / "c.yaml"), "--dtype", "float32", "--output", str(tmp_path / "out")]) == 0
    saved = json.load(open(tmp_path / "out" / "mixturekit_config.json"))
    assert saved["dtype"] == "float32" and "output_dir" not in saved
    assert not (tmp_path / "fromfile").exists()


@pytest.mark.parametrize("content", ["model_type: x\n", "moe_method: btx\nmodel_type: x\nexperts: []\nstreaming: 'false'\n"])
def test_invalid_build_files_exit_with_a_message(tmp_path, content):
    (tmp_path / "bad.yaml").write_text(content)
    with pytest.raises(SystemExit):
        cli(["build", str(tmp_path / "bad.yaml")])
    with pytest.raises(SystemExit):
        cli(["build", str(tmp_path / "missing.yaml")])


def test_card_on_lora_adapters_explains_what_to_do(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(SystemExit) as error:
        cli(["card", str(tmp_path)])
    assert "LoRA" in str(error.value)


def test_push_keeps_training_state_and_paths_out_and_respects_private(experts, tmp_path, monkeypatch):
    import huggingface_hub

    uploads = []

    class Info:
        private = False

    class FakeApi:
        def __init__(self, token=None):
            pass

        def repo_info(self, repo_id):
            if repo_id == "me/public":
                return Info()
            raise RuntimeError("not found")

        def create_repo(self, repo_id, private=False, exist_ok=False):
            pass

        def upload_folder(self, folder_path, repo_id, commit_message, ignore_patterns):
            uploads.append(ignore_patterns)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    output, _ = build_with_cli(tmp_path, experts["llama"])
    assert cli(["push", str(output), "me/new", "--private"]) == 0
    assert {"training_args.bin", "mixturekit_config.json", "optimizer.pt"} <= set(uploads[0])
    with pytest.raises(SystemExit):
        cli(["push", str(output), "me/public", "--private"])


def test_benchmark_command_checks_its_file(tmp_path):
    (tmp_path / "b.yaml").write_text("models: []\n")
    with pytest.raises(SystemExit):
        cli(["benchmark", str(tmp_path / "b.yaml")])
