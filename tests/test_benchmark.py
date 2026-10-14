"""mixturekit benchmark: parameters, resumable results and the Markdown table."""

import json

import pytest

from MixtureKit.benchmark import count_parameters, main_metric, run


def test_active_parameters(experts, build):
    dense, dense_active = count_parameters(experts["qwen3"][0])
    assert dense == dense_active
    total, active = count_parameters(build("traditional", experts["qwen3"], "qwen3traditional"))
    # 3 copies of the MLPs and the routers (4 layers, hidden size 32, 3 experts):
    # 2 of the 3 MLP copies are used per token
    routers = 4 * 32 * 3
    mlps = (total - routers - dense) / 2
    assert active == total - mlps
    bts_total, bts_active = count_parameters(build("bts", experts["qwen3"], "qwen3bts"))
    assert bts_total == bts_active


def test_main_metric():
    assert main_metric({"acc,none": 0.5, "acc_norm,none": 0.6, "acc_stderr,none": 0.1}) == ("acc_norm", 0.6)
    assert main_metric({"exact_match,strict-match": 0.3, "alias": "gsm8k"}) == ("exact_match", 0.3)
    assert main_metric({"word_perplexity,none": 12.0}) == ("word_perplexity", 12.0)


def test_run_resumes_and_writes_the_table(tmp_path):
    calls = []

    def evaluate(model_id, settings):
        calls.append(model_id)
        return {"arc": {"acc_norm,none": 0.5 if model_id == "a" else 0.75, "acc_norm_stderr,none": 0.01}}

    config = {
        "models": [{"name": "expert", "model_id": "a"}, {"name": "MoE", "model_id": "b"}],
        "tasks": ["arc"],
        "output": str(tmp_path / "bench.md"),
    }
    sizes = lambda model_id: (3_000_000_000, 2_000_000_000)
    markdown = run(config, evaluate=evaluate, parameters=sizes)
    assert "| MoE | 3.00B | 2.00B | 75.0 |" in markdown and "arc (acc_norm)" in markdown
    assert json.load(open(tmp_path / "bench.json"))["expert"]["tasks"]["arc"]["acc_norm,none"] == 0.5
    run(config, evaluate=evaluate, parameters=sizes)
    assert calls == ["a", "b"]  # already evaluated
    run(config | {"limit": 10}, evaluate=evaluate, parameters=sizes)
    assert calls == ["a", "b", "a", "b"]  # other settings: evaluated again
    with pytest.raises(ValueError):
        run(config | {"models": [{"name": "x", "model_id": "a"}] * 2}, evaluate=evaluate, parameters=sizes)


def test_lm_eval_on_a_local_task(experts, build, tmp_path, monkeypatch):
    pytest.importorskip("lm_eval")
    rows = [
        {"question": "the quick brown", "choices": ["fox", "dog"], "answer": 0},
        {"question": "hello", "choices": ["world", "plus"], "answer": 0},
        {"question": "one two", "choices": ["three", "lazy"], "answer": 0},
    ]
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "tiny.yaml").write_text(
        "task: tiny_choice\ndataset_path: json\n"
        f"dataset_kwargs:\n  data_files:\n    test: {data}\n"
        "test_split: test\noutput_type: multiple_choice\n"
        "doc_to_text: '{{question}}'\ndoc_to_choice: '{{choices}}'\ndoc_to_target: '{{answer}}'\n"
        "metric_list:\n  - metric: acc\n"
    )
    monkeypatch.chdir(tmp_path)
    markdown = run({
        "models": [
            {"name": "expert", "model_id": experts["llama"][0]},
            {"name": "MoE", "model_id": build("traditional", experts["llama"], "llamatraditional")},
        ],
        "tasks": ["tiny_choice"], "include_path": str(tasks), "dtype": "float32", "batch_size": 1,
    })
    assert "tiny_choice (acc)" in markdown and markdown.count("\n| ") == 3
