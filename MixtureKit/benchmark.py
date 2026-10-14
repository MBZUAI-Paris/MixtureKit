"""
Benchmarks of merged models against their experts, with lm-evaluation-harness.

    mixturekit benchmark benchmark.yaml

The configuration lists the models (the experts and the merged models, Hub ids or
directories) and the lm-evaluation-harness tasks. The scores and the parameters (total and
active per token) are written as a Markdown table; the raw results are kept in a JSON file,
and the models already evaluated there are skipped when the command runs again.
"""

import json
import os

# The metric reported for a task: the first one found
METRICS = [
    "acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,flexible-extract",
    "exact_match,none", "f1,none", "pass@1,create_test",
]
# What changes the scores: a model evaluated with other settings is evaluated again
SCORE_SETTINGS = ("num_fewshot", "limit", "dtype", "apply_chat_template")

DEFAULTS = {
    "num_fewshot": None,
    "batch_size": "auto",
    "limit": None,
    "dtype": "bfloat16",
    "apply_chat_template": False,
    "output": "benchmark.md",
    "include_path": None,
    "device": None,  # cuda, mps or cpu; the first one available by default
}


def main_metric(task_results):
    """The name and value of the metric reported for a task, from its lm-eval results."""
    for metric in METRICS:
        if isinstance(task_results.get(metric), (int, float)):
            return metric.split(",")[0], task_results[metric]
    for metric, value in task_results.items():
        if isinstance(value, (int, float)) and "stderr" not in metric and metric != "alias":
            return metric.split(",")[0], value
    return None, None


def count_parameters(model_id):
    """
    The number of parameters of a model and the number used for every token: for BTX and
    traditional MoE, the routed experts count num_experts_per_tok / num_experts times.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    total = shared = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if ".experts." not in name:
            shared += param.numel()
    method = getattr(config, "moe_method", None)
    if method in ("btx", "traditional"):
        active = shared + (total - shared) * config.num_experts_per_tok / config.num_experts
    else:
        active = total  # a dense model, or BTS, which runs every expert
    return total, int(active)


def _evaluate(model_id, settings):
    import lm_eval

    model_args = f"pretrained={model_id},trust_remote_code=True,dtype={settings['dtype']}"
    task_manager = None
    if settings["include_path"]:  # a directory of your own task definitions
        from lm_eval.tasks import TaskManager

        task_manager = TaskManager(include_path=settings["include_path"])
    from .routing import default_device

    output = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        device=settings["device"] or default_device(),
        tasks=list(settings["tasks"]),
        num_fewshot=settings["num_fewshot"],
        batch_size=settings["batch_size"],
        limit=settings["limit"],
        apply_chat_template=settings["apply_chat_template"],
        task_manager=task_manager,
    )
    return output["results"]


def _billions(n):
    return f"{n / 1e9:.2f}B" if n >= 1e8 else f"{n / 1e6:.1f}M"


def table(settings, results):
    """The Markdown table of the scores, one row per model and one column per task."""
    tasks = list(settings["tasks"])
    header = ["Model", "Parameters", "Active"]
    for task in tasks:
        metric = next(
            (main_metric(r["tasks"].get(task, {}))[0] for r in results.values() if task in r["tasks"]), None
        )
        header.append(f"{task} ({metric})" if metric else task)
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for model in settings["models"]:
        result = results.get(model["name"])
        if result is None:
            continue
        row = [model["name"], _billions(result["parameters"]), _billions(result["active_parameters"])]
        for task in tasks:
            value = main_metric(result["tasks"].get(task, {}))[1]
            row.append("–" if value is None else f"{100 * value:.1f}")
        lines.append("| " + " | ".join(row) + " |")
    notes = [f"{k}: {settings[k]}" for k in ("num_fewshot", "limit", "apply_chat_template") if settings[k]]
    footer = "\nScores in %, lm-evaluation-harness" + (f" ({', '.join(notes)})" if notes else "") + ".\n"
    return "\n".join(lines) + "\n" + footer


def run(config, evaluate=_evaluate, parameters=count_parameters):
    """
    This function evaluates every model of the configuration that is not in the results
    file yet, and writes the results and the Markdown table. It returns the table.

    Parameters
    ----------
    config: dict
        `models` (a list of {name, model_id}), `tasks` (lm-evaluation-harness task names)
        and optionally num_fewshot, batch_size, limit, dtype, device, apply_chat_template,
        include_path (a directory of your own lm-evaluation-harness tasks) and output (the
        Markdown file; the results go next to it as .json).
    """
    settings = DEFAULTS | config
    names = [model["name"] for model in settings["models"]]
    if len(set(names)) != len(names):
        raise ValueError("The models need different names")
    output = settings["output"]
    results_file = os.path.splitext(output)[0] + ".json"
    results = {}
    if os.path.exists(results_file):
        with open(results_file, encoding="utf-8") as f:
            results = json.load(f)
    for model in settings["models"]:
        done = results.get(model["name"], {})
        score_settings = {key: settings[key] for key in SCORE_SETTINGS}
        if (
            done.get("model_id") == model["model_id"]
            and done.get("settings") == score_settings
            and set(settings["tasks"]) <= set(done.get("tasks", {}))
        ):
            print(f"{model['name']}: already evaluated")
            continue
        print(f"{model['name']}: evaluating {model['model_id']}")
        total, active = parameters(model["model_id"])
        results[model["name"]] = {
            "model_id": model["model_id"],
            "parameters": total,
            "active_parameters": active,
            "settings": score_settings,
            "tasks": evaluate(model["model_id"], settings),
        }
        # Saved after every model: a long run can be resumed
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    markdown = table(settings, results)
    with open(output, "w", encoding="utf-8") as f:
        f.write(markdown)
    return markdown
