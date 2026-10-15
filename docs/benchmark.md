# Benchmarks

`mixturekit benchmark` scores merged models next to their experts with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), the evaluation
library behind the Open LLM Leaderboard, and writes a Markdown table with the parameters of
every model (total, and used per token).

```bash
pip install "mixturekit[bench]"
mixturekit benchmark examples/benchmark.yaml
```

```yaml
models:
  - {name: Qwen3-0.6B-Base, model_id: Qwen/Qwen3-0.6B-Base}
  - {name: Qwen3-0.6B, model_id: Qwen/Qwen3-0.6B}
  - {name: MoE 2x0.6B (trained routers), model_id: qwen3moe2x-trained}
tasks: [arc_challenge, hellaswag, gsm8k, mmlu]
```

| Key | Meaning |
| --- | --- |
| `models` | `{name, model_id}`: Hub ids or directories, the experts and the merged models |
| `tasks` | lm-evaluation-harness task names (`lm_eval ls tasks` lists them) |
| `num_fewshot` | The number of examples in the prompt, the default of every task if unset |
| `batch_size` | `auto` by default |
| `limit` | Evaluate on the first examples only, e.g. `100` for a quick check |
| `dtype` | `bfloat16` by default |
| `device` | `cuda`, `mps` or `cpu`, the first one available by default |
| `apply_chat_template` | Format the prompts with the chat template of each model |
| `include_path` | A directory of your own task definitions (YAML) |
| `output` | The Markdown table, `benchmark.md` by default |

The raw results are saved next to the table (`benchmark.json`) after every model. Running
the command again skips the models already evaluated with the same settings, so a long run
can be resumed, and new models or tasks added.

The table has one row per model: its name, its parameters (total and active) and its score
on every task, in %, with the metric used in the column header (e.g. `arc_challenge (acc_norm)`).

For BTX and traditional MoE, the active parameters count the routed experts
`num_experts_per_tok / num_experts` times; BTS runs every expert.

```{tip}
Compare the merged model with the best expert on each task: a good merge keeps the strengths
of every expert. Train the routers on a mix of the experts' domains.
```
