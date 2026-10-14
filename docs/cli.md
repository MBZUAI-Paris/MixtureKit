# Command line

```text
mixturekit build CONFIG.yaml [--output DIR] [--dtype bfloat16|float16|float32] [--no-streaming]
                             [--push-to-hub ORG/NAME] [--private]
mixturekit train CONFIG.yaml
mixturekit visualize MODEL [--text TEXT] [--html FILE] [--device cuda|mps|cpu] [--port 8501]
mixturekit card MODEL [--repo-id ORG/NAME]
mixturekit push MODEL ORG/NAME [--private]
mixturekit export MODEL OUTPUT
```

`build`
: Merges the experts described in the YAML file (see {doc}`configuration`) and saves the
  configuration it was built from, with the datatype used, as `mixturekit_config.json`.
  `--push-to-hub` uploads the result.

`train`
: Trains the routers or stitch layers (see {doc}`training`). Needs `mixturekit[train]`.

`visualize`
: Opens the routing app, or with `--text` or `--html` prints the share of every expert and writes an
  HTML view (see {doc}`visualizer`). Needs `mixturekit[vis]` for the app.

`card`
: Writes the model card (`README.md`) of a merged model.

`push`
: Uploads a merged model with its card to the Hugging Face Hub. Training state
  (checkpoints, optimizer) and the local build configuration are not uploaded; the card
  holds the build configuration with the local paths removed. `--private` refuses to
  upload into an existing public repository.

`export`
: Writes a traditional MoE of Llama, Mistral or Qwen3 experts as a native Mixtral or
  Qwen3-MoE model (see {doc}`export`).
