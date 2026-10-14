# Older models and limitations

## Models built with earlier versions

Models merged with MixtureKit before version 1.1 carry modeling files generated for
transformers 4.52 and do not load with recent transformers versions. Upgrade them in place:
the weights are kept, traditional MoE weights are renamed to the current layout and missing
BTS stitch layers are added.

```python
from MixtureKit import upgrade_checkpoint

upgrade_checkpoint("models_merge/gemmax", base_model_class="Gemma3ForCausalLM")
```

`base_model_class` is the transformers causal LM class of the experts, e.g.
`Qwen3ForCausalLM` or `LlamaForCausalLM` (`Gemma3ForCausalLM` for Gemma 3, multimodal
included). The weights are checked against the upgraded model before any file is changed.

On a model already in the current layout (e.g. built from a development version), it only
replaces the modeling files by the ones of the installed MixtureKit.

## Known limitations

- BTS: `output_hidden_states` returns the hub and expert streams packed on the last
  dimension; with transformers 5 the outputs of the expert layers are listed too.
- BTS with Gemma 4: only the hub's per-layer embeddings are kept, and the expert streams use
  the hub's per-layer inputs, so an expert stream differs slightly from its original model.
- BTS needs `input_ids` (`inputs_embeds` alone are not supported).
- BTS with `device_map="auto"` across several GPUs: a hub layer and its expert layers
  (`model.expert_layers.N`) should be on the same device; pass an explicit `device_map`.
- trl 1.15 and later require Triton (Linux) for training.
