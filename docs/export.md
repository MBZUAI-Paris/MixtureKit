# Export to Mixtral, Qwen3-MoE and GGUF

A **traditional** MoE of Llama, Mistral or Qwen3 experts has the exact layout of a native
transformers MoE: one router per layer over the experts' feed-forward blocks, with the
softmax over the top-k experts. `mixturekit export` writes it as that native model:

| Experts | Native model |
| --- | --- |
| Llama, Mistral | `MixtralForCausalLM` |
| Qwen3 | `Qwen3MoeForCausalLM` |

```bash
mixturekit export models_merge/llama3moe llama3moe-mixtral
```

```python
from MixtureKit.export import export_native

export_native("models_merge/llama3moe", "llama3moe-mixtral")
```

The exported model computes the same outputs in transformers (4.52 to 5.x, whichever version
exported it), loads without `trust_remote_code` and works with the tools that support these
architectures, such as llama.cpp (GGUF), vLLM, SGLang and MLX.

Train the routers first (see {doc}`training`): a fresh merge does not store its routers (they
are random at every load), so it is refused. The output directory must be new, empty, or a
previous export, which is replaced.

## GGUF and llama.cpp

```bash
python llama.cpp/convert_hf_to_gguf.py llama3moe-mixtral --outfile llama3moe.gguf --outtype bf16
llama.cpp/build/bin/llama-quantize llama3moe.gguf llama3moe-Q4_K_M.gguf Q4_K_M
llama.cpp/build/bin/llama-cli -m llama3moe-Q4_K_M.gguf -p "Hello"
```

The weights are written tensor by tensor, like a merge.

## Limitations

- BTX models (one router per projection) and BTS models (every expert runs) have no native
  equivalent and cannot be exported; neither can experts of other families (e.g. Gemma 3).
- Llama experts with biases in their projections, and Qwen3 experts with
  `use_sliding_window`, are refused.
- llama.cpp itself ignores Mistral's sliding window and only supports SiLU Llama
  projections, as for the dense models; Qwen3 experts with `attention_bias` export for
  transformers but do not load in llama.cpp.
