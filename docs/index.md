# MixtureKit

**Turn the fine-tuned models you already have into one Mixture-of-Experts model.**

MixtureKit merges Hugging Face checkpoints that share an architecture (e.g. a base model and its
math, code or language fine-tunes) into a single MoE model, trains the routers that pick the
experts for every token, and shows which expert every token uses.

```bash
pip install "mixturekit[vis]"
mixturekit build examples/configs/traditional_qwen3.yaml   # merge the experts -> models_merge/qwen3moe2x
mixturekit train my_training.yaml                          # train the routers -> e.g. qwen3moe2x-trained
mixturekit visualize qwen3moe2x-trained                    # see which expert every token uses
```

- **Three ways to compose experts**: Branch-Train-MiX (BTX), Mixtral-style MoE and
  Branch-Train-Stitch (BTS). See {doc}`methods`.
- **Low memory merges**: the experts are read and the merged model written tensor by tensor.
  Merging runs on a laptop; training the routers wants a GPU.
- **Plain transformers models**: a merged model loads with
  `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`, MixtureKit not needed,
  tested on transformers 4.52 to 5.19.
- **Router training** with a load balancing loss, optionally with LoRA on the experts.
- **Routing visualizer**: an app, a shareable HTML page, or inline in notebooks.
- **Hub ready**: `mixturekit push` uploads the model with a generated model card.
- **Native export**: a traditional MoE of Llama, Mistral or Qwen3 experts becomes a Mixtral
  or Qwen3-MoE model, ready for GGUF and llama.cpp, vLLM or MLX. See {doc}`export`.

[Try it in Colab](https://colab.research.google.com/github/MBZUAI-Paris/MixtureKit/blob/main/examples/build_your_own_moe.ipynb):
merge two Qwen3-0.6B models, train the routers and look at the routing in about 15 minutes on a free GPU.
The [tutorial](https://huggingface.co/blog/AC-723/build-your-own-mixture-of-experts-from-two-fine-tu) explains each step.

```{toctree}
:maxdepth: 2
:caption: Guide

quickstart
methods
configuration
cli
training
visualizer
export
benchmark
upgrade
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

## Citation

MixtureKit was published at ACL 2026 (System Demonstrations): [paper](https://aclanthology.org/2026.acl-demo.15/), also on [arXiv](https://arxiv.org/abs/2512.12121).

```bibtex
@inproceedings{chamma-etal-2026-mixturekit,
  title = "{M}ixture{K}it: A General Framework for Composing, Training, and Visualizing Mixture-of-Experts Models",
  author = "Chamma, Ahmad and El Herraoui, Omar and Shang, Guokan",
  booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 3: System Demonstrations)",
  month = jul,
  year = "2026",
  address = "San Diego, California, United States",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2026.acl-demo.15/",
  doi = "10.18653/v1/2026.acl-demo.15",
  pages = "148--156"
}
```
