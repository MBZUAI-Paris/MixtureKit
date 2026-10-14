import glob
import json
import os
import re

import pytest
import torch
from safetensors.torch import load_file

from MixtureKit import build_moe
from conftest import FAMILIES, PROMPT, load, moe_config

METHODS = ["btx", "traditional", "bts"]


def read_weights(path):
    weights = {}
    for file in glob.glob(os.path.join(path, "*.safetensors")):
        weights.update(load_file(file))
    return weights


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_forward_and_cached_generation(experts, build, family, method):
    model = load(build(method, experts[family], f"{family}{method}")).eval()
    with torch.no_grad():
        logits = model(input_ids=PROMPT).logits
    assert torch.isfinite(logits).all()

    cached = model.generate(PROMPT, max_new_tokens=6, do_sample=False, use_cache=True)
    uncached = model.generate(PROMPT, max_new_tokens=6, do_sample=False, use_cache=False)
    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_trainable_weights_receive_gradients(experts, build, family, method):
    model = load(build(method, experts[family], f"{family}{method}")).train()
    if method == "bts":
        trainable = [p for n, p in model.named_parameters() if ".stitches.w_proj." in n]
    else:
        trainable = [p for n, p in model.named_parameters() if n.endswith(".gate.weight")]
    assert trainable
    model(input_ids=PROMPT, labels=PROMPT).loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", ["btx", "traditional"])
def test_identical_experts_reproduce_the_base_model(experts, build, family, method):
    same = [experts[family][0]] * 3
    model = load(build(method, same, f"{family}{method}same")).eval()
    base = load(experts[family][0]).eval()
    with torch.no_grad():
        diff = (model(input_ids=PROMPT).logits - base(input_ids=PROMPT).logits).abs().max()
    assert diff < 1e-5


@pytest.mark.parametrize("reentrant", [None, True, False])
@pytest.mark.parametrize("method", ["btx", "traditional"])
def test_load_balancing_loss_trains_routers(experts, build, method, reentrant):
    model = load(build(method, experts["qwen3"], f"qwen3{method}"))
    for name, param in model.named_parameters():
        param.requires_grad_(name.endswith(".gate.weight"))
    if reentrant is not None:
        model.gradient_checkpointing_enable({"use_reentrant": reentrant})
        model.enable_input_require_grads()
    model.train()

    grads = []
    for alpha in (0.0, 1.0):
        model.set_load_balancing_alpha(alpha)
        model.zero_grad()
        model(input_ids=PROMPT, labels=PROMPT).loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad]))
    assert not torch.allclose(grads[0], grads[1])

    # The gradient is the one of loss + aux_loss
    model.set_load_balancing_alpha(1.0)
    outputs = model(input_ids=PROMPT, labels=PROMPT)
    # Averaged over the routers: 1 for a perfectly balanced load, at most num_experts / top_k
    assert 0.9 < outputs.aux_loss.item() <= 1.5
    model.eval()
    assert getattr(model(input_ids=PROMPT), "aux_loss", None) is None


def test_load_balancing_gradient_matches_explicit_loss(experts, build):
    model = load(build("btx", experts["qwen3"], "qwen3btx")).train()
    model.set_load_balancing_alpha(1.0)
    routers = [p for n, p in model.named_parameters() if n.endswith(".gate.weight")]

    model.zero_grad()
    model(input_ids=PROMPT, labels=PROMPT).loss.backward()
    attached = [p.grad.clone() for p in routers]

    # Same gradient with the loss written explicitly from the router outputs
    from importlib import import_module

    modeling = import_module(type(model).__module__)
    model.zero_grad()
    captured = []
    hooks = [
        m.gate.register_forward_hook(lambda mod, inp, out: captured.append(out))
        for m in model._routed_modules
    ]
    model.set_load_balancing_alpha(0.0)
    ce = model(input_ids=PROMPT, labels=PROMPT).loss
    for h in hooks:
        h.remove()
    k = model.config.num_experts_per_tok
    aux = sum(
        modeling.switch_load_balancing_loss(logits, logits.topk(k, dim=-1).indices, model.config.num_experts)
        for logits in captured
    ) / len(captured)
    (ce + aux).backward()
    for a, b in zip(attached, routers):
        assert torch.allclose(a, b.grad, atol=1e-6)


@pytest.mark.parametrize("family", ["qwen3", "gemma3"])
def test_bts_beam_search_matches_uncached(experts, build, family):
    model = load(build("bts", experts[family], f"{family}bts")).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".stitches." in name:
                param.normal_(0, 0.5)
    kwargs = dict(max_new_tokens=6, do_sample=False, num_beams=3, pad_token_id=0)
    assert torch.equal(
        model.generate(PROMPT, use_cache=True, **kwargs),
        model.generate(PROMPT, use_cache=False, **kwargs),
    )


@pytest.mark.parametrize("method", METHODS)
def test_streaming_matches_full_loading(experts, build, method):
    streamed = read_weights(build(method, experts["llama"], f"llama{method}", streaming=True))
    loaded = read_weights(build(method, experts["llama"], f"llama{method}", streaming=False))
    assert streamed.keys() == loaded.keys()
    assert all(torch.equal(streamed[k], loaded[k]) for k in streamed)


def test_fresh_bts_build_has_all_stitch_weights(experts, build):
    path = build("bts", experts["gemma3"], "gemma3bts")
    _, info = load(path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    weights = read_weights(path)
    stitches = [k for k in weights if ".stitches." in k]
    assert stitches
    for key in stitches:
        expected = torch.eye(32) if ".w_proj." in key else torch.zeros(32, 32)
        assert torch.equal(weights[key], expected)


def test_merged_config(experts, build):
    config = json.load(open(os.path.join(build("btx", experts["llama"], "llamabtx"), "config.json")))
    assert config["architectures"] == ["MixtureKitForCausalLM"]
    assert config["base_model_class"] == "LlamaForCausalLM"
    assert config["num_experts"] == 3
    assert config["router_layers_index"] == [0, 1, 2, 3]


def test_rebuild_replaces_previous_shards(experts, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build_moe(config=moe_config("btx", "rebuilt", experts["qwen3"][:2]), torch_dtype=torch.float32)
    from MixtureKit.MoeExperts import MoeExperts

    merger = MoeExperts(
        moe_config("btx", "rebuilt", experts["qwen3"]), torch_dtype=torch.float32, max_shard_size="20KB"
    )
    merger.compose()
    merger.save_checkpoint("models_merge/rebuilt")
    files = sorted(os.listdir("models_merge/rebuilt"))
    assert "model.safetensors" not in files
    assert "model.safetensors.index.json" in files

    model = load("models_merge/rebuilt")
    expert = load(experts["qwen3"][2])
    assert torch.equal(
        model.model.layers[0].mlp.up_proj.experts.expert_2.weight,
        expert.model.layers[0].mlp.up_proj.weight,
    )


def test_experts_with_different_architectures_raise(experts, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mixed = [experts["llama"][0], experts["qwen3"][1]]
    with pytest.raises(ValueError):
        build_moe(config=moe_config("btx", "mixed", mixed), torch_dtype=torch.float32)


def test_checkpoint_layout(experts, build):
    btx = read_weights(build("btx", experts["llama"], "llamabtx"))
    assert "model.layers.0.mlp.up_proj.gate.weight" not in btx  # routers start untrained
    assert "model.layers.0.mlp.up_proj.experts.expert_2.weight" in btx
    traditional = read_weights(build("traditional", experts["llama"], "llamatraditional"))
    assert "model.layers.0.mlp.experts.2.up_proj.weight" in traditional
    bts = read_weights(build("bts", experts["llama"], "llamabts"))
    assert "model.expert_layers.3.1.mlp.up_proj.weight" in bts
    assert "model.expert_embed_tokens.1.weight" in bts


def test_modeling_files_are_small_and_standalone(experts, build):
    path = build("btx", experts["llama"], "llamabtx")
    modeling = open(os.path.join(path, "modeling_llamabtx.py")).read()
    # Loadable without MixtureKit installed
    assert not re.search(r"^\s*(import|from)\s+MixtureKit\b", modeling, flags=re.MULTILINE)
    assert "class MixtureKitForCausalLM(_Base)" in modeling


@pytest.mark.parametrize("method", METHODS)
def test_multimodal_experts_merge_their_language_model(experts, build, method):
    path = build(method, experts["gemma3mm"], f"gemma3mm{method}")
    weights = read_weights(path)
    assert not any("vision" in k or "multi_modal" in k for k in weights)
    model = load(path).eval()
    cached = model.generate(PROMPT, max_new_tokens=4, do_sample=False, use_cache=True)
    assert torch.equal(cached, model.generate(PROMPT, max_new_tokens=4, do_sample=False, use_cache=False))

    # The averaged attention is the mean of the three language models
    if method != "bts":
        # Read from the files: the layout of the language model depends on the version
        stored = [read_weights(p) for p in experts["gemma3mm"]]
        source = next(k for k in stored[0] if k.endswith("layers.1.self_attn.o_proj.weight"))
        key = "model.layers.1.self_attn.o_proj.weight"
        mean = sum(w[source] for w in stored) / 3
        assert torch.allclose(weights[key], mean, atol=1e-6)


@pytest.mark.parametrize("method", ["btx", "traditional"])
def test_load_balancing_gradient_follows_loss_scaling(experts, build, method):
    """Gradient accumulation and fp16 loss scaling multiply the whole loss."""
    model = load(build(method, experts["qwen3"], f"qwen3{method}")).train()
    routers = [p for n, p in model.named_parameters() if n.endswith(".gate.weight")]

    def router_grads(alpha, scale):
        model.set_load_balancing_alpha(alpha)
        model.zero_grad()
        (model(input_ids=PROMPT, labels=PROMPT).loss * scale).backward()
        return torch.cat([p.grad.flatten() for p in routers])

    for scale in (0.0, 0.5, 4.0):
        aux_part = router_grads(1.0, scale) - router_grads(0.0, scale)
        reference = router_grads(1.0, 1.0) - router_grads(0.0, 1.0)
        assert torch.allclose(aux_part, scale * reference, atol=1e-6)


def test_return_dict_false_in_training(experts, build):
    model = load(build("btx", experts["qwen3"], "qwen3btx")).train()
    model.set_load_balancing_alpha(1.0)
    loss = model(input_ids=PROMPT, labels=PROMPT, return_dict=False)[0]
    reference = model(input_ids=PROMPT, labels=PROMPT)
    assert torch.allclose(loss, reference.loss, atol=1e-5)


def test_bts_deepcopy_uses_its_own_experts(experts, build):
    import copy

    model = load(build("bts", experts["qwen3"], "qwen3bts")).eval()
    twin = copy.deepcopy(model)
    with torch.no_grad():
        before = twin(input_ids=PROMPT).logits
        for embed in model.model.expert_embed_tokens:
            embed.weight.zero_()
        assert torch.equal(twin(input_ids=PROMPT).logits, before)
        for embed in twin.model.expert_embed_tokens:
            embed.weight.zero_()
        assert not torch.equal(twin(input_ids=PROMPT).logits, before)


@pytest.mark.parametrize("method", METHODS)
def test_save_pretrained_roundtrip(experts, build, method, tmp_path):
    model = load(build(method, experts["llama"], f"llama{method}")).eval()
    model.save_pretrained(tmp_path)
    reloaded = load(str(tmp_path)).eval()
    with torch.no_grad():
        assert torch.equal(model(input_ids=PROMPT).logits, reloaded(input_ids=PROMPT).logits)


def test_bts_layers_are_not_split_across_devices(experts, build):
    model = load(build("bts", experts["llama"], "llamabts"))
    assert any(name.startswith("Bts") for name in model._no_split_modules)


def test_generation_config_has_no_null_values(experts, build):
    path = build("btx", experts["gemma3"], "gemma3btx")
    generation = json.load(open(os.path.join(path, "generation_config.json")))
    assert None not in generation.values()


def test_legacy_rope_fields():
    from MixtureKit.MoeExperts import _add_legacy_rope_fields

    config = {"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}}
    _add_legacy_rope_fields(config)
    assert config["rope_theta"] == 1e6 and "rope_scaling" not in config

    gemma = {
        "rope_parameters": {
            "full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1e6},
            "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
        }
    }
    _add_legacy_rope_fields(gemma)
    assert gemma["rope_theta"] == 1e6 and gemma["rope_local_base_freq"] == 1e4
    assert gemma["rope_scaling"] == {"rope_type": "linear", "factor": 8.0}


def test_upgrade_traditional_checkpoint_from_previous_layout(experts, build, tmp_path):
    """Previous versions stored the router on the layer and the experts per projection."""
    import shutil

    from MixtureKit import upgrade_checkpoint
    from safetensors.torch import save_file

    current = build("traditional", experts["llama"], "llamatraditional")
    old = tmp_path / "oldtraditional"
    shutil.copytree(current, old)
    for name in os.listdir(old):
        if name.endswith(".py"):
            os.remove(old / name)
    weights = read_weights(str(old))
    previous = {}
    for key, value in weights.items():
        match = re.match(r"(.*\.layers\.\d+)\.mlp\.experts\.(\d+)\.([^.]+)\.(weight|bias)$", key)
        previous[f"{match[1]}.mlp.{match[3]}.experts.expert_{match[2]}.{match[4]}" if match else key] = value
    for layer in range(4):
        previous[f"model.layers.{layer}.gate.weight"] = torch.randn(3, 32)
    save_file(previous, str(old / "model.safetensors"), metadata={"format": "pt"})
    config = json.load(open(old / "config.json"))
    for key in ("auto_map", "base_model_class", "architectures"):
        config.pop(key)
    json.dump(config, open(old / "config.json", "w"))

    upgrade_checkpoint(str(old), "LlamaForCausalLM")
    _, info = load(str(old), output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    upgraded = read_weights(str(old))
    assert torch.equal(upgraded["model.layers.2.mlp.gate.weight"], previous["model.layers.2.gate.weight"])


def test_upgrade_rejects_a_wrong_base_class(experts, build, tmp_path):
    import shutil

    from MixtureKit import upgrade_checkpoint

    old = tmp_path / "wrongbase"
    shutil.copytree(build("btx", experts["llama"], "llamabtx"), old)
    config = json.load(open(old / "config.json"))
    for key in ("auto_map", "base_model_class", "architectures"):
        config.pop(key)
    json.dump(config, open(old / "config.json", "w"))
    before = sorted(os.listdir(old))
    with pytest.raises(ValueError):
        upgrade_checkpoint(str(old), "Gemma3ForCausalLM")
    # Nothing was changed
    assert sorted(os.listdir(old)) == before


def test_bts_rejects_experts_with_different_vocabularies(experts, tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    path = tmp_path / "bigger_vocab"
    model = AutoModelForCausalLM.from_pretrained(experts["llama"][1])
    model.resize_token_embeddings(model.config.vocab_size + 8)
    model.save_pretrained(path)
    for name in os.listdir(experts["llama"][1]):
        if name.startswith(("tokenizer", "special_tokens")):
            import shutil

            shutil.copy(os.path.join(experts["llama"][1], name), path)
    monkeypatch.chdir(tmp_path)
    for streaming in (True, False):
        with pytest.raises(ValueError, match="vocabulary"):
            build_moe(
                config=moe_config("bts", "mixedvocab", [experts["llama"][0], str(path)]),
                torch_dtype=torch.float32,
                streaming=streaming,
            )


def _without_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    return model


@pytest.mark.parametrize("family", ["llama", "gemma3"])
def test_bts_two_forwards_before_backward_with_checkpointing(experts, build, family):
    """Each forward pass keeps its own expert embeddings when the layers are recomputed."""
    other = PROMPT.flip(-1)
    grads = []
    for checkpointing in (False, True):
        model = _without_dropout(load(build("bts", experts[family], f"{family}bts")))
        if checkpointing:
            model.gradient_checkpointing_enable()
        model.train()
        loss = model(input_ids=PROMPT, labels=PROMPT).loss + model(input_ids=other, labels=other).loss
        loss.backward()
        grads.append({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        assert torch.allclose(grads[0][name], grads[1][name], atol=1e-6), name


def test_bts_needs_input_ids(experts, build):
    model = load(build("bts", experts["llama"], "llamabts")).eval()
    with torch.no_grad():
        embeds = model.get_input_embeddings()(PROMPT)
        model(input_ids=PROMPT)  # the embeddings of another forward pass are not reused
        with pytest.raises(ValueError, match="input_ids"):
            model(inputs_embeds=embeds)


@pytest.mark.parametrize("family", ["llama", "gemma3"])
def test_bts_with_embedding_noise(experts, build, family):
    """Hooks changing the hub embedding (NEFTune, PEFT on the embeddings) keep the expert streams."""
    model = _without_dropout(load(build("bts", experts[family], f"{family}bts")))
    reference = model(input_ids=PROMPT, labels=PROMPT).loss
    handle = model.get_input_embeddings().register_forward_hook(lambda module, args, output: output + 0.0)
    model.gradient_checkpointing_enable()
    model.train()
    loss = model(input_ids=PROMPT, labels=PROMPT).loss
    loss.backward()
    handle.remove()
    assert torch.allclose(loss, reference, atol=1e-6)


def test_bts_streams_keep_their_shared_keys_and_values(experts, tmp_path, monkeypatch):
    """Gemma 4 layers reusing the keys and values of earlier layers read those of their stream."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4ForCausalLM"):
        pytest.skip("Gemma 4 needs a recent transformers version")
    from conftest import family_config, make_tokenizer

    tokenizer = make_tokenizer()
    config = family_config(
        "gemma4", len(tokenizer),
        layer_types=["sliding_attention", "full_attention"] * 2, num_kv_shared_layers=2,
    )
    paths = []
    for idx in range(2):
        torch.manual_seed(idx)
        path = tmp_path / f"kvshared_{idx}"
        transformers.AutoModelForCausalLM.from_config(config).save_pretrained(path)
        tokenizer.save_pretrained(path)
        paths.append(str(path))
    monkeypatch.chdir(tmp_path)
    # A single stitch layer, on the last block: the hub is the first expert before it
    build_moe(moe_config("bts", "kvshared", paths, stitch_freq=5), torch_dtype=torch.float32)
    model = load("models_merge/kvshared").eval()
    hub = load(paths[0]).eval()
    outputs = {}
    for name, m in (("moe", model), ("hub", hub)):
        m.model.layers[2].register_forward_hook(
            lambda module, args, output, name=name: outputs.__setitem__(name, output[0] if isinstance(output, tuple) else output)
        )
    with torch.no_grad():
        output = model(input_ids=PROMPT, return_shared_kv_states=True)
        hub(input_ids=PROMPT)
    # The store returned keeps the layout of transformers (layer type -> keys and values)
    assert set(output.shared_kv_states) <= set(config.layer_types)
    hidden_size = config.hidden_size
    assert (outputs["moe"][..., :hidden_size] - outputs["hub"]).abs().max() < 1e-5


def test_btx_without_any_routed_layer_raises(experts, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = moe_config("btx", "norouter", experts["llama"], router_layers=["mlp.missing_proj"])
    with pytest.raises(ValueError, match="router_layers_index"):
        build_moe(config, torch_dtype=torch.float32)
    assert not os.path.exists("models_merge/norouter/config.json")


def test_saving_again_keeps_the_rotary_fields_of_transformers_4(experts, build, tmp_path):
    """A model trained (saved again) with transformers 5 still loads the same with transformers 4."""
    built = json.load(open(os.path.join(build("btx", experts["llama"], "llamabtx"), "config.json")))
    load(build("btx", experts["llama"], "llamabtx")).save_pretrained(tmp_path / "saved")
    saved = json.load(open(tmp_path / "saved" / "config.json"))
    for key in ("rope_theta", "rope_scaling", "rope_local_base_freq"):
        assert saved.get(key) == built.get(key), key
    assert "rope_theta" in saved
