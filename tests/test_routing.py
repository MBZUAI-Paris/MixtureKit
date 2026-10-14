"""The routing traces behind the visualizer."""

import pytest
import torch

from MixtureKit.cli import main as cli
from MixtureKit.routing import trace_routing
from conftest import FAMILIES, load, make_tokenizer

TEXT = "the quick brown fox jumps over the lazy dog"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", ["btx", "traditional"])
def test_router_weights_are_top_k_distributions(experts, build, family, method):
    path = build(method, experts[family], f"{family}{method}")
    model, tokenizer = load(path).eval(), make_tokenizer()
    trace = trace_routing(model, tokenizer, TEXT)
    assert trace.method == method
    assert trace.text == TEXT  # the tokens read back as the text, spaces included
    n_layers = len(model._routed_modules)  # one per routed projection (BTX) or block
    assert n_layers >= 4
    assert len(trace.layers) == n_layers
    for weights in trace.layers.values():
        assert weights.shape == (len(trace.tokens), 3)
        assert torch.allclose(weights.sum(-1), torch.ones(len(trace.tokens)), atol=1e-5)
        assert ((weights > 0).sum(-1) == model.config.num_experts_per_tok).all()
    assert len(trace.dominant_experts()) == len(trace.tokens)
    assert abs(sum(trace.expert_share()) - 1) < 1e-6
    assert trace.layer_load().shape == (n_layers, 3)


def test_traced_choices_match_the_router(experts, build):
    """The recorded experts are the router's top-k for the hidden states it received."""
    model = load(build("traditional", experts["qwen3"], "qwen3traditional")).eval()
    tokenizer = make_tokenizer()
    router = model.model.layers[2].mlp.gate
    seen = []
    hook = router.register_forward_hook(lambda m, i, o: seen.append(o.detach()))
    trace = trace_routing(model, tokenizer, TEXT)
    hook.remove()
    expected = seen[0].topk(2, dim=-1).indices.sort(dim=-1).values
    recorded = (trace.layers["layer 2 · mlp"] > 0).nonzero()[:, 1].reshape(-1, 2)
    special = len(seen[0]) - len(trace.tokens)
    assert torch.equal(recorded, expected[special:])


def test_bts_trace_shows_the_stream_shares(experts, build):
    model = load(build("bts", experts["llama"], "llamabts")).eval()
    trace = trace_routing(model, make_tokenizer(), TEXT)
    assert trace.method == "bts"
    assert list(trace.layers) == ["layer 3"]  # the last stitch merges into the hub
    weights = trace.layers["layer 3"]
    assert torch.allclose(weights.sum(-1), torch.ones(len(trace.tokens)), atol=1e-5)
    assert trace.experts[0].endswith("(hub)")


def test_expert_names_come_from_the_config(experts, build):
    model = load(build("btx", experts["llama"], "llamabtx"))
    trace = trace_routing(model, make_tokenizer(), TEXT)
    assert trace.experts == ["expert_0", "expert_1", "expert_2"]


def test_html_view(experts, build):
    model = load(build("btx", experts["llama"], "llamabtx"))
    page = trace_routing(model, make_tokenizer(), "fox <b>").to_html(title="demo")
    assert "fox" in page and "demo" in page and "expert_2" in page
    assert "<b>" not in page.split("demo", 1)[1]  # token text is escaped


def test_hooks_are_removed(experts, build):
    model = load(build("btx", experts["llama"], "llamabtx"))
    count = lambda: sum(len(m._forward_hooks) for m in model.modules())
    before = count()
    trace_routing(model, make_tokenizer(), TEXT)
    assert count() == before


def test_command_writes_an_html_view(experts, build, tmp_path, capsys):
    path = build("btx", experts["llama"], "llamabtx")
    out = tmp_path / "routing.html"
    assert cli(["visualize", path, "--text", TEXT, "--html", str(out), "--device", "cpu"]) == 0
    assert "Token routing" in out.read_text(encoding="utf-8")
    assert "expert_0" in capsys.readouterr().out


def test_streamlit_app_runs(experts, build):
    pytest.importorskip("streamlit")
    pytest.importorskip("pandas")
    from streamlit.testing.v1 import AppTest
    import os

    path = build("btx", experts["llama"], "llamabtx")
    app = os.path.join(os.path.dirname(__file__), "..", "MixtureKit", "visualizer_app.py")
    os.environ["MIXTUREKIT_MODEL"] = path
    try:
        at = AppTest.from_file(app, default_timeout=120).run()
    finally:
        del os.environ["MIXTUREKIT_MODEL"]
    assert not at.exception, at.exception
    assert at.title[0].value.startswith("Which expert")


def test_token_texts_keep_spaces_symbols_and_typed_special_tokens(experts, build):
    model = load(build("btx", experts["llama"], "llamabtx"))
    text = "the  fox 🫠 jumps <eos> over"
    trace = trace_routing(model, make_tokenizer(), text)
    assert trace.text == text
    assert any("<eos>" in token for token in trace.tokens)
    assert all(weights.shape[0] == len(trace.tokens) for weights in trace.layers.values())


def test_empty_or_invalid_text_leaves_no_hook(experts, build):
    model = load(build("btx", experts["llama"], "llamabtx"))
    count = lambda: sum(len(m._forward_hooks) for m in model.modules())
    before = count()
    with pytest.raises(ValueError):
        trace_routing(model, make_tokenizer(), "")
    with pytest.raises(TypeError):
        trace_routing(model, make_tokenizer(), None)
    assert count() == before


def test_concurrent_traces_do_not_mix(experts, build):
    from concurrent.futures import ThreadPoolExecutor

    model = load(build("btx", experts["llama"], "llamabtx")).eval()
    tokenizer = make_tokenizer()
    texts = ["the fox", "the quick brown fox jumps over the lazy dog"] * 6
    with ThreadPoolExecutor(4) as pool:
        traces = list(pool.map(lambda t: trace_routing(model, tokenizer, t), texts))
    reference = {t: trace_routing(model, tokenizer, t) for t in set(texts)}
    for text, trace in zip(texts, traces):
        assert trace.text == text
        for name, weights in trace.layers.items():
            assert torch.equal(weights, reference[text].layers[name])


def test_repeated_expert_names_are_numbered():
    from types import SimpleNamespace

    from MixtureKit.routing import expert_names

    config = SimpleNamespace(expert_names=["qwen", "qwen", "math"], num_experts=3, moe_method="btx")
    assert expert_names(config) == ["qwen #0", "qwen #1", "math"]
    config = SimpleNamespace(num_experts=2, moe_method="bts")
    assert expert_names(config) == ["expert 0 (hub)", "expert 1"]
