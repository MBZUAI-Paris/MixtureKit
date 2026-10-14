"""
Token routing of a model merged with MixtureKit: which experts every token is sent to,
layer by layer.

    from MixtureKit.routing import load_model, trace_routing

    model, tokenizer = load_model("models_merge/gemmax")
    trace = trace_routing(model, tokenizer, "Bonjour, how are you?")
    trace.dominant_experts()      # the main expert of every token
    trace.to_html()               # a colour-coded view, displayed inline in notebooks

BTX and traditional MoE: the routers' top-k weights over the experts. BTS has no router:
the weights are the share of every stream (hub and experts) in the hub at the stitch
layers that merge the experts into the hub.
"""

import html
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

# Distinct colours that stay readable under dark text
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]


def available_devices():
    """The devices torch can use here: CUDA, Apple silicon (MPS), CPU."""
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append("mps")
    return devices + ["cpu"]


def default_device():
    """CUDA, then Apple silicon (MPS), then CPU."""
    return available_devices()[0]


def load_model(model_id, device=None, dtype=None):
    """
    This function loads a merged model and its tokenizer for inspection.

    Parameters
    ----------
    model_id: str
        A local directory or a Hub id of a model merged with MixtureKit.
    device: str, default=None
        "cuda", "mps" or "cpu"; the first one available by default.
    dtype: torch.dtype, default=None
        bfloat16 on CUDA, float32 elsewhere by default.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or default_device()
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    from ._streaming import dtype_kwargs

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, **dtype_kwargs(dtype)
    )
    return model.to(device).eval(), tokenizer


def expert_names(config):
    """The names of the experts (streams for BTS) stored in the merged configuration."""
    names = list(getattr(config, "expert_names", None) or [])
    n_experts = getattr(config, "num_experts", len(names))
    names = names[:n_experts] + [f"expert {i}" for i in range(len(names), n_experts)]
    if getattr(config, "moe_method", None) == "bts" and names:
        names[0] = f"{names[0]} (hub)"
    # Copies of one model share a name: they are numbered
    return [f"{name} #{i}" if names.count(name) > 1 else name for i, name in enumerate(names)]


@dataclass
class RoutingTrace:
    """
    The routing of one text.

    Attributes
    ----------
    tokens: list of str
        The text of every token.
    experts: list of str
        The expert names, in router order.
    layers: dict
        Layer name -> tensor (tokens, experts): the weight of every expert for every token,
        0 for the experts a token is not sent to. Every row sums to 1.
    method: str
        "btx", "traditional" or "bts".
    """

    tokens: List[str]
    experts: List[str]
    layers: Dict[str, torch.Tensor] = field(default_factory=dict)
    method: str = "btx"

    @property
    def text(self) -> str:
        """The tokens put back together."""
        return "".join(self.tokens)

    def mean_weights(self, layers: Optional[List[str]] = None) -> torch.Tensor:
        """(tokens, experts): the weights averaged over the layers."""
        chosen = [self.layers[name] for name in (layers or list(self.layers))]
        return torch.stack(chosen).mean(dim=0)

    def dominant_experts(self, layers: Optional[List[str]] = None) -> List[int]:
        """The expert with the largest average weight, for every token."""
        return self.mean_weights(layers).argmax(dim=-1).tolist()

    def expert_share(self, layers: Optional[List[str]] = None) -> List[float]:
        """The share of the tokens whose main expert is each expert."""
        dominant = self.dominant_experts(layers)
        return [dominant.count(e) / max(len(dominant), 1) for e in range(len(self.experts))]

    def layer_load(self) -> torch.Tensor:
        """(layers, experts): the mean weight of every expert in every layer."""
        return torch.stack([weights.mean(dim=0) for weights in self.layers.values()])

    def to_html(self, layers: Optional[List[str]] = None, title: Optional[str] = None) -> str:
        """
        A self-contained, colour-coded view of the routing: every token is coloured by its
        main expert, more or less opaque with that expert's average weight. Notebooks
        display it inline.
        """
        weights = self.mean_weights(layers)
        dominant = weights.argmax(dim=-1).tolist()
        share = self.expert_share(layers)
        legend = "".join(
            f"<span style='display:inline-block;margin:2px 10px 2px 0'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;"
            f"background:{PALETTE[e % len(PALETTE)]};vertical-align:middle;margin-right:6px'></span>"
            f"{html.escape(name)} <span style='opacity:.7'>({share[e]:.0%} of tokens)</span></span>"
            for e, name in enumerate(self.experts)
        )
        spans = []
        for token, expert, row in zip(self.tokens, dominant, weights.tolist()):
            colour = PALETTE[expert % len(PALETTE)]
            alpha = 0.25 + 0.65 * row[expert]
            tip = " | ".join(f"{self.experts[e]}: {w:.0%}" for e, w in enumerate(row) if w > 0.005)
            text = html.escape(token).replace("\n", "↵<br>") or "&nbsp;"
            spans.append(
                f"<span title='{html.escape(tip, quote=True)}' style='background:{_rgba(colour, alpha)};"
                f"border-radius:4px;padding:1px 2px;margin:1px 0;white-space:pre-wrap'>{text}</span>"
            )
        n_layers = len(layers) if layers else len(self.layers)
        what = "stitch layers merging into the hub" if self.method == "bts" else "routed layers"
        header = html.escape(title) if title else "Token routing"
        return (
            "<div style='font-family:system-ui,sans-serif;line-height:2;color:#1f2328;"
            "background:#ffffff;padding:12px 16px;border-radius:8px;border:1px solid #d0d7de'>"
            f"<div style='font-weight:600;font-size:15px'>{header}</div>"
            f"<div style='font-size:12px;opacity:.75;margin-bottom:6px'>{self.method.upper()} · "
            f"main expert per token, averaged over {n_layers} {what}; hover a token for its weights</div>"
            f"<div style='font-size:13px;margin-bottom:8px'>{legend}</div>"
            f"<div dir='auto' style='font-size:15px'>{''.join(spans)}</div></div>"
        )

    def _repr_html_(self):
        return self.to_html()


def _rgba(hex_colour, alpha):
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _top_k_weights(logits, top_k):
    """The router weights over all the experts: softmax over the top-k logits, 0 elsewhere."""
    top_logits, top_ids = logits.float().topk(top_k, dim=-1)
    weights = torch.zeros_like(logits, dtype=torch.float)
    return weights.scatter(-1, top_ids, top_logits.softmax(dim=-1))


def _layer_label(name):
    """model.layers.3.mlp.up_proj -> layer 3 · mlp.up_proj"""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part.isdigit() and i > 0 and parts[i - 1] == "layers":
            rest = ".".join(parts[i + 1 :])
            return f"layer {part}" + (f" · {rest}" if rest else "")
    return name


def trace_routing(model, tokenizer, text, add_special_tokens=True) -> RoutingTrace:
    """
    This function runs the model on a text and records the routing of every token.

    Parameters
    ----------
    model: PreTrainedModel
        A model merged with MixtureKit (loaded with trust_remote_code=True).
    tokenizer: PreTrainedTokenizer
        Its tokenizer.
    text: str
        The text to route.
    add_special_tokens: bool, default=True
        Whether the tokenizer adds its special tokens (e.g. BOS). They are not shown.
    """
    config = model.config
    method = getattr(config, "moe_method", None)
    if method not in ("btx", "traditional", "bts"):
        raise ValueError("This model was not merged with MixtureKit (no moe_method in its config)")
    names = {id(module): name for name, module in model.named_modules()}
    top_k = getattr(config, "num_experts_per_tok", 1)
    recorded = {}

    encoded = _encode(tokenizer, text, add_special_tokens)
    input_ids = encoded["input_ids"].to(model.device)
    if input_ids.shape[-1] == 0:
        raise ValueError("The text has no token to route")

    # One trace at a time: the hooks of a shared model (e.g. a Space) would otherwise
    # also record the forward passes of other users
    with _TRACE_LOCK:
        hooks = []
        try:
            if method == "bts":
                for name, module in model.named_modules():
                    if name.endswith(".stitches") and getattr(module, "merge_into_hub", False):
                        hooks.append(module.register_forward_hook(_stream_share_hook(recorded, name)))
            else:
                routed = model.__dict__.get("_routed_modules") or [
                    m for m in model.modules() if hasattr(m, "gate") and hasattr(m, "experts")
                ]
                for module in routed:
                    name = names.get(id(module), "router")
                    hooks.append(module.gate.register_forward_hook(_router_hook(recorded, name, top_k)))
            if not hooks:
                raise ValueError("No router or stitch layer found in this model")
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            for hook in hooks:
                hook.remove()

    tokens, keep = _token_texts(tokenizer, text, encoded)
    n_ids = input_ids.shape[-1]
    layers = {}
    for name, weights in recorded.items():
        layers[_layer_label(name)] = weights.reshape(n_ids, -1).cpu()[keep]
    layers = dict(sorted(layers.items(), key=lambda item: _layer_order(item[0])))
    return RoutingTrace(tokens=tokens, experts=expert_names(config), layers=layers, method=method)


_TRACE_LOCK = threading.Lock()


def _router_hook(recorded, name, top_k):
    def record(gate, inputs, logits):
        recorded[name] = _top_k_weights(logits, top_k)

    return record


def _stream_share_hook(recorded, name):
    """
    BTS: the share of every stream (hub, experts) in the hub merged by the stitch layer,
    as the norm of its gated contribution relative to the sum of the norms.
    """

    def record(module, inputs, output):
        x_hub, x_experts = inputs[0][0], inputs[0][1:]
        gates = torch.stack([gate(x_hub) for gate in module.w_gate], dim=-1).float().softmax(dim=-1)
        streams = [x_hub] + [proj(x) for proj, x in zip(module.w_proj, x_experts)]
        contribution = (gates * torch.stack(streams, dim=-1).float()).norm(dim=-2)
        recorded[name.rsplit(".stitches", 1)[0]] = contribution / contribution.sum(-1, keepdim=True).clamp(min=1e-12)

    return record


def _encode(tokenizer, text, add_special_tokens):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if getattr(tokenizer, "is_fast", False):
        return tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens, return_offsets_mapping=True
        )
    return tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)


def _token_texts(tokenizer, text, encoded):
    """
    The text of every token and the positions kept (the special tokens the tokenizer
    added are dropped, special tokens written in the text are kept). With a fast
    tokenizer, a token's text is its span of the original text, preceded by the spaces
    before it: the tokens read back as the text, spaces and multi-byte characters included
    (pieces of one character show the character once).
    """
    ids = encoded["input_ids"][0].tolist()
    special = set(tokenizer.all_special_ids)
    if "offset_mapping" in encoded:
        offsets = encoded["offset_mapping"][0].tolist()
        keep, tokens, previous_end = [], [], 0
        for i, (token_id, (start, end)) in enumerate(zip(ids, offsets)):
            if token_id in special and start == end:
                continue  # added by the tokenizer
            keep.append(i)
            tokens.append(text[min(previous_end, start) : end] if end > previous_end else "")
            previous_end = max(previous_end, end)
        return tokens, keep
    # Slow tokenizers: each token is what it adds to the decoded text
    keep = [i for i, token_id in enumerate(ids) if token_id not in special]
    tokens, decoded = [], ""
    for i in keep:
        upto = tokenizer.decode([ids[k] for k in keep if k <= i])
        tokens.append(upto[len(decoded):])
        decoded = upto
    return tokens, keep


def _layer_order(label):
    parts = label.split(" ", 2)
    return (int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1, label)
