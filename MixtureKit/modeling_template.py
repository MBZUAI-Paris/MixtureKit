"""
Modeling file of a Mixture-of-Experts model built with MixtureKit.

The model is the unmodified transformers causal LM of the experts, converted to a
Mixture-of-Experts when it is created:
- btx: the listed projections become routed experts, each with its own router
- traditional: the feed-forward block of every layer becomes routed expert MLPs
- bts: the experts run as parallel streams next to the hub, exchanging
  information through stitch layers

This file only depends on torch and transformers. MixtureKit writes it next to the
merged weights, with the name of the base class filled in.
"""

import copy
import functools
import inspect
import warnings
from typing import List, Optional, Sequence

import torch
import transformers
from torch import nn

from .__CONFIG_MODULE__ import MixtureKitConfig

BASE_MODEL_CLASS = "__BASE_MODEL_CLASS__"


# ----------------------------------------------------------------------------------------
# Load balancing
# ----------------------------------------------------------------------------------------


def switch_load_balancing_loss(router_logits, top_ids, num_experts: int):
    """
    The load balancing loss of Switch Transformers and Mixtral:
    num_experts * sum_i(f_i * P_i), with f_i the fraction of the routing slots sent
    to expert i and P_i the mean router probability of expert i over the tokens.
    The fraction of slots is normalized by top_k, so the loss equals 1 when the load is
    perfectly balanced and grows up to num_experts / top_k when every token is sent to
    the same experts (it is the transformers `load_balancing_loss_func` divided by top_k).
    Padding tokens are not masked and count as regular tokens.
    """
    probs = router_logits.reshape(-1, num_experts).softmax(dim=-1, dtype=torch.float)
    top_ids = top_ids.reshape(-1, top_ids.shape[-1])
    slots = torch.nn.functional.one_hot(top_ids, num_experts).float()
    tokens_per_expert = slots.mean(dim=(0, 1))
    prob_per_expert = probs.mean(dim=0)
    return num_experts * (tokens_per_expert * prob_per_expert).sum()


class RoutedModule(nn.Module):
    """
    Base of the routed layers: computes the load balancing loss of the router during
    training. MixtureKitForCausalLM adds it to the training loss, inside the autograd
    graph, so it follows the loss scaling (gradient accumulation, fp16 loss scaling).
    """

    aux_loss: Optional[torch.Tensor] = None
    # Weight of this router's loss: alpha averaged over all the routers of the model
    loss_scale: float = 0.0
    # Set by MixtureKitForCausalLM during its forward pass only: the recomputations of the
    # backward pass (gradient checkpointing) are not kept
    collect_aux_loss: bool = False

    def _add_load_balancing_loss(self, output, logits, top_ids):
        if self.training and self.loss_scale:
            # Computed in every pass, recomputations included: gradient checkpointing
            # expects the same operations, but only the forward pass keeps the value
            loss = self.loss_scale * switch_load_balancing_loss(logits, top_ids, self.num_experts)
            if self.collect_aux_loss:
                self.aux_loss = loss
        return output


def _dispatch(inputs, logits, top_k, experts):
    """Sends every token to its top-k experts and sums their weighted outputs."""
    top_logits, top_ids = logits.topk(top_k, dim=-1)
    top_weights = top_logits.softmax(dim=-1, dtype=torch.float).to(inputs.dtype)
    output = None
    for expert_id, expert in enumerate(experts):
        rows, slots = (top_ids == expert_id).nonzero(as_tuple=True)
        contribution = expert(inputs[rows]) * top_weights[rows, slots, None]
        if output is None:
            output = inputs.new_zeros(inputs.shape[0], contribution.shape[-1])
        output.index_add_(0, rows, contribution)
    return output, top_ids


# ----------------------------------------------------------------------------------------
# BTX: routed projections
# ----------------------------------------------------------------------------------------


class RoutedExperts(RoutedModule):
    """A projection replicated across experts, with a learned top-k router."""

    def __init__(self, projection: nn.Module, num_experts: int, top_k: int):
        super().__init__()
        weight = projection.weight
        if isinstance(projection, nn.Linear):
            self.in_features, self.out_features = projection.in_features, projection.out_features
        else:  # transformers Conv1D stores (in_features, out_features)
            self.in_features, self.out_features = weight.shape
        self.num_experts = num_experts
        self.num_experts_per_tok = top_k
        # The parameter names (gate, experts.expert_i) define the checkpoint layout
        self.gate = nn.Linear(
            self.in_features, num_experts, bias=False, device=weight.device, dtype=weight.dtype
        )
        self.experts = nn.ModuleDict(
            {f"expert_{i}": copy.deepcopy(projection) for i in range(num_experts)}
        )

    def forward(self, inputs):
        flat = inputs.reshape(-1, inputs.shape[-1])
        logits = self.gate(flat)
        output, top_ids = _dispatch(flat, logits, self.num_experts_per_tok, self.experts.values())
        output = self._add_load_balancing_loss(output, logits, top_ids)
        return output.reshape(*inputs.shape[:-1], self.out_features)


# ----------------------------------------------------------------------------------------
# Traditional: routed expert MLPs
# ----------------------------------------------------------------------------------------


class TraditionalMoE(RoutedModule):
    """The feed-forward block replicated across experts, with one top-k router."""

    def __init__(self, mlp: nn.Module, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        weight = next(mlp.parameters())
        self.num_experts = num_experts
        self.num_experts_per_tok = top_k
        self.gate = nn.Linear(
            hidden_size, num_experts, bias=False, device=weight.device, dtype=weight.dtype
        )
        self.experts = nn.ModuleList(copy.deepcopy(mlp) for _ in range(num_experts))

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = self.gate(flat)
        output, top_ids = _dispatch(flat, logits, self.num_experts_per_tok, self.experts)
        output = self._add_load_balancing_loss(output, logits, top_ids)
        return output.reshape(hidden_states.shape)


# ----------------------------------------------------------------------------------------
# BTS: hub, expert streams and stitch layers
# ----------------------------------------------------------------------------------------


class StitchLayer(nn.Module):
    """This class implements the BTS stitch layer (eq. 3 & eq. 4 in the paper)."""

    def __init__(self, hidden_size: int, n_experts: int, merge_into_hub: bool, device=None, dtype=None):
        """
        Parameters
        ----------
        hidden_size: int
            The dimensionality of all hidden states.
        n_experts: int
            The total number of pretrained models including the hub.
        merge_into_hub: bool
            if True, merging from Experts to Hub (eq. 3),
            else merging from Hub to Experts (eq. 4).
        """
        super().__init__()
        self.merge_into_hub = merge_into_hub
        self.n_experts = n_experts
        self.hidden_size = hidden_size
        linear = lambda: nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        # W_proj[i] projects Expert-space into Hub-space
        self.w_proj = nn.ModuleList(linear() for _ in range(self.n_experts - 1))
        self.w_gate = nn.ModuleList(linear() for _ in range(self.n_experts))
        self.dropout = nn.Dropout(0.1)

    def forward(self, xs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        assert len(xs) == self.n_experts
        x_hub, x_ex = xs[0], xs[1:]
        g = torch.stack([gate(x_hub) for gate in self.w_gate], dim=-1)  # (B, S, D, 1+N)

        if self.merge_into_hub:  # Experts to Hub
            g = torch.softmax(self.dropout(g), dim=-1)
            h_ex = [p(x_ex[i]) for i, p in enumerate(self.w_proj)]
            stacked_result = torch.stack([x_hub] + h_ex, dim=-1)  # (B, S, D, 1+N)
            h_hub = (g * stacked_result).sum(-1)  # eq. 3
            return [h_hub] + h_ex
        g = torch.sigmoid(self.dropout(g))  # eq. 4
        h_ex = [
            (1 - g[..., i + 1]) * x_ex[i] + g[..., i + 1] * self.w_proj[i](x_hub)
            for i in range(self.n_experts - 1)
        ]
        return [x_hub] + h_ex


_CACHE_EDITS = (
    "reorder_cache",
    "crop",
    "batch_repeat_interleave",
    "batch_select_indices",
    "reset",
)
_BTS_CACHE_CLASSES = {}


def _bts_cache_class(base):
    """A subclass of the cache class forwarding the edits to `bts_expert_caches`."""
    if base not in _BTS_CACHE_CLASSES:

        def forward_edit(method):
            def edit_all(self, *args, **kwargs):
                result = getattr(base, method)(self, *args, **kwargs)
                for expert_cache in self.bts_expert_caches:
                    getattr(expert_cache, method)(*args, **kwargs)
                return result

            return edit_all

        methods = {m: forward_edit(m) for m in _CACHE_EDITS if hasattr(base, m)}
        _BTS_CACHE_CLASSES[base] = type(f"Bts{base.__name__}", (base,), methods)
    return _BTS_CACHE_CLASSES[base]


def bts_expert_caches(cache, n_streams: int):
    """
    One key/value cache per expert stream. The hub keeps the cache handled by
    transformers, the experts get copies of it attached to that cache, so they persist
    across the generation steps and follow its edits (beam search, assisted decoding).
    """
    if cache is None:
        return [None] * n_streams
    if not hasattr(cache, "bts_expert_caches"):
        # Copied while still empty, before the first layer of the first forward pass
        cache.bts_expert_caches = [copy.deepcopy(cache) for _ in range(n_streams)]
        cache.__class__ = _bts_cache_class(type(cache))
    return cache.bts_expert_caches


_BTS_LAYER_CLASSES = {}
_CACHE_ARGUMENTS = ("past_key_values", "past_key_value")


def _bts_layer_class(base):
    """
    A subclass of the decoder layer running the hub layer, the expert layers of the same
    depth and the stitch layer. The streams travel packed on the last dimension of the
    hidden states, so the layer keeps the interface (and the gradient checkpointing)
    of the original decoder layer.
    """
    if base not in _BTS_LAYER_CLASSES:

        def forward(self, hidden_states, *args, **kwargs):
            experts = self.__dict__["_bts_experts"]
            streams = hidden_states.split(self.__dict__["_bts_hidden_size"], dim=-1)
            cache_argument = next((name for name in _CACHE_ARGUMENTS if name in kwargs), None)
            # Created before the hub layer writes into the shared cache
            caches = bts_expert_caches(kwargs.get(cache_argument), len(experts))

            hub_output = base.forward(self, streams[0], *args, **kwargs)
            hub = hub_output[0] if isinstance(hub_output, tuple) else hub_output
            expert_states = []
            for expert_layer, stream, cache in zip(experts, streams[1:], caches):
                expert_kwargs = dict(kwargs)
                if cache_argument is not None:
                    expert_kwargs[cache_argument] = cache
                output = expert_layer(stream, *args, **expert_kwargs)
                expert_states.append(output[0] if isinstance(output, tuple) else output)

            if hasattr(self, "stitches"):
                hub, *expert_states = self.stitches([hub] + expert_states)
            packed = torch.cat([hub] + list(expert_states), dim=-1)
            if isinstance(hub_output, tuple):
                return (packed,) + tuple(hub_output[1:])
            return packed

        _BTS_LAYER_CLASSES[base] = type(f"Bts{base.__name__}", (base,), {"forward": forward})
    return _BTS_LAYER_CLASSES[base]


def bts_stitch_layers(config):
    """The blocks followed by a stitch layer: every stitch_freq blocks and the last one."""
    n_blocks = config.num_hidden_layers
    stitch_freq = getattr(config, "stitch_freq", 5)
    stitch_layers = list(getattr(config, "stitch_layers_index", None) or []) or [
        i for i in range(n_blocks) if i % stitch_freq == 0 and i != 0
    ]
    if n_blocks - 1 not in stitch_layers:
        stitch_layers.append(n_blocks - 1)
    return stitch_layers


def convert_to_bts(decoder, config):
    n_streams = config.num_experts
    hidden_size = config.hidden_size
    layers = decoder.layers
    weight = decoder.embed_tokens.weight

    # The experts start as copies of the hub, their own weights are loaded from the checkpoint
    decoder.expert_embed_tokens = nn.ModuleList(
        copy.deepcopy(decoder.embed_tokens) for _ in range(n_streams - 1)
    )
    decoder.expert_layers = nn.ModuleList(
        nn.ModuleList(copy.deepcopy(layer) for _ in range(n_streams - 1)) for layer in layers
    )

    merge_into_hub = False  # The first stitch goes from the hub to the experts
    for block_id in bts_stitch_layers(config):
        # The last stitch layer merges the experts into the hub
        if block_id == len(layers) - 1:
            merge_into_hub = True
        layers[block_id].stitches = StitchLayer(
            hidden_size, n_streams, merge_into_hub, device=weight.device, dtype=weight.dtype
        )
        merge_into_hub = not merge_into_hub

    for block_id, layer in enumerate(layers):
        # Not registered as submodules: they are stored in decoder.expert_layers
        layer.__dict__["_bts_experts"] = list(decoder.expert_layers[block_id])
        layer.__dict__["_bts_hidden_size"] = hidden_size
        layer.__class__ = _bts_layer_class(type(layer))

    # Partial functions (not closures): a deep copy of the model gets hooks bound to its
    # own modules
    decoder.embed_tokens.register_forward_hook(functools.partial(_pack_streams, decoder))
    decoder.norm.register_forward_pre_hook(functools.partial(_keep_hub_stream, hidden_size))


def _copy_unsaved_buffers(decoder):
    """
    Buffers that are not saved (e.g. the embedding scale of Gemma) are restored by
    transformers for the hub only: the experts take them from the hub.
    """
    pairs = [(decoder.embed_tokens, copy_) for copy_ in decoder.expert_embed_tokens]
    pairs += [
        (layer, copy_) for layer, copies in zip(decoder.layers, decoder.expert_layers) for copy_ in copies
    ]
    for hub_module, copy_module in pairs:
        hub_buffers = dict(hub_module.named_buffers())
        for name, buffer in list(copy_module.named_buffers()):
            if buffer.is_meta and name in hub_buffers:
                owner, _, attribute = name.rpartition(".")
                target = copy_module.get_submodule(owner) if owner else copy_module
                target._buffers[attribute] = hub_buffers[name].clone()


def _pack_streams(decoder, module, inputs, output):
    """Embeds the tokens for the hub and every expert, packed on the last dimension."""
    if not decoder.__dict__.get("_bts_buffers_ready"):
        _copy_unsaved_buffers(decoder)
        decoder.__dict__["_bts_buffers_ready"] = True
    input_ids = inputs[0]
    return torch.cat([output] + [embed(input_ids) for embed in decoder.expert_embed_tokens], dim=-1)


def _keep_hub_stream(hidden_size, module, args):
    """The final norm and the LM head only see the hub stream."""
    return (args[0][..., :hidden_size],) + tuple(args[1:])


# ----------------------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------------------


def convert_to_moe(model, config):
    """Converts the dense causal LM into the Mixture-of-Experts described by the config."""
    decoder = model.get_decoder()
    layers = decoder.layers
    routed = []

    if config.moe_method == "btx":
        for block_id in config.router_layers_index:
            for name in config.router_layers:
                parent_name, _, attribute = name.rpartition(".")
                parent = layers[block_id].get_submodule(parent_name)
                experts = RoutedExperts(
                    getattr(parent, attribute), config.num_experts, config.num_experts_per_tok
                )
                setattr(parent, attribute, experts)
                routed.append(experts)
    elif config.moe_method == "traditional":
        for layer in layers:
            layer.mlp = TraditionalMoE(
                layer.mlp, config.hidden_size, config.num_experts, config.num_experts_per_tok
            )
            routed.append(layer.mlp)
    elif config.moe_method == "bts":
        convert_to_bts(decoder, config)
    else:
        raise ValueError(f"Unknown moe_method: {config.moe_method}")

    # alpha weights the load balancing loss averaged over all the routers of the model
    for module in routed:
        module.loss_scale = getattr(config, "alpha", 0.0) / len(routed)
    return routed


_Base = getattr(transformers, BASE_MODEL_CLASS)
_FORWARD_PARAMETERS = list(inspect.signature(_Base.forward).parameters)[1:]
_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])


def _forward_argument(args, kwargs, name):
    """An argument of the forward, passed by name or by position."""
    if name in kwargs:
        return kwargs[name]
    if name in _FORWARD_PARAMETERS and _FORWARD_PARAMETERS.index(name) < len(args):
        return args[_FORWARD_PARAMETERS.index(name)]
    return None


def load_balancing_weight(labels, num_items_in_batch=None):
    """
    The weight of the load balancing loss of a batch. With gradient accumulation (or
    several devices), transformers divides the language modeling loss of each batch by
    the number of target tokens of all the accumulated batches (`num_items_in_batch`):
    the load balancing loss gets the share of the tokens of its batch, so that it is
    averaged over the accumulated batches instead of summed.
    """
    if num_items_in_batch is None:
        return 1.0
    # Counted like transformers counts num_items_in_batch: shifted labels from version 5
    targets = labels[..., 1:] if _TRANSFORMERS_MAJOR >= 5 else labels
    n_tokens = (targets != -100).sum()
    if isinstance(num_items_in_batch, torch.Tensor):
        num_items_in_batch = num_items_in_batch.to(n_tokens.device)
    else:
        num_items_in_batch = torch.tensor(num_items_in_batch, device=n_tokens.device)
    # No target token in the accumulated batches: no loss, and no 0 / 0
    return n_tokens / num_items_in_batch.clamp(min=1)


class MixtureKitForCausalLM(_Base):
    config_class = MixtureKitConfig

    def __init__(self, config):
        super().__init__(config)
        self.__dict__["_routed_modules"] = convert_to_moe(self, config)
        if config.moe_method == "bts":
            # A hub layer runs its expert layers and stitch layer: keep them on one device
            swapped = {type(layer).__name__ for layer in self.get_decoder().layers}
            self._no_split_modules = list(getattr(self, "_no_split_modules", None) or []) + sorted(swapped)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        """
        Non-reentrant checkpointing keeps the autograd graph of the load balancing loss,
        computed inside the checkpointed layers (reentrant checkpointing drops it).
        """
        gradient_checkpointing_kwargs = dict(gradient_checkpointing_kwargs or {})
        if gradient_checkpointing_kwargs.get("use_reentrant") and self._routed_modules:
            warnings.warn(
                "use_reentrant=True would stop the load balancing loss from training the "
                "routers, non-reentrant checkpointing is used instead."
            )
        gradient_checkpointing_kwargs["use_reentrant"] = False
        return super().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs, **kwargs
        )

    def set_load_balancing_alpha(self, alpha: float):
        """Changes the weight of the load balancing loss after loading."""
        self.config.alpha = alpha
        for module in self._routed_modules:
            module.loss_scale = alpha / len(self._routed_modules)

    # Same signature as the base forward: generate() and Trainer inspect it
    @functools.wraps(_Base.forward)
    def forward(self, *args, **kwargs):
        routed = self._routed_modules
        for module in routed:
            module.collect_aux_loss = True
        try:
            outputs = super().forward(*args, **kwargs)
        finally:
            # The load balancing loss of the routers, computed during training. It is not
            # kept on the modules: the model stays copyable and keeps no graph between steps
            aux = [m.aux_loss for m in routed if m.aux_loss is not None]
            for module in routed:
                module.collect_aux_loss = False
                module.aux_loss = None
        if not aux:
            return outputs

        # aux_loss is the unweighted load balancing loss of this batch; added to the loss,
        # it is weighted like the language modeling loss (see load_balancing_weight)
        aux_loss = torch.stack([loss.float() for loss in aux]).sum()
        labels = _forward_argument(args, kwargs, "labels")
        weighted = None
        if labels is not None:
            weighted = aux_loss * load_balancing_weight(labels, kwargs.get("num_items_in_batch"))
        if isinstance(outputs, tuple):  # return_dict=False: (loss, logits, ...) with labels
            if weighted is not None:
                outputs = (outputs[0] + weighted.to(outputs[0].dtype),) + tuple(outputs[1:])
            return outputs
        # A key (not an attribute): mixed precision rebuilds the outputs from their keys
        outputs["aux_loss"] = aux_loss
        if outputs.loss is not None and weighted is not None:
            outputs["loss"] = outputs.loss + weighted.to(outputs.loss.dtype)
        return outputs


# save_pretrained copies this file and the configuration file next to the weights
MixtureKitForCausalLM.register_for_auto_class("AutoModelForCausalLM")
