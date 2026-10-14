def make_moe_projection(
    name: str,
    config: dict,
    layer_idx: int,
    in_features: int,
    out_features: int,
    bias: bool = True,
    mlp_function: str = "linear",
    block_transformer: str = "mlp",
):
    """
    This function builds the projection used at a given position of the transformer block.
    Projections listed in the routed layers become a routed set of experts (BTX),
    all the others keep their original dense form.

    Parameters
    ----------
    name: str
        The name of the corresponding layer.
    config: dict
        The configuration of the composed checkpoint.
    layer_idx: int
        The id of the transformer block.
    in_features: int
        The first dimension passed to the original layer constructor.
    out_features: int
        The second dimension passed to the original layer constructor.
    bias: bool, default=True
        Whether bias is applied in the corresponding layer.
    mlp_function: str, default="linear"
        Whether it is a "linear" or "conv" layer.
        The Conv1D is just a linear layer with transposed weights as introduced
        in GPT2 model, its constructor takes (out_features, in_features).
    block_transformer: str, default="mlp"
        A string to specify which part of the model is being converted,
        attn (attention) or mlp.

    Returns
    -------
    A RoutedExperts module when the layer is routed, otherwise the original
    nn.Linear or Conv1D layer.
    """
    from transformers.pytorch_utils import Conv1D

    routed_names = {f"{block_transformer}.{name}", f"self_{block_transformer}.{name}"}
    is_routed = layer_idx in config.router_layers_index and not routed_names.isdisjoint(
        config.router_layers
    )

    if mlp_function == "conv":
        # Conv1D(nf, nx) maps nx -> nf
        d_in, d_out = out_features, in_features
        make_expert = lambda: Conv1D(in_features, out_features)
    else:
        d_in, d_out = in_features, out_features
        make_expert = lambda: nn.Linear(in_features, out_features, bias)

    if not is_routed:
        return make_expert()

    return RoutedExperts(
        d_in=d_in,
        d_out=d_out,
        make_expert=make_expert,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        moe_config=config,
    )


class RoutedExperts(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        make_expert,
        num_experts: int,
        top_k: int = 2,
        moe_config=None,
    ):
        """
        A projection replicated across experts with a learned top-k router.

        Parameters
        ----------
        d_in: int
            The dimension of the input features.
        d_out: int
            The dimension of the output features.
        make_expert: callable
            Returns a freshly initialized expert projection.
        num_experts: int
            The total number of experts handled by the router (gate).
        top_k: int, default=2
            The number of active experts per token, weighted with a softmax.
        moe_config: PretrainedConfig, default=None
            The model configuration. Its `alpha` weights the load balancing loss averaged
            over all the routers of the model, computed in training only.
        """
        super().__init__()
        self.in_features = d_in
        self.out_features = d_out
        self.num_experts_per_tok = top_k
        self.moe_config = moe_config
        # The parameter names (gate, experts.expert_i) define the checkpoint layout
        self.gate = nn.Linear(d_in, num_experts, bias=False)
        self.experts = nn.ModuleDict(
            {f"expert_{i}": make_expert() for i in range(num_experts)}
        )

    def forward(self, inputs: torch.Tensor):
        logits = self.gate(inputs)
        top_logits, top_ids = logits.topk(self.num_experts_per_tok, dim=-1)
        top_weights = top_logits.softmax(dim=-1, dtype=torch.float).to(inputs.dtype)

        # Dispatch over the flattened tokens
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_ids = top_ids.reshape(-1, self.num_experts_per_tok)
        flat_weights = top_weights.reshape(-1, self.num_experts_per_tok)
        output = flat_inputs.new_zeros(flat_inputs.shape[0], self.out_features)

        for expert_id, expert in enumerate(self.experts.values()):
            rows, slots = (flat_ids == expert_id).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            contribution = expert(flat_inputs[rows]) * flat_weights[rows, slots, None]
            output.index_add_(0, rows, contribution)

        # Read by the parent module within the same forward pass
        self.loss_lb = 0
        alpha = getattr(self.moe_config, "alpha", 0)
        if self.training and alpha:
            # Every routed projection of every selected block has its own router
            n_routers = len(self.moe_config.router_layers_index) * len(
                self.moe_config.router_layers
            )
            self.loss_lb = alpha / max(n_routers, 1) * switch_load_balancing_loss(
                logits, top_ids, len(self.experts)
            )

        return output.reshape(*inputs.shape[:-1], self.out_features)
