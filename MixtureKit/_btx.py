def convert_linear_to_moe(
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
    This function converts nn.Linear Layer to Mixture of Experts (Moe) Layer.

    Parameters
    ----------
    name: str
        The name of the corresponding layer.
    config: dict
        The configuration of the composed checkpoint.
    layer_idx: int
        The id of the transformer block.
    in_features: int
        The dimesnion of the input features.
    out_features: int
        The dimension of the output features.
    bias: bool, default=True
        Whether bias is applied in the corresponding layer.
    mlp_function: str, default="linear"
        Whether it is a "linear" or "conv" layer.
        The Conv1D is just a linear layer with transposed weights as introduced
        in GPT2 model.
    block_transformer: str, default="mlp"
        A string to specify which part of the model is being converted,
        attn (attention) or mlp.

    Returns
    -------
    Following the input, if the provided part is to be converted, the return is a MoE class.
    Otherwise, it is a nn.Linear or Conv1D according to the MLP layer.
    """
    from transformers.pytorch_utils import Conv1D

    if (layer_idx in config.router_layers_index) and (
        f"{block_transformer}.{name}" in config.router_layers
        or f"self_{block_transformer}.{name}" in config.router_layers
    ):
        return MoeLayer(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            mlp_function=mlp_function,
            alpha=config.alpha,
        )

    if mlp_function == "linear":
        return nn.Linear(in_features, out_features, bias=bias)
    elif mlp_function == "conv":
        return Conv1D(in_features, out_features)


class MoeLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        num_experts: int,
        num_experts_per_tok: int = 2,
        mlp_function: str = "linear",
        alpha: float = 0,
    ):
        """
        The class highlighting the Mixture of Expert Layer.

        Parameters
        ----------
        in_features: int
            The dimension of the input features.
        out_features: int
            The dimension of the output features.
        bias: bool,
            Whether bias is applied in the corresponding layer.
        num_experts: int
            The total number of experts that Router Layer or Gate would handle.
        num_experts_per_tok: int, default=2
            The number of active experts per token with a softmax activation.
        mlp_function: str, default="linear"
            Whether it is a "linear" or "conv" layer.
            The Conv1D is just a linear layer with transposed weights as introduced
            in GPT2 model.
        alpha: float, default=0.01
            The regularization parameter for the load balancing loss
        """
        super().__init__()
        if mlp_function == "linear":
            self.in_features = in_features
            self.out_features = out_features
            self.experts = nn.ModuleDict(
                {
                    f"expert_{i}": nn.Linear(in_features, out_features, bias)
                    for i in range(num_experts)
                }
            )

        elif mlp_function == "conv":
            # The I/O are transposed with Conv1D
            self.in_features = out_features
            self.out_features = in_features
            self.experts = nn.ModuleDict(
                {
                    f"expert_{i}": Conv1D(in_features, out_features)
                    for i in range(num_experts)
                }
            )

        self.num_experts_per_tok = num_experts_per_tok
        self.gate = nn.Linear(in_features, num_experts, bias=False)
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor):

        logits = self.gate(inputs)
        weights, selected_experts = torch.topk(logits, self.num_experts_per_tok)
        weights = torch.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)

        results = torch.zeros(
            (inputs.shape[0], inputs.shape[1], self.out_features),
            device=inputs.device,
            dtype=inputs.dtype,
        )

        for ix, expert in enumerate(self.experts.values()):
            batch_idx, tok_idx, expert_idx = torch.where(selected_experts == ix)
            results[batch_idx, tok_idx] += expert(inputs[batch_idx, tok_idx]) * weights[
                batch_idx, tok_idx, expert_idx
            ].unsqueeze(-1)

        if self.alpha != 0:
            # Compute Load Balancing Loss
            # Reconstruct the sparse matrix using the above selected experts
            u_i = torch.zeros_like(logits)
            # We need to index scatter values into sparse_tensor at the right positions
            # First, create broadcasted indices for B and N dims
            batch_idx = (
                torch.arange(inputs.shape[0])
                .view(inputs.shape[0], 1, 1)
                .expand(weights.shape)
            )
            token_idx = (
                torch.arange(inputs.shape[1])
                .view(1, inputs.shape[1], 1)
                .expand(weights.shape)
            )

            # Now scatter values at (batch_idx, token_idx, indices)
            u_i[batch_idx, token_idx, selected_experts] = weights

            p_i = nn.functional.softmax(logits, dim=2, dtype=torch.float).to(
                inputs.dtype
            )

            current_loss_lb = (
                self.alpha * len(self.experts) / (inputs.shape[0] * inputs.shape[1])
            ) * torch.sum(u_i * p_i)
            self.loss_lb = current_loss_lb

        return results
