cache_selected_experts = None


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

    if block_transformer == "mlp":
        return TraditionalMoeLayer(
            in_features,
            out_features,
            num_experts=config.num_experts,
            mlp_function=mlp_function,
            bias=bias,
        )

    if mlp_function == "linear":
        return nn.Linear(in_features, out_features, bias=bias)
    elif mlp_function == "conv":
        return Conv1D(in_features, out_features)


class TraditionalMoeLayer(nn.Module):
    """
    Sparse MoE layer whose router is external.
    The router gives the indices (`selected_experts`) and the caller usually
    multiplies the returned tensor with the routing weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        mlp_function: str = "linear",
        bias: bool = True,
    ):
        super().__init__()

        from transformers.pytorch_utils import Conv1D

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

    def forward(self, x):
        """
        The choice of the selected expert per token is propagated across all the MLP layers.
        """
        batch_idx = cache_selected_experts["batch_idx"]
        token_idx = cache_selected_experts["token_idx"]
        ix = cache_selected_experts["expert_idx"]

        expert_input = x[batch_idx, token_idx] if x.dim() == 3 else x
        expert_layer = self.experts[f"expert_{ix}"]
        expert_output = expert_layer(expert_input)
        return expert_output
