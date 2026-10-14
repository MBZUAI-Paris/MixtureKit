def switch_load_balancing_loss(router_logits, top_ids, num_experts: int):
    """
    The load balancing loss of Switch Transformers and Mixtral:
    num_experts * sum_i(f_i * P_i), with f_i the fraction of the routing slots sent
    to expert i and P_i the mean router probability of expert i over the tokens.
    The fraction of slots is normalized by top_k, so the loss equals 1 when the load is
    perfectly balanced and grows up to num_experts / top_k when every token is sent to
    the same experts (it is the transformers `load_balancing_loss_func` divided by top_k).
    Padding tokens are not masked and count as regular tokens.

    Parameters
    ----------
    router_logits: torch.Tensor
        The router outputs, of shape (..., num_experts).
    top_ids: torch.Tensor
        The selected experts, of shape (..., top_k).
    num_experts: int
        The total number of experts of the router.
    """
    probs = router_logits.reshape(-1, num_experts).softmax(dim=-1, dtype=torch.float)
    top_ids = top_ids.reshape(-1, top_ids.shape[-1])
    slots = torch.nn.functional.one_hot(top_ids, num_experts).float()
    # (tokens, top_k, experts) -> fraction of the slots per expert, sums to 1
    tokens_per_expert = slots.mean(dim=(0, 1))
    prob_per_expert = probs.mean(dim=0)
    return num_experts * (tokens_per_expert * prob_per_expert).sum()
