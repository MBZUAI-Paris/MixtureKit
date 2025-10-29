from typing import Sequence, List
import torch
from torch import nn


class StitchLayer(nn.Module):
    """This class implements the BTS stitch layer (eq. 3 & eq. 4 in the paper)."""

    def __init__(self, hidden_size: int, n_experts: int, merge_into_hub: bool):
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

        # W_proj[i] projects Expert-space into Hub-space
        self.w_proj = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            for _ in range(self.n_experts - 1)
        )

        self.w_gate = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            for _ in range(self.n_experts)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, xs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        xs: list
            returns the same-length list of updated states {'[x_hub, x_1, ..., x_N]'}.

        """
        assert len(xs) == self.n_experts
        x_hub, x_ex = xs[0], xs[1:]

        g = torch.stack(
            [gate(x_hub) for i, gate in enumerate(self.w_gate)], dim=-1
        )  # (B, S, D, 1+N)

        if self.merge_into_hub:  # Experts to Hub
            g = torch.softmax(self.dropout(g), dim=-1)
            h_ex = [p(x_ex[i]) for i, p in enumerate(self.w_proj)]
            stacked_result = torch.stack([x_hub] + h_ex, dim=-1)  # (B, S, D, 1+N)
            h_hub = (g * stacked_result).sum(-1)  # eq. 3
            return [h_hub] + h_ex
        else:  # Hub to Experts
            g = torch.sigmoid(self.dropout(g))  # eq. 4
            h_ex = [
                (1 - g[..., i + 1]) * x_ex[i] + g[..., i + 1] * self.w_proj[i](x_hub)
                for i in range(self.n_experts - 1)
            ]
            return [x_hub] + h_ex
