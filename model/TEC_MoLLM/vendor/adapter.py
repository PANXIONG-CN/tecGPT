import torch
import torch.nn as nn
from .tec_mollm import TEC_MoLLM


class TECMoLLMAdapter(nn.Module):
    """Adapter to interface CSA-WTConvLSTM batches with TEC_MoLLM.

    Input:  x in [B, T, 1, H, W] normalized (0..1)
    Output: y in [B, T, 1, H, W] normalized (0..1)
    """

    def __init__(self, H=71, W=73, **cfg):
        super().__init__()
        self.H, self.W = H, W
        self.N = H * W
        self.model = TEC_MoLLM(num_nodes=self.N, **cfg)
        self.edge_index = None

    def set_edge_index(self, edge_index: torch.Tensor):
        self.edge_index = edge_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.edge_index is not None, 'edge_index not set.'
        B, T, C, H, W = x.shape
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B, T, self.N, C)  # [B,T,N,1]
        tod = torch.arange(T, device=x.device).view(1, T, 1, 1).expand(B, T, self.N, 1)
        edge_index = self.edge_index
        if edge_index.device != x.device:
            edge_index = edge_index.to(x.device)
        y = self.model(x_flat, tod, edge_index)  # [B,T,N,1]
        y = y.view(B, T, self.H, self.W, 1).permute(0, 1, 4, 2, 3)
        return y

