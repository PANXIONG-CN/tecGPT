import torch
from torch import nn
from .vendor.Models.CSA_WTConvLSTM_Model import CSA_WTConvLSTM as VendorCSA


class CSA_WTConvLSTM(nn.Module):
    def __init__(self, args_predictor, device, dim_in, dim_out):
        super().__init__()
        H = getattr(args_predictor, 'height')
        W = getattr(args_predictor, 'width')
        self.H = H
        self.W = W
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.input_window = args_predictor.input_window
        self.output_window = args_predictor.output_window
        self.core = VendorCSA(
            input_dim=dim_in,
            CSA_hidden_dim=args_predictor.CSA_hidden_dim,
            CSA_num_layers=args_predictor.CSA_num_layers,
            WTConvLSTM_hidden_dim=args_predictor.WTConvLSTM_hidden_dim,
            WT_num_layers=args_predictor.WT_num_layers,
            height=H,
            width=W,
            kernel_size=args_predictor.kernel_size,
            CA_bool=args_predictor.CA_bool,
            SA_bool=args_predictor.SA_bool,
            predict_step=self.output_window,
            batch_first=True,
            channel_second=False,
        )

    def forward(self, x):
        B, T, N, C = x.shape
        assert C == self.dim_in
        assert N == self.H * self.W
        xg = x.permute(0, 1, 3, 2).contiguous().view(B, T, C, self.H, self.W)
        y = self.core(xg)  # [B, T_out, 1, H, W]
        y = y.view(B, self.output_window, self.dim_out, -1).permute(0, 1, 3, 2).contiguous()
        return y
