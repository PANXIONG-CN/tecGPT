import torch
from torch import nn


# Reuse vendor implementation (optimized) if available; otherwise fall back to baseline vendor.
try:
    from model.CSA_WTConvLSTM.vendor.Models.Optimized_CSA_WTConvLSTM import OptimizedCSA_WTConvLSTM as VendorOptimized
except Exception:
    from model.CSA_WTConvLSTM.vendor.Models.CSA_WTConvLSTM_Model import CSA_WTConvLSTM as VendorOptimized


class OptimizedCSA_WTConvLSTM(nn.Module):
    """
    Wrapper to adapt optimized CSA-WTConvLSTM to repo predictor signature.
    Input: [B, T, N, C] -> reshape to [B,T,C,H,W]; Output: [B, T_out, N, 1]
    """
    def __init__(self, args_predictor, device, dim_in, dim_out):
        super().__init__()
        H = getattr(args_predictor, 'height')
        W = getattr(args_predictor, 'width')
        self.H = H
        self.W = W
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.output_window = args_predictor.output_window
        # VendorOptimized uses CSA_hidden_dim / WTConvLSTM_hidden_dim lists
        # Some optimized implementations support extra flags; pass if available via kwargs
        kwargs = {}
        for k in ['use_checkpoint', 'use_flash_attn', 'use_compile', 'batch_first', 'bias', 'return_all_layers']:
            if hasattr(VendorOptimized.__init__, '__code__') and k in VendorOptimized.__init__.__code__.co_varnames:
                if k == 'use_checkpoint':
                    kwargs[k] = True
                elif k == 'use_compile':
                    kwargs[k] = False
                else:
                    kwargs[k] = True if k in ['batch_first'] else False
        self.core = VendorOptimized(
            input_dim=dim_in,
            CSA_hidden_dim=getattr(args_predictor, 'CSA_hidden_dim', [32]),
            CSA_num_layers=getattr(args_predictor, 'CSA_num_layers', 1),
            WTConvLSTM_hidden_dim=getattr(args_predictor, 'WTConvLSTM_hidden_dim', [32]),
            WT_num_layers=getattr(args_predictor, 'WT_num_layers', 1),
            height=H,
            width=W,
            kernel_size=getattr(args_predictor, 'kernel_size', 3),
            predict_step=self.output_window,
            **kwargs
        )

    def forward(self, x):
        B, T, N, C = x.shape
        assert C == self.dim_in
        assert N == self.H * self.W
        xg = x.permute(0, 1, 3, 2).contiguous().view(B, T, C, self.H, self.W)
        y = self.core(xg)
        y = y.view(B, self.output_window, self.dim_out, -1).permute(0, 1, 3, 2).contiguous()
        return y
