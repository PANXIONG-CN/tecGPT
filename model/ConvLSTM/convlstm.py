import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=p)
        self.hid_ch = hid_ch

    def forward(self, x, h, c):
        hc = torch.cat([x, h], dim=1)
        i, f, o, g = torch.chunk(self.conv(hc), 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, args_predictor, device, dim_in, dim_out):
        super().__init__()
        H = getattr(args_predictor, 'height', 71)
        W = getattr(args_predictor, 'width', 73)
        self.H, self.W = H, W
        self.out_win = getattr(args_predictor, 'output_window', 12)
        ch_in = dim_in
        ch_hid = getattr(args_predictor, 'hidden', 32)
        layers = 2
        cells = []
        for _ in range(layers):
            cells.append(ConvLSTMCell(ch_in, ch_hid, k=3))
            ch_in = ch_hid
        self.cells = nn.ModuleList(cells)
        self.readout = nn.Conv2d(ch_hid, dim_out, 1)

    def forward(self, x):
        # x: [B,T,N,C] -> [B,T,C,H,W]
        B, T, N, C = x.shape
        assert N == self.H * self.W
        x = x.permute(0, 1, 3, 2).contiguous().view(B, T, C, self.H, self.W)
        h = c = None
        outs = []
        for t in range(T):
            xt = x[:, t]
            for i, cell in enumerate(self.cells):
                if h is None:
                    h = torch.zeros(x.size(0), cell.hid_ch, self.H, self.W, device=x.device)
                    c = torch.zeros_like(h)
                h, c = cell(xt, h, c); xt = h
            y = self.readout(h)  # [B,dim_out,H,W]
            outs.append(y)
        y = torch.stack(outs[-self.out_win:], dim=1)  # [B,T_out,C,H,W]
        y = y.view(B, self.out_win, -1, 1).permute(0, 1, 2, 3).contiguous()
        return y

