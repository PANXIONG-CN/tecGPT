import torch
import torch.nn as nn


class SpatiotemporalLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k // 2
        self.hid_ch = hid_ch
        self.conv_x = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=p)
        self.conv_m = nn.Conv2d(hid_ch, 3 * hid_ch, k, padding=p)

    def forward(self, x, h, c, m):
        gates = self.conv_x(torch.cat([x, h], 1))
        i, f, o, g = torch.chunk(gates, 4, 1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        m = torch.tanh(self.conv_m(m))
        h = o * torch.tanh(c + m)
        return h, c, m


class PredRNN(nn.Module):
    def __init__(self, args_predictor, device, dim_in, dim_out):
        super().__init__()
        self.H = getattr(args_predictor, 'height', 71)
        self.W = getattr(args_predictor, 'width', 73)
        self.out_win = getattr(args_predictor, 'output_window', 12)
        ch_in = dim_in
        ch_hid = getattr(args_predictor, 'hidden', 64)
        layers = 2
        self.cells = nn.ModuleList([SpatiotemporalLSTMCell(ch_in if i == 0 else ch_hid, ch_hid) for i in range(layers)])
        self.head = nn.Conv2d(ch_hid, dim_out, 1)

    def forward(self, x):
        B, T, N, C = x.shape
        assert N == self.H * self.W
        x = x.permute(0, 1, 3, 2).contiguous().view(B, T, C, self.H, self.W)
        h = [None] * len(self.cells); c = [None] * len(self.cells); m = None
        outs = []
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.cells):
                if h[i] is None:
                    h[i] = torch.zeros(x.size(0), cell.hid_ch, self.H, self.W, device=x.device)
                    c[i] = torch.zeros_like(h[i])
                    if m is None:
                        m = torch.zeros_like(h[i])
                h[i], c[i], m = cell(inp, h[i], c[i], m); inp = h[i]
            outs.append(self.head(inp))
        y = torch.stack(outs[-self.out_win:], 1).view(B, self.out_win, -1, 1)
        return y

