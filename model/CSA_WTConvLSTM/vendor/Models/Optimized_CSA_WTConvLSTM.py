"""
从 VendorCode 复制的优化版 CSA-WTConvLSTM（保留原注释与结构）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial
import math

from ..Modules.CSA_Module import CSA_Module
from ..Modules.WTConvLSTM_Modules import WTConvLSTMCell


class EfficientWTConvLSTMCell(nn.Module):
    """高效的WTConvLSTM单元，使用深度可分离卷积和融合操作"""
    
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride=1):
        super().__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        
        # 使用深度可分离卷积替代标准卷积
        self.depthwise_x = nn.Conv2d(
            in_channel, in_channel * 4, 
            kernel_size=filter_size, 
            stride=stride, 
            padding=self.padding,
            groups=in_channel,  # 深度卷积
            bias=False
        )
        self.pointwise_x = nn.Conv2d(
            in_channel * 4, num_hidden * 4,
            kernel_size=1,
            bias=False
        )
        
        self.depthwise_h = nn.Conv2d(
            num_hidden, num_hidden * 4,
            kernel_size=filter_size,
            stride=stride,
            padding=self.padding, 
            groups=num_hidden,  # 深度卷积
            bias=False
        )
        self.pointwise_h = nn.Conv2d(
            num_hidden * 4, num_hidden * 4,
            kernel_size=1,
            bias=False
        )
        
        # 层归一化
        self.ln_x = nn.LayerNorm([num_hidden * 4, height, width])
        self.ln_h = nn.LayerNorm([num_hidden * 4, height, width])
        
        # Gate融合优化：减少sigmoid计算次数
        self.gate_norm = nn.LayerNorm([num_hidden * 3, height, width])
        
    def forward(self, x_t, h_t, c_t):
        # 深度可分离卷积
        x_conv = self.depthwise_x(x_t)
        x_conv = self.pointwise_x(x_conv)
        x_concat = self.ln_x(x_conv)
        
        h_conv = self.depthwise_h(h_t)
        h_conv = self.pointwise_h(h_conv)
        h_concat = self.ln_h(h_conv)
        
        # 分割门控
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        
        # 融合门控计算 - 减少内存访问
        gates = torch.cat([
            i_x + i_h,
            f_x + f_h, 
            o_x + o_h
        ], dim=1)
        
        gates = self.gate_norm(gates)
        i_t, f_t, o_t = torch.split(torch.sigmoid(gates), self.num_hidden, dim=1)
        g_t = torch.tanh(g_x + g_h)
        
        # LSTM状态更新
        c_new = f_t * c_t + i_t * g_t
        h_new = o_t * torch.tanh(c_new)
        
        return h_new, c_new
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.num_hidden, height, width, device=device),
            torch.zeros(batch_size, self.num_hidden, height, width, device=device),
        )


class OptimizedCSA_WTConvLSTM(nn.Module):
    """优化版CSA-WTConvLSTM模型"""
    
    def __init__(
        self, 
        input_dim, 
        CSA_hidden_dim, 
        CSA_num_layers,
        WTConvLSTM_hidden_dim, 
        WT_num_layers,
        height, 
        width, 
        kernel_size,
        predict_step=12,
        use_checkpoint=True,
        use_flash_attn=False,
        use_compile=True,
        batch_first=True,
        bias=False,
        return_all_layers=False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.CSA_hidden_dim = CSA_hidden_dim
        self.CSA_num_layers = CSA_num_layers
        self.WTConvLSTM_hidden_dim = WTConvLSTM_hidden_dim
        self.WT_num_layers = WT_num_layers
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.predict_step = predict_step
        
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        self.use_checkpoint = use_checkpoint
        
        # CSA模块
        self.CSA_Module = CSA_Module(
            in_channel=self.input_dim,
            CSA_hidden_dim=self.CSA_hidden_dim,
            CSA_num_layers=self.CSA_num_layers,
            kernel_size=self.kernel_size,
            channel_second=False,
            CA_bool=True,
            SA_bool=True
        )
        
        # 使用优化的WTConvLSTM单元
        cell_list = []
        for i in range(WT_num_layers):
            cur_input_dim = CSA_hidden_dim[-1] if i == 0 else WTConvLSTM_hidden_dim[i - 1]
            cell_list.append(
                EfficientWTConvLSTMCell(
                    in_channel=cur_input_dim,
                    num_hidden=WTConvLSTM_hidden_dim[i],
                    height=height,
                    width=width,
                    filter_size=kernel_size
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        
        # 批归一化
        self.layer_bns = nn.ModuleList([
            nn.BatchNorm2d(WTConvLSTM_hidden_dim[i]) 
            for i in range(WT_num_layers)
        ])
        
        # 输出层 - 使用1x1卷积减少参数
        self.conv_reduce = nn.Conv2d(
            WTConvLSTM_hidden_dim[-1],
            WTConvLSTM_hidden_dim[-1] // 2,
            kernel_size=1
        )
        self.Conv_last = nn.Conv3d(
            in_channels=WTConvLSTM_hidden_dim[-1] // 2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=1
        )
        self.sig = nn.Sigmoid()
        
        # 编译优化(PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            self._forward_impl = torch.compile(self._forward_impl)
    
    def _forward_impl(self, input_tensor, hidden_state=None):
        """前向传播实现"""
        # CSA模块
        if self.use_checkpoint:
            input_tensor = checkpoint(self.CSA_Module, input_tensor)
        else:
            input_tensor = self.CSA_Module(input_tensor)
        
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4).contiguous()
        
        b, seq_len, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.WT_num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # 时间步循环 - 考虑使用JIT编译
            for t in range(seq_len):
                if self.use_checkpoint and t % 2 == 0:  # 每隔一步使用checkpoint
                    h, c = checkpoint(
                        self.cell_list[layer_idx],
                        cur_layer_input[:, t, :, :, :], h, c
                    )
                else:
                    h, c = self.cell_list[layer_idx](
                        cur_layer_input[:, t, :, :, :], h, c
                    )
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            
            # 批归一化 - 融合reshape操作
            b, t, cch, hh, ww = layer_output.shape
            layer_output = self.layer_bns[layer_idx](
                layer_output.reshape(b * t, cch, hh, ww)
            ).reshape(b, t, cch, hh, ww)
            
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]
        
        # 输出处理
        output = layer_output_list[:, -self.predict_step:, :, :, :]
        
        # 降维以减少计算量
        b, t, c, h, w = output.shape
        output = output.reshape(b * t, c, h, w)
        output = self.conv_reduce(output)
        output = output.reshape(b, t, -1, h, w)
        
        # 转换为3D卷积格式
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        output = self.Conv_last(output)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        output = self.sig(output)
        
        return output
    
    def forward(self, input_tensor, hidden_state=None):
        """前向传播"""
        return self._forward_impl(input_tensor, hidden_state)
    
    def _init_hidden(self, batch_size, image_size):
        """初始化隐藏状态"""
        init_states = []
        for i in range(self.WT_num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size)
            )
        return init_states
