import torch
import torch.nn as nn
from .modules import SpatialEncoder, MultiScaleConvEmbedder, LatentPatching, LLMBackbone, PredictionHead


class TEC_MoLLM(nn.Module):
    def __init__(
        self,
        num_nodes=5183,
        d_emb=0,
        spatial_in_base=1,
        spatial_out=11,
        heads=2,
        temporal_channels=(64, 128),
        temporal_strides=(2, 2),
        patch_len=4,
        d_llm=768,
        llm_layers=3,
        pred_horizon=12,
        temporal_seq_len=12,
        node_chunk=512,
        use_ln: bool = False,
        hf_model_name: str = None,
        hf_cache_dir: str = None,
    ):
        super().__init__()
        self.spatial = SpatialEncoder(in_ch=spatial_in_base + d_emb, out_ch=spatial_out, heads=heads)
        t_in = self.spatial.out_dim
        self.temporal = MultiScaleConvEmbedder(in_ch=t_in, ch_list=temporal_channels, strides=temporal_strides)
        conv_out_len = temporal_seq_len
        for s in temporal_strides:
            conv_out_len //= s
        if conv_out_len <= 0:
            conv_out_len = 1
        if patch_len <= 0 or (conv_out_len % patch_len) != 0:
            patch_len = 1
        self.patch = LatentPatching(latent_dim=self.temporal.out_ch, patch_len=patch_len, d_llm=d_llm)
        self.llm = LLMBackbone(layers=llm_layers, d_llm=d_llm, hf_model_name=hf_model_name, cache_dir=hf_cache_dir, local_only=True, freeze=True)
        num_patches = max(1, conv_out_len // patch_len)
        self.head = PredictionHead(in_dim=d_llm * num_patches, out_dim=pred_horizon)
        self.num_nodes = num_nodes
        self.pred_horizon = pred_horizon
        self.node_chunk = max(1, int(node_chunk))
        self.use_ln = bool(use_ln)
        if self.use_ln:
            self.ln = nn.LayerNorm(self.spatial.out_dim)

    def forward(self, x, tod, edge_index):
        # x: [B, L, N, C=1]; tod: [B, L, N, 1] (unused in minimal variant)
        B, L, N, C = x.shape
        # Spatial per-timestep: [B,L,N,C] -> [L,B,N,C] -> [L*B,N,C]
        x_4g = x.permute(1, 0, 2, 3).reshape(-1, N, C)
        s = self.spatial(x_4g, edge_index)  # [L*B, N, Cs]
        # Residual align if possible
        if s.size(-1) % C == 0:
            s = s + x_4g.repeat(1, 1, s.size(-1) // C)
        # Back to [B,N,L,Cs]
        Cs = s.size(-1)
        s = s.view(L, B, N, Cs).permute(1, 2, 0, 3)
        # Temporal + LLM + head with node chunking to limit peak memory
        L_out = self.pred_horizon
        y_all = x.new_zeros((B, L_out, N, 1))
        # To avoid oversized batch for attention kernels, cap B*Nc <= 16384
        # Effective chunk adapts to batch size dynamically.
        chunk = min(self.node_chunk, max(1, 16384 // max(1, B)))
        for start in range(0, N, chunk):
            end = min(N, start + chunk)
            s_ch = s[:, start:end]  # [B, Nc, L, Cs]
            if hasattr(self, 'ln'):
                s_ch = self.ln(s_ch)
            Bc, Nc = B, (end - start)
            t = s_ch.reshape(-1, L, Cs).transpose(1, 2)  # [B*Nc, Cs, L]
            t = self.temporal(t).transpose(1, 2)        # [B*Nc, L', Ct]
            p = self.patch(t)                           # [B*Nc, P, d_llm]
            attn_mask = torch.ones(p.shape[:-1], device=p.device, dtype=torch.long)
            h = self.llm(inputs_embeds=p, attention_mask=attn_mask)
            y = self.head(h)                            # [B*Nc, L_out]
            y = y.view(Bc, -1, Nc, 1)                  # [B, L_out, Nc, 1]
            y_all[:, :, start:end, :] = y
        return y_all
