import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sinusoidal_embedding(t, dim=256):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim_in=256, dim_out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )

    def forward(self, t_emb):
        return self.net(t_emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=256, groups=8):
        super().__init__()
        self.norm1     = nn.GroupNorm(groups, in_ch)
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2     = nn.GroupNorm(groups, out_ch)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act       = nn.SiLU()
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        t = self.time_proj(t_emb)[:, :, None, None]
        h = h + t
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H*W).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=256, use_attn=False, groups=8):
        super().__init__()
        self.res  = ResBlock(in_ch, out_ch, time_dim, groups)
        self.attn = SelfAttention(out_ch, groups=groups) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x    = self.res(x, t_emb)
        x    = self.attn(x)
        skip = x
        x    = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim=256, use_attn=False, groups=8):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.res  = ResBlock(in_ch + skip_ch, out_ch, time_dim, groups)
        self.attn = SelfAttention(out_ch, groups=groups) if use_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_dim=256, groups=8):
        super().__init__()
        self.time_emb  = TimeEmbedding(time_dim, time_dim)
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down1 = DownBlock(base_ch,   base_ch*2, time_dim, use_attn=False, groups=groups)
        self.down2 = DownBlock(base_ch*2, base_ch*4, time_dim, use_attn=False, groups=groups)
        self.down3 = DownBlock(base_ch*4, base_ch*8, time_dim, use_attn=False, groups=groups)
        self.down4 = DownBlock(base_ch*8, base_ch*8, time_dim, use_attn=True,  groups=groups)

        self.mid_res1 = ResBlock(base_ch*8, base_ch*8, time_dim, groups)
        self.mid_attn = SelfAttention(base_ch*8, groups=groups)
        self.mid_res2 = ResBlock(base_ch*8, base_ch*8, time_dim, groups)

        self.up4 = UpBlock(base_ch*8, base_ch*8, base_ch*4, time_dim, use_attn=False, groups=groups)
        self.up3 = UpBlock(base_ch*4, base_ch*8, base_ch*2, time_dim, use_attn=False, groups=groups)
        self.up2 = UpBlock(base_ch*2, base_ch*4, base_ch,   time_dim, use_attn=False, groups=groups)
        self.up1 = UpBlock(base_ch,   base_ch*2, base_ch//2,time_dim, use_attn=False, groups=groups)

        self.out_norm = nn.GroupNorm(groups, base_ch//2)
        self.out_conv = nn.Conv2d(base_ch//2, in_ch, 1)

    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, dim=256)
        t_emb = self.time_emb(t_emb)

        x = self.init_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)
        x, s4 = self.down4(x, t_emb)

        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)

        x = self.up4(x, s4, t_emb)
        x = self.up3(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)

        x = F.silu(self.out_norm(x))
        x = self.out_conv(x)
        return x