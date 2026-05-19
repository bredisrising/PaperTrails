import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels, groups=32):
    return nn.GroupNorm(min(groups, channels), channels)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().view(-1, 1) * freqs.view(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, t_dim, dropout=0.1):
        super().__init__()
        self.norm1 = _gn(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_c)
        self.norm2 = _gn(out_c)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = _gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16, 8),
        time_emb_dim=128,
        dropout=0.1,
        img_size=32,
    ):
        super().__init__()
        self.img_size = img_size
        t_hidden = base_channels * 4

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, t_hidden),
            nn.SiLU(),
            nn.Linear(t_hidden, t_hidden),
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.skip_channels = [base_channels]
        cur_c = base_channels
        cur_res = img_size
        for i, mult in enumerate(channel_mult):
            out_c = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(cur_c, out_c, t_hidden, dropout)])
                if cur_res in attn_resolutions:
                    block.append(SelfAttention(out_c))
                self.down_blocks.append(block)
                cur_c = out_c
                self.skip_channels.append(cur_c)
            if i != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(cur_c)]))
                self.skip_channels.append(cur_c)
                cur_res //= 2

        self.mid = nn.ModuleList([
            ResBlock(cur_c, cur_c, t_hidden, dropout),
            SelfAttention(cur_c),
            ResBlock(cur_c, cur_c, t_hidden, dropout),
        ])

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_c = base_channels * mult
            for j in range(num_res_blocks + 1):
                skip_c = self.skip_channels.pop()
                block = nn.ModuleList([ResBlock(cur_c + skip_c, out_c, t_hidden, dropout)])
                if cur_res in attn_resolutions:
                    block.append(SelfAttention(out_c))
                self.up_blocks.append(block)
                cur_c = out_c
            if i != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(cur_c)]))
                cur_res *= 2

        self.norm_out = _gn(cur_c)
        self.conv_out = nn.Conv2d(cur_c, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        h = self.conv_in(x)
        skips = [h]

        for block in self.down_blocks:
            if isinstance(block[0], Downsample):
                h = block[0](h)
            else:
                h = block[0](h, t_emb)
                if len(block) > 1:
                    h = block[1](h)
            skips.append(h)

        h = self.mid[0](h, t_emb)
        h = self.mid[1](h)
        h = self.mid[2](h, t_emb)

        for block in self.up_blocks:
            if isinstance(block[0], Upsample):
                h = block[0](h)
            else:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block[0](h, t_emb)
                if len(block) > 1:
                    h = block[1](h)

        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


if __name__ == "__main__":
    from torchinfo import summary
    net = UNet()
    summary(net, input_size=[(1, 3, 32, 32), (1,)], device="cpu")
