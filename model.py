# model.py
# Put this file next to main.py in your Streamlit project.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) ATTENTION U-NET (SMALL) — MATCHES YOUR best_attention_unet.pth
#    Keys look like: enc1.*, enc2.*, bottleneck.*, up2.*, att2.*, dec2.*, up1.*, att1.*, dec1.*, out.*
# ============================================================

class DoubleConvSmall(nn.Module):
    """
    Matches checkpoint pattern:
    conv.0 (Conv + bias) -> conv.1 (BN) -> conv.2 (ReLU) -> conv.3 (Conv + bias) -> conv.4 (BN) -> conv.5 (ReLU)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),   # 0
            nn.BatchNorm2d(out_ch),                              # 1
            nn.ReLU(inplace=True),                               # 2
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),  # 3
            nn.BatchNorm2d(out_ch),                              # 4
            nn.ReLU(inplace=True),                               # 5
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGateSmall(nn.Module):
    """
    Matches checkpoint keys:
    att?.Wg.weight/bias, att?.Wx.weight/bias, att?.psi.weight/bias (NO BatchNorm inside)
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.Wg = nn.Conv2d(F_g, F_int, 1, bias=True)
        self.Wx = nn.Conv2d(F_l, F_int, 1, bias=True)
        self.psi = nn.Conv2d(F_int, 1, 1, bias=True)

    def forward(self, g, x):
        psi = torch.sigmoid(self.psi(torch.relu(self.Wg(g) + self.Wx(x))))
        return x * psi


class AttentionUNetSmallCheckpoint(nn.Module):
    """
    Exact small Attention U-Net:
      enc1: DoubleConv(1 -> 32)
      enc2: DoubleConv(32 -> 64)
      bottleneck: DoubleConv(64 -> 128)
      up2: ConvT(128 -> 64), att2, dec2
      up1: ConvT(64 -> 32),  att1, dec1
      out: Conv(32 -> 1)
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConvSmall(in_channels, base)          # 1 -> 32
        self.enc2 = DoubleConvSmall(base, base * 2)             # 32 -> 64
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConvSmall(base * 2, base * 4)   # 64 -> 128

        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)  # 128 -> 64
        self.att2 = AttentionGateSmall(base * 2, base * 2, base)         # 64,64,32
        self.dec2 = DoubleConvSmall(base * 4, base * 2)                  # 128 -> 64

        self.up1  = nn.ConvTranspose2d(base * 2, base, 2, stride=2)      # 64 -> 32
        self.att1 = AttentionGateSmall(base, base, base // 2)            # 32,32,16
        self.dec1 = DoubleConvSmall(base * 2, base)                      # 64 -> 32

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        d2 = self.up2(b)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# ============================================================
# 2) ATTENTION U-NET (BIG / 4-level) — OPTIONAL
#    Use only if you have a checkpoint trained with downs/ups style
#    (May NOT match your current .pth unless trained that way)
# ============================================================

class ConvBlockBig(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlockBig(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class AttentionUNetFromWeights(nn.Module):
    """
    4-level Attention U-Net (common version).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base: int = 32, p_drop: float = 0.0):
        super().__init__()
        self.downs = nn.ModuleList([
            ConvBlockBig(in_channels, base, p_drop),
            ConvBlockBig(base, base * 2, p_drop),
            ConvBlockBig(base * 2, base * 4, p_drop),
            ConvBlockBig(base * 4, base * 8, p_drop),
        ])
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = ConvBlockBig(base * 8, base * 16, p_drop)

        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(base * 16, base * 8, 2, 2),
            ConvBlockBig(base * 16, base * 8, p_drop),

            nn.ConvTranspose2d(base * 8, base * 4, 2, 2),
            ConvBlockBig(base * 8, base * 4, p_drop),

            nn.ConvTranspose2d(base * 4, base * 2, 2, 2),
            ConvBlockBig(base * 4, base * 2, p_drop),

            nn.ConvTranspose2d(base * 2, base, 2, 2),
            ConvBlockBig(base * 2, base, p_drop),
        ])

        self.attentions = nn.ModuleList([
            AttentionBlockBig(base * 8, base * 8, base * 4),
            AttentionBlockBig(base * 4, base * 4, base * 2),
            AttentionBlockBig(base * 2, base * 2, base),
            AttentionBlockBig(base,     base,     max(1, base // 2)),
        ])

        self.final_conv = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        att_i = 0
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = self.attentions[att_i](g=x, x=skips[att_i])

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
            att_i += 1

        return self.final_conv(x)


# ============================================================
# 3) DAR-UNet — (Dual Attention + Residual blocks in encoder)
#    Use with BestModel_DAR.pth IF it was trained with same naming/structure.
# ============================================================

# =========================
# DAR-UNet (Checkpoint-matched)
# =========================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    # IMPORTANT: checkpoint uses sa.conv1.weight
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class DAR_Block(nn.Module):
    """
    IMPORTANT: checkpoint uses:
      downs.i.conv_res.0 / conv_res.1 / conv_res.4 / conv_res.5
    so we must name it conv_res and keep indices compatible.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()

        # Residual path named conv_res (matches checkpoint)
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),  # 0
            nn.BatchNorm2d(out_channels),                                     # 1
            nn.ReLU(inplace=True),                                            # 2
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),# 3
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),  # 4
            nn.BatchNorm2d(out_channels),                                     # 5
        )

        # shortcut (may or may not exist in checkpoint; safe to keep)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res_out = self.conv_res(x)
        sc = self.shortcut(x)
        pre = self.relu(res_out + sc)

        pre = pre * self.ca(pre)
        pre = pre * self.sa(pre)
        return pre


# This ConvBlock matches the decoder block keys you have:
# ups.1.conv.0, ups.1.conv.1, ups.1.conv.4, ups.1.conv.5 ...
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),               # 0
            nn.BatchNorm2d(out_ch),                                           # 1
            nn.ReLU(inplace=True),                                            # 2
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),# 3
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),              # 4
            nn.BatchNorm2d(out_ch),                                           # 5
            nn.ReLU(inplace=True),                                            # 6
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    # matches keys: attentions.i.W_g.0, W_g.1, W_x.0, W_x.1, psi.0, psi.1 ...
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class DAR_UNet(nn.Module):
    """
    Decoder naming MUST be:
      ups.0 (ConvTranspose)
      ups.1 (ConvBlock)
      ups.2 (ConvTranspose)
      ups.3 (ConvBlock)
      ...
    AND attentions.0..3
    """
    def __init__(self, in_channels=3, out_channels=1, base=32, dropout_rate=0.0):
        super().__init__()

        # Encoder (DAR blocks)
        self.downs = nn.ModuleList([
            DAR_Block(in_channels, base, dropout_rate),          # 3 -> 32
            DAR_Block(base, base * 2, dropout_rate),             # 32 -> 64
            DAR_Block(base * 2, base * 4, dropout_rate),         # 64 -> 128
            DAR_Block(base * 4, base * 8, dropout_rate),         # 128 -> 256
        ])
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DAR_Block(base * 8, base * 16, dropout_rate)       # 256 -> 512

        # Decoder (same style as your checkpoint: ups + attentions)
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(base * 16, base * 8, 2, 2),       # ups.0 512->256
            ConvBlock(base * 16, base * 8, dropout_rate),        # ups.1 (concat 256+256)

            nn.ConvTranspose2d(base * 8, base * 4, 2, 2),        # ups.2 256->128
            ConvBlock(base * 8, base * 4, dropout_rate),         # ups.3

            nn.ConvTranspose2d(base * 4, base * 2, 2, 2),        # ups.4 128->64
            ConvBlock(base * 4, base * 2, dropout_rate),         # ups.5

            nn.ConvTranspose2d(base * 2, base, 2, 2),            # ups.6 64->32
            ConvBlock(base * 2, base, dropout_rate),             # ups.7
        ])

        self.attentions = nn.ModuleList([
            AttentionBlock(base * 8, base * 8, base * 4),
            AttentionBlock(base * 4, base * 4, base * 2),
            AttentionBlock(base * 2, base * 2, base),
            AttentionBlock(base,     base,     max(1, base // 2)),
        ])

        # final layer name could be out or final_conv; checkpoint sometimes uses final_conv
        self.final_conv = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]  # 256,128,64,32

        att_i = 0
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # ConvTranspose
            skip = skips[att_i]

            # attention on skip
            skip = self.attentions[att_i](g=x, x=skip)

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)  # ConvBlock

            att_i += 1

        return self.final_conv(x)
