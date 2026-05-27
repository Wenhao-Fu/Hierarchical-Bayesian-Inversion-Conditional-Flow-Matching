import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emd = self.emb_layer(t)
        emd = emd[:, :, None, None]
        emb = emd.repeat(1, 1, x.shape[-2], x.shape[-1])
        # emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


# model = Down(32, 64)
# input = torch.randn(16, 32, 64, 64)
# cond = torch.randn(16, 256)
# out = model(input, cond)
# print(out.shape)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


# Sinusoidal Positional Embedding，用于将噪声尺度 sigma 转换成高维嵌入
class SinusoidalPosEmb(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, x):
        """
        x: tensor shape [B]
        output: [B, emb_dim]
        """
        device = x.device
        half_dim = self.emb_dim // 2
        # 这里常数 10000 可调
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda", cond=None):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)
        self.down1 = Down(32, 64)
        self.sa1 = SelfAttention(64, 32)
        self.down2 = Down(64, 128)
        self.sa2 = SelfAttention(128, 16)
        self.down3 = Down(128, 128)
        self.sa3 = SelfAttention(128, 8)

        self.bot1 = DoubleConv(128, 256)
        self.bot2 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 128)

        self.up1 = Up(256, 64)
        self.sa4 = SelfAttention(64, 16)
        self.up2 = Up(128, 32)
        self.sa5 = SelfAttention(32, 32)
        self.up3 = Up(64, 32)
        # self.sa6 = SelfAttention(32, 64)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

        if cond is not None:
            # self.label_emb = nn.Embedding(num_classes, time_dim)
            self.fc = nn.Sequential(
                nn.Linear(cond, 1024),
                nn.SiLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 256),
            )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, conds=None):
        pos_embedding = t.unsqueeze(-1).type(torch.float) * 1000
        pos_embedding = self.pos_encoding(pos_embedding, self.time_dim)
        # pos_embedding = self.pos_emb(sigma)

        if conds is not None:
            conds = self.fc(conds)
            pos_embedding += conds  # (1)
            # t = torch.cat((t, y), 1)  # (2)

        # print(x.shape)
        x1 = self.inc(x)  # (N, 32, 64, 64)
        # print(x1.shape)
        x2 = self.down1(x1, pos_embedding)
        # print(x2.shape)
        x2 = self.sa1(x2)
        # print(x2.shape)
        x3 = self.down2(x2, pos_embedding)
        # print(x3.shape)
        x3 = self.sa2(x3)
        # print(x3.shape)
        x4 = self.down3(x3, pos_embedding)
        # print(x4.shape)
        x4 = self.sa3(x4)
        # print(x4.shape)

        x4 = self.bot1(x4)
        # print(x4.shape)
        x4 = self.bot2(x4)
        # print(x4.shape)
        x4 = self.bot3(x4)
        # print(x4.shape)

        x = self.up1(x4, x3, pos_embedding)
        # print('-----')
        # print(x.shape)
        x = self.sa4(x)
        # print(x.shape)
        x = self.up2(x, x2, pos_embedding)
        # print(x.shape)
        x = self.sa5(x)
        # print(x.shape)
        x = self.up3(x, x1, pos_embedding)
        # print(x.shape)
        # x = self.sa6(x)
        # print(x.shape)
        output = self.outc(x)
        # print(output.shape)
        return output


if __name__ == '__main__':
    # net = UNet(device="cpu")
    # from torchviz import make_dot
    net = UNet_conditional(device="cpu", cond=2)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(16, 1, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = torch.randn(16, 2)
    output = net(x, t, y)
    print(output.shape)
