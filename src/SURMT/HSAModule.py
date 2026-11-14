import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import MultiScaleSpatialFeatureBlock
from timm.layers import trunc_normal_


def create_2d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi, dtype=type)
    rec_lo = torch.tensor(w.rec_lo, dtype=type)
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_2d_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_2d_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_2d_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False)
            for _ in range(self.wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([_ScaleModule([1, in_channels*4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)])

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_2d_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        if self.do_stride is not None:
            x = self.do_stride(x)
        return x


class WTConvBlock(nn.Module):
    def __init__(self, channels, wt_levels=1, wt_type='db1'):
        super(WTConvBlock, self).__init__()
        self.wtconv1 = WTConv2d(channels, channels, kernel_size=5, stride=1, bias=False, wt_levels=wt_levels, wt_type=wt_type)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.wtconv2 = WTConv2d(channels, channels, kernel_size=5, stride=1, bias=False, wt_levels=wt_levels, wt_type=wt_type)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.wtconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.wtconv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class DCTAttention(nn.Module):
    """FcaNet: 用DCT代替GAP做通道注意力"""

    def __init__(self, C, num_freqs=8):
        super().__init__()
        self.num_freqs = num_freqs
        self.fc = nn.Sequential(
            nn.Linear(num_freqs, C // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(C // 4, C, bias=False)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        orig_dtype = x.dtype

        if orig_dtype == torch.float16 or orig_dtype == torch.bfloat16:
            x = x.float()

        dct = torch.fft.rfft(x.flatten(2).transpose(1, 2), dim=-1).real  # (B, H*W, C//2+1)
        dct = dct[:, :, :self.num_freqs]  # (B, H*W, num_freqs)

        # 通道间建模
        dct = dct.transpose(1, 2)  # (B, num_freqs, H*W)
        weight = self.fc(dct.mean(-1))  # (B, C)
        weight = weight.sigmoid()

        out = x * weight.view(B, C, 1, 1)

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)

        return out


class WFCA_Simple(nn.Module):
    """简化版WFCA：频域通道注意力，取低频分量"""
    def __init__(self, C, num_freqs=8):
        super().__init__()
        self.num_freqs = num_freqs
        self.fc = nn.Sequential(
            nn.Linear(2 * C, C // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(C // 4, C, bias=False)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        orig_dtype = x.dtype

        if orig_dtype == torch.float16 or orig_dtype == torch.bfloat16:
            x = x.float()

        fft = torch.fft.rfft2(x, norm='ortho')
        real = fft.real[..., :self.num_freqs]
        imag = fft.imag[..., :self.num_freqs]

        real_stat = real.mean(dim=(2, 3))
        imag_stat = imag.mean(dim=(2, 3))

        weight = self.fc(torch.cat([real_stat, imag_stat], dim=1))
        weight = weight.sigmoid().view(B, C, 1, 1)

        out = x * weight

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)

        return out





class RoPE2D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        angle = 1.0 / (10000 ** torch.linspace(0, 1, self.head_dim // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, H: int, W: int, device=None):
        index = torch.arange(H * W, device=device or self.angle.device)

        # 计算sin和cos
        sin = torch.sin(index[:, None] * self.angle[None, :])  # (H*W, head_dim)
        cos = torch.cos(index[:, None] * self.angle[None, :])  # (H*W, head_dim)

        # 重塑为2D格式
        sin = sin.reshape(H, W, -1)  # (H, W, head_dim)
        cos = cos.reshape(H, W, -1)  # (H, W, head_dim)

        return sin, cos


class FourierAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, value_head_dim: int,
                 freq_modulation_weight=None, freq_modulation_bias=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.scaling = head_dim ** -0.5

        if freq_modulation_weight is None:
            self.freq_modulation_weight = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.freq_modulation_weight = freq_modulation_weight

        if freq_modulation_bias is None:
            self.freq_modulation_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        else:
            self.freq_modulation_bias = freq_modulation_bias

    def complex_activation(self, z):
        mag = torch.abs(z)
        mag_act = F.gelu(mag)
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def forward(self, query_heads, key_heads, value_heads, freq_weight, H, W, sin=None, cos=None):
        B = query_heads.shape[0]
        orig_dtype = query_heads.dtype

        if sin is not None and cos is not None:
            sin_flat = sin.view(H*W, -1)
            cos_flat = cos.view(H*W, -1)

            query_heads = theta_shift(query_heads, sin_flat.unsqueeze(0).unsqueeze(0), cos_flat.unsqueeze(0).unsqueeze(0))
            key_heads = theta_shift(key_heads, sin_flat.unsqueeze(0).unsqueeze(0), cos_flat.unsqueeze(0).unsqueeze(0))

        query_2d = query_heads.reshape(B, self.num_heads, H, W, self.head_dim)
        key_2d = key_heads.reshape(B, self.num_heads, H, W, self.head_dim)
        value_2d = value_heads.reshape(B, self.num_heads, H, W, self.value_head_dim)

        if orig_dtype == torch.float16 or orig_dtype == torch.bfloat16:
            query_2d = query_2d.float()
            key_2d = key_2d.float()
            value_2d = value_2d.float()

        query_fft = torch.fft.rfft2(query_2d, dim=(2, 3), norm='ortho')
        key_fft = torch.fft.rfft2(key_2d, dim=(2, 3), norm='ortho')
        value_fft = torch.fft.rfft2(value_2d, dim=(2, 3), norm='ortho')

        freq_mod_weight = self.freq_modulation_weight.unsqueeze(0).unsqueeze(-1) * freq_weight.view(B, self.num_heads, 1, 1, 1)
        freq_mod_bias = self.freq_modulation_bias.unsqueeze(0).unsqueeze(-1) * freq_weight.view(B, self.num_heads, 1, 1, 1)

        key_fft_mod = key_fft * freq_mod_weight + freq_mod_bias
        value_fft_mod = value_fft * freq_mod_weight + freq_mod_bias

        key_fft_act = self.complex_activation(key_fft_mod)
        value_fft_act = self.complex_activation(value_fft_mod)
        qk_fft = query_fft * torch.conj(key_fft_act)
        qk_freq_spatial = torch.fft.irfft2(qk_fft, s=(H, W), dim=(2, 3), norm='ortho')
        qk_freq_spatial = qk_freq_spatial.sum(dim=-1)
        qk_freq_spatial = qk_freq_spatial.reshape(B, self.num_heads, H*W)

        qk_freq_spatial = qk_freq_spatial * self.scaling

        attn_freq = F.softmax(qk_freq_spatial.unsqueeze(-1), dim=-2)

        value_freq_out = torch.fft.irfft2(value_fft_act, s=(H, W), dim=(2, 3), norm='ortho')
        value_freq_out = value_freq_out.reshape(B, self.num_heads, H*W, self.value_head_dim)
        out_freq = value_freq_out * attn_freq

        if out_freq.dtype != orig_dtype:
            out_freq = out_freq.to(orig_dtype)

        return out_freq


class SpectralSpatialAttention(nn.Module):
    def __init__(self, in_channels: int = 512, out_channels: int = 512,
                 ratio: int = 2, num_heads: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.key_channels = in_channels // ratio
        self.value_channels = in_channels // ratio
        self.head_dim = self.key_channels // num_heads
        self.value_head_dim = self.value_channels // num_heads
        self.scaling = self.head_dim ** -0.5

        self.InitialConvQ = nn.Conv2d(in_channels, self.key_channels, 1, bias=True)
        self.InitialConvK = nn.Conv2d(in_channels, self.key_channels, 1, bias=True)
        self.InitialConvV = nn.Conv2d(in_channels, self.value_channels, 1, bias=True)

        self.SpatialFeatureK = MultiScaleSpatialFeatureBlock(self.key_channels)
        self.SpatialFeatureV = MultiScaleSpatialFeatureBlock(self.value_channels)

        self.dct_attention = DCTAttention(self.value_channels, num_freqs=8)
        self.wfca = WFCA_Simple(self.value_channels, num_freqs=8)
        self.proj_low = nn.Conv2d(self.key_channels * 3, self.value_channels, 1, bias=False)
        self.proj_high = nn.Conv2d(self.key_channels * 6, self.value_channels, 1, bias=False)

        wt_filter, _ = create_2d_wavelet_filter('db1', self.key_channels, self.key_channels, torch.float)
        self.register_buffer('wt_filter', wt_filter)

        self.pool_weight = nn.Parameter(torch.tensor(0.1))
        self.freq_weight = nn.Parameter(torch.tensor(0.9))

        self.ReorganizeConv = nn.Conv2d(self.value_channels, out_channels, 1, bias=True)
        self.norm = nn.LayerNorm(self.value_channels)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        query = self.InitialConvQ(x)  # (B, key_channels, H, W)
        key = self.InitialConvK(x)
        value = self.InitialConvV(x)

        key_features = self.SpatialFeatureK(key)  # (B, L_extended, key_channels)
        value_features = self.SpatialFeatureV(value)  # (B, L_extended, value_channels)

        L_spatial = H * W
        key_spatial = key_features[:, :L_spatial, :]  # (B, HW, key_channels)
        value_spatial = value_features[:, :L_spatial, :]  # (B, HW, value_channels)

        # --- DWT frequency separation ---
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        Hp, Wp = H + pad_h, W + pad_w

        key_2d = key_spatial.permute(0, 2, 1).reshape(B, self.key_channels, H, W)
        value_2d = value_spatial.permute(0, 2, 1).reshape(B, self.value_channels, H, W)
        if pad_h or pad_w:
            key_2d = F.pad(key_2d, (0, pad_w, 0, pad_h))
            value_2d = F.pad(value_2d, (0, pad_w, 0, pad_h))

        # DWT: (B, C, 4, Hp/2, Wp/2) -> [LL, LH, HL, HH]
        key_dwt = wavelet_2d_transform(key_2d, self.wt_filter)
        value_dwt = wavelet_2d_transform(value_2d, self.wt_filter)

        _, _, _, Hd, Wd = key_dwt.shape

        # Low frequency: LL (index 0)
        key_ll = key_dwt[:, :, 0, :, :]  # (B, key_channels, Hd, Wd)
        value_ll = value_dwt[:, :, 0, :, :]

        # High frequency: LH+HL+HH (indices 1:4)
        key_h = key_dwt[:, :, 1:4, :, :].reshape(B, self.key_channels * 3, Hd, Wd)
        value_h = value_dwt[:, :, 1:4, :, :].reshape(B, self.value_channels * 3, Hd, Wd)

        # Downsample query to match DWT resolution
        q_down = F.adaptive_avg_pool2d(query, (Hd, Wd))

        # Low path: concat(key_ll, value_ll, q_down) → project → WFCA
        low_feat = torch.cat([key_ll, value_ll, q_down], dim=1)  # (B, 3*key_channels, H/2, W/2)
        low_feat = self.proj_low(low_feat)  # (B, value_channels, H/2, W/2)
        low_feat = self.wfca(low_feat)
        low_feat = F.interpolate(low_feat, size=(H, W), mode='bilinear', align_corners=False)
        out_spatial = low_feat.flatten(2).permute(0, 2, 1) * self.pool_weight

        # High path: concat(key_h, value_h) → project → DCTAttention
        high_feat = torch.cat([key_h, value_h], dim=1)  # (B, 6*key_channels, H/2, W/2)
        high_feat = self.proj_high(high_feat)  # (B, value_channels, H/2, W/2)
        high_feat = self.dct_attention(high_feat)
        high_feat = F.interpolate(high_feat, size=(H, W), mode='bilinear', align_corners=False)
        out_freq = high_feat.flatten(2).permute(0, 2, 1)

        output = self.freq_weight * out_freq + out_spatial

        output = output.permute(0, 2, 1).reshape(B, self.value_channels, H, W)
        output = self.ReorganizeConv(output)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class HSAModule(nn.Module):
    def __init__(self, in_channels: int = 512, out_channels: int = 512,
                 ratio: int = 2, num_heads: int = 4):
        super().__init__()

        self.spectral_attention = SpectralSpatialAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            ratio=ratio,
            num_heads=num_heads
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.spectral_attention(x)
        output = att + x
        return output
