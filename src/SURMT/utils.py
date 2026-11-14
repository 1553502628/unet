import pywt
import torch

from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


def create_1d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo, dec_hi], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1)

    rec_hi = torch.tensor(w.rec_hi, dtype=type)
    rec_lo = torch.tensor(w.rec_lo, dtype=type)
    rec_filters = torch.stack([rec_lo, rec_hi], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1)

    return dec_filters, rec_filters


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


def wavelet_1d_transform(x, filters):
    b, c, len = x.shape
    pad = (filters.shape[2] // 2 - 1)
    x = F.conv1d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 2, len // 2)
    return x


def inverse_1d_wavelet_transform(x, filters):
    b, c, _, l_half = x.shape
    pad = (filters.shape[2] // 2 - 1)
    x = x.reshape(b, c * 2, l_half)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad)
    return x


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

        self.base_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels*4,
                    out_channels=in_channels*4,
                    kernel_size=kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False
                )
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels*4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

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


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConvBlock(nn.Module):
    def __init__(self, channels, wt_levels=1, wt_type='db1'):
        super(WTConvBlock, self).__init__()
        self.wtconv1 = WTConv2d(
            channels, channels,
            kernel_size=5, stride=1, bias=False, wt_levels=wt_levels, wt_type=wt_type
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.wtconv2 = WTConv2d(
            channels, channels,
            kernel_size=5, stride=1, bias=False, wt_levels=wt_levels, wt_type=wt_type
        )
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


class FourierBlock(nn.Module):
    def __init__(self, channels, spatial_shape=(224, 224), mlp_ratio=0.5, dropout=0.0):
        super().__init__()
        self.dim = channels
        self.h, self.w = spatial_shape
        self.mlp_dim = int(channels * mlp_ratio)

        self.complex_weight = nn.Parameter(
            torch.randn(channels, self.h, self.w//2 + 1, 2, dtype=torch.float32) * 0.02
        )

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.fc1 = nn.Conv2d(channels, self.mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(self.mlp_dim, channels, 1)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        residual = x

        # LayerNorm: (B, C, H, W) -> (B, H*W, C) -> norm -> (B, C, H, W)
        x_flat = x.flatten(2).transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)
        x_norm = self.norm1(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(B, C, H, W)
        x_fft = self._fft_forward(x_norm)

        x_fft_flat = x_fft.flatten(2).transpose(1, 2)
        x_fft_norm = self.norm2(x_fft_flat)
        x_fft_norm = x_fft_norm.transpose(1, 2).reshape(B, C, H, W)

        x_mlp = self._mlp_forward(x_fft_norm)

        out = residual + self.drop(x_mlp)

        return out

    def _fft_forward(self, x):
        x = x.to(torch.float32)

        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')

        if H != self.h or W // 2 + 1 != self.w // 2 + 1:
            w = self.complex_weight.permute(3, 0, 1, 2).reshape(1, 2 * C, self.h, self.w // 2 + 1)
            w = F.interpolate(w, size=(H, W // 2 + 1), mode='bilinear', align_corners=False)
            w = w.reshape(2, C, H, W // 2 + 1).permute(1, 2, 3, 0).contiguous()
            weight = torch.view_as_complex(w)
        else:
            weight = torch.view_as_complex(self.complex_weight)
        x_freq = x_freq * weight

        x_spatial = torch.fft.irfft2(x_freq, s=(x.shape[2], x.shape[3]), dim=(2, 3), norm='ortho')

        return x_spatial

    def _mlp_forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MultiScaleSpatialFeatureBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleSpatialFeatureBlock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.dilate1 = nn.Conv2d(in_channels, in_channels, 3, dilation=1, padding=1, bias=True)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, 3, dilation=3, padding=3, bias=True)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, 3, dilation=5, padding=5, bias=True)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1, dilation=1, padding=0, bias=True)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # Pooling features
        pool_1 = self.pool1(x).view(b, c, -1)
        pool_2 = self.pool2(x).view(b, c, -1)
        pool_3 = self.pool3(x).view(b, c, -1)
        pool_4 = self.pool4(x).view(b, c, -1)
        poolingfeatures = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)

        # Dilated convolutions
        dilate1_out = F.relu(self.dilate1(x), inplace=True)
        dilate2_out = F.relu(self.conv1x1(self.dilate2(x)), inplace=True)
        dilate3_out = F.relu(self.conv1x1(self.dilate2(self.dilate1(x))), inplace=True)
        dilate4_out = F.relu(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))), inplace=True)
        convfeatures = dilate1_out + dilate2_out + dilate3_out + dilate4_out

        convfeatures = convfeatures.view(b, c, -1)
        out = torch.cat([convfeatures, poolingfeatures], -1)
        out = out.permute(0, 2, 1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ParallelWienerBranch(nn.Module):
    """Learnable Wiener filter branch in frequency domain"""
    def __init__(self, channels, init_K=0.01):
        super().__init__()
        self.channels = channels
        self.K = nn.Parameter(torch.tensor(init_K))
        self.freq_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        dtype = x.dtype
        if dtype == torch.float16 or dtype == torch.bfloat16:
            x = x.float()

        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        power = torch.abs(x_fft) ** 2

        Hw = power / (power + self.K ** 2 + 1e-8)
        x_fft = x_fft * Hw * self.freq_scale

        x = torch.fft.irfft2(x_fft, s=(H, W), dim=(2, 3), norm='ortho')
        if x.dtype != dtype:
            x = x.to(dtype)
        return self.conv(x)


class ParallelWienerFusion(nn.Module):
    """2路并行：低频 + 高频，去掉中频"""
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        self._skip_proj = None

        self.branch_low = ParallelWienerBranch(channels, init_K=0.003)
        self.branch_high = ParallelWienerBranch(channels, init_K=0.1)

        self.weight_net = nn.Sequential(
            nn.Linear(channels * 2, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 2),
        )

        self.reconstruct = nn.Conv2d(channels, channels, 3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, skip, up):
        in_ch = skip.size(1)
        if in_ch != self.channels:
            if self._skip_proj is None or self._skip_proj.in_channels != in_ch:
                self._skip_proj = nn.Conv2d(in_ch, self.channels, 1).to(skip.device, skip.dtype)
            skip = self._skip_proj(skip)
        b_low = self.branch_low(skip)
        b_high = self.branch_high(skip)

        g_low = b_low.mean(dim=(2, 3))
        g_high = b_high.mean(dim=(2, 3))
        w = self.weight_net(torch.cat([g_low, g_high], dim=1))
        w = F.softmax(w, dim=1)
        w = w.unsqueeze(-1).unsqueeze(-1)

        fused = w[:, 0:1] * b_low + w[:, 1:2] * b_high
        out = self.reconstruct(fused)
        if up.size(1) != self.channels:
            up = up.mean(dim=1, keepdim=True).expand(-1, self.channels, -1, -1)
        return self.alpha * out + (1 - self.alpha) * up


class GLCAttention(nn.Module):
    """Global-Local Collaborative Attention: 全局大核 + 局部自注意力 + 门控融合"""
    def __init__(self, channels, g_kernel=31, l_kernel=3):
        super().__init__()
        self.global_branch = nn.Sequential(
            nn.Conv2d(channels, channels, g_kernel, padding=g_kernel//2, groups=channels),
            nn.Sigmoid()
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, l_kernel, padding=l_kernel//2, groups=channels),
            nn.Sigmoid()
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        g = self.global_branch(x)
        l = self.local_branch(x)
        w = self.gate(x)
        return w[:, 0:1] * g + w[:, 1:2] * l + x


class DecoderStage(nn.Module):
    """简单解码阶段：上采样 → GLCAttention → 残差"""
    def __init__(self, in_channels, out_channels, skip_channels=None):
        super().__init__()
        if skip_channels is None:
            skip_channels = out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.attn = GLCAttention(out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.fusion(x)
        x = self.attn(x)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.ConvTranspose2d(in_channels, in_channels//2, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.GELU(),
            nn.Conv2d(in_channels//2, in_channels//2, 3, 1, 1),
            nn.BatchNorm2d(in_channels//2),
            nn.GELU(),

            nn.ConvTranspose2d(in_channels//2, in_channels//4, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.GELU(),
            nn.Conv2d(in_channels//4, out_channels, 3, 1, 1)
        )

    def forward(self, d1, patch_features_enhanced):
        fused_features = d1 + patch_features_enhanced

        output = self.fusion_conv(fused_features)

        return output
