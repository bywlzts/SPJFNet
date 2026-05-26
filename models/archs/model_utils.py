import functools
import torch
import torch.nn as nn
import models.archs.arch_util as arch_util
from models.archs.ffc import FFCResnetBlock
import torch.nn.functional as F
from models.archs.wtconv import WTConv2d
from models.loss import *
import cv2
import utils.util as util
import numpy as np

class LowFrequencyProcessing(nn.Module):
    def __init__(self, nf=64, num_blocks=6, input_channels=3):
        """
        Unified Stage --- Combines Fourier Reconstruction and Spatial-Texture Reconstruction.
        """
        super(LowFrequencyProcessing, self).__init__()

        # Initial feature extraction
        self.initial_conv = nn.Conv2d(input_channels, nf, kernel_size=1, stride=1, padding=0)
        # FFT-based feature extraction blocks (First Stage)
        self.fft_blocks = nn.ModuleList([FFT_Process(nf) for _ in range(6)])
        # Dual-branch blocks (Second Stage)
        self.ffc_blocks = nn.ModuleList([FFCResnetBlock(nf) for _ in range(num_blocks)])
        self.multi_blocks = nn.ModuleList([MultiConvBlock(nf) for _ in range(num_blocks)])
        self.fusion_block = ChannelAttentionFusion(nf)
        # Downsampling layers for feature fusion
        self.concat_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0),
                SEBlock(nf)
            ) for _ in range(3)
        ])
        # Reconstruction trunk
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, 1)

        # Upsample convolution layers
        self.upconv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, fr, y_map):
        # Initial feature extraction
        xori = x
        x0 = self.initial_conv(x)
        # FFT-based feature extraction with skip connections
        x, fr, y_map = self.fft_blocks[0](x0, fr, y_map)
        x1, fr, y_map = self.fft_blocks[1](x, fr, y_map)
        x2, fr, y_map = self.fft_blocks[2](x1, fr, y_map)

        # Downsample and fuse features
        x3_input = torch.cat((x2, x1), dim=1)
        x3_input = self.concat_layers[0](x3_input)
        x3, fr, y_map = self.fft_blocks[3](x3_input, fr, y_map)

        x4_input = torch.cat((x3, x), dim=1)
        x4_input = self.concat_layers[1](x4_input)
        x4, fr, y_map = self.fft_blocks[4](x4_input, fr, y_map)

        x5_input = torch.cat((x4, x0), dim=1)
        x5_input = self.concat_layers[2](x5_input)
        x5, fr, y_map = self.fft_blocks[5](x5_input, fr, y_map)

        fft_features = x5
        multi_features = x5
        for ffc_block, multi_block in zip(self.ffc_blocks, self.multi_blocks):
            fft_features = ffc_block(fft_features)
            multi_features = multi_block(multi_features)
            # Fuse features using Channel Attention Fusion
        fused_features = self.fusion_block(fft_features, multi_features)
        out_noise = self.upconv_last(fused_features) + xori
        return out_noise

class FFT_Process(nn.Module):
    def __init__(self, nf):
        super(FFT_Process, self).__init__()
        # Preprocessing for frequency domain
        self.nf = nf
        self.freq_preprocess = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        self.feature_fusion = nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0)
        self.process_amp = self._make_process_block(nf)
        self.process_pha = self._make_process_block(nf)
        self.process_fr = self._make_process_block(nf)
        self.process_map = self._make_process_block(nf)
        self.process_sigmoid_amp = FrequencyFusion(nf)
        self.process_amp_post = self._make_process_block(nf)
        self.process_pha_post = self._make_process_block_pha(nf)

    def _make_process_block(self, nf):
        return nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        )

    def _make_process_block_pha(self, nf):
        return nn.Sequential(
            nn.Conv2d(nf*2, nf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, fr, y_map):
        _, _, H, W = x.shape
        # Frequency domain processing
        x_freq = torch.fft.rfft2(self.freq_preprocess(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process_amp(mag)
        pha = self.process_pha(pha)

        # Process infrared features and Cross-modality interaction
        fr = self.process_fr(fr)
        pha = torch.cat([pha, fr],dim=1)

        # Process brightness attention map
        mag = self.process_sigmoid_amp(mag, y_map)
        pha = self.process_pha_post(pha)
        mag = self.process_amp_post(mag)

        # Reconstruct frequency domain features
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out_ff = x_out + x
        return x_out_ff, fr, y_map

class DownFRG(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = arch_util.DWT()  # 小波下采样

    def forward(self, x):
        # 使用小波变换进行下采样
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)
        return x_LL, (x_HL, x_LH, x_HH)

class UpFRG(nn.Module):
    def __init__(self):
        super().__init__()
        self.iwt = arch_util.IWT()  # 小波上采样

    def forward(self, x_LL, x_H):
        # 使用逆小波变换进行上采样
        x_HL, x_LH, x_HH = x_H
        x = self.iwt(torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1))
        return x

class DeformableDilatedConv(nn.Module):
    """可变形卷积 + 空洞卷积模块 + 小波卷积 + 残差"""
    def __init__(self, in_channels, out_channels, dilation=2):
        super(DeformableDilatedConv, self).__init__()
        self.dilated_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            dilation=2, padding=2
        )
        self.dilated_conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=4, padding=4)

        self.wtconv = WTConv2d(in_channels, out_channels, kernel_size=5, wt_levels=3)
        self.relu = nn.ReLU()
        self.fuseconv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x1 = self.relu(self.dilated_conv(x))
        x2 = self.relu(self.dilated_conv3(x))
        x3 = self.wtconv(x)
        x = self.fuseconv(torch.cat([x1,x2,x3],dim=1))
        return x + residual

class SpatialAugmentModel(nn.Module):
    def __init__(self, nf=16):
        super(SpatialAugmentModel, self).__init__()
        # 输入卷积：3 通道 -> nf 通道
        self.input_conv = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=True)
        # 两个 WTConv 模块
        self.spatialaugment1 = DeformableDilatedConv(in_channels=nf, out_channels=nf, dilation=2)
        self.spatialaugment2 = DeformableDilatedConv(in_channels=nf, out_channels=nf, dilation=2)
        # 输出卷积：nf 通道 -> 3 通道
        self.output_conv = nn.Conv2d(nf, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # 输入卷积
        res = x
        x = self.input_conv(x)
        x = self.spatialaugment1(x)
        x = self.spatialaugment2(x)
        # 输出卷积
        x = self.output_conv(x)
        return x + res

class WTConvAttentionModel(nn.Module):
    def __init__(self, nf=16):
        super(WTConvAttentionModel, self).__init__()
        # 输入卷积：3 通道 -> nf 通道
        self.input_conv = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=True)
        # 两个 WTConv 模块
        self.wtconv1 = WTConv2d(nf, nf, kernel_size=5, wt_levels=3)
        self.wtconv2 = WTConv2d(nf, nf, kernel_size=5, wt_levels=3)
        self.output_conv = nn.Conv2d(nf, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # 输入卷积
        res = x
        x = self.input_conv(x)
        # 两个 WTConv 操作
        x = self.wtconv1(x)
        x = self.wtconv2(x)
        x = self.output_conv(x)
        return x + res


class MultiConvBlock(nn.Module):
    def __init__(self, dim, num_heads=4, expand_ratio=2):
        super(MultiConvBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Channel reduction layer
        self.conv_reduction = nn.Conv2d(dim, dim // 4, kernel_size=1, stride=1, bias=True)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

        # Multi-scale convolution layers
        self.local_convs = nn.ModuleList([
            nn.Conv2d(
                dim // 4, dim // 4,
                kernel_size=(3 + i * 2),
                padding=(1 + i),
                stride=1,
                groups=dim // 4  # Grouped convolution
            ) for i in range(num_heads)
        ])

        # Feature fusion layer
        self.conv_fusion = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.se_block = SEBlock(dim)

    def forward(self, x):
        # Channel reduction
        x_reduced = self.leakyrelu(self.conv_reduction(x))

        # Multi-scale feature extraction
        multi_scale_features = []
        for conv in self.local_convs:
            x_scale = self.leakyrelu(conv(x_reduced))  # Apply multi-scale convolution
            x_scale = x_scale * torch.sigmoid(x_reduced)  # Element-wise modulation
            multi_scale_features.append(x_scale)

        # Concatenate multi-scale features
        x_concat = torch.cat(multi_scale_features, dim=1)

        # Feature fusion and residual connection
        x_fused = self.conv_fusion(x_concat)
        x_fused = self.se_block(x_fused)
        return x + x_fused  # Residual connection


class FrequencyFusion(nn.Module):
    def __init__(self, channels):
        super(FrequencyFusion, self).__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )

        # 最终调制
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, D, G):
        # 拼接特征
        cat_feature = torch.cat([D, G], dim=1)
        # 通道注意力权重
        channel_weight = self.channel_attention(cat_feature)
        # 加权特征
        weighted_D = D * channel_weight
        # 特征融合
        fused_feature = self.fusion_conv(cat_feature)
        # 生成门控权重
        gate_weight = self.gate(fused_feature)
        # 最终输出
        output = weighted_D + G * gate_weight

        return output

class ChannelAttentionFusion(nn.Module):
    def __init__(self, nf):
        """
        Channel Attention Fusion module for combining fft_features and multi_features.

        Args:
            nf (int): Number of feature channels.
        """
        super(ChannelAttentionFusion, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc = nn.Sequential(
            nn.Conv2d(nf * 2, nf // 4, 1, bias=False),  # Reduce channels
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // 4, nf * 2, 1, bias=False),  # Restore channels
            nn.Sigmoid()
        )

    def forward(self, fft_features, multi_features):
        # Concatenate features along the channel dimension
        combined_features = torch.cat([fft_features, multi_features], dim=1)

        # Generate attention weights
        attention_weights = self.fc(self.global_avg_pool(combined_features))

        # Split attention weights for fft_features and multi_features
        fft_weight, multi_weight = torch.split(attention_weights, fft_features.size(1), dim=1)

        # Apply attention weights
        fused_features = fft_weight * fft_features + multi_weight * multi_features
        return fused_features


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale + x
