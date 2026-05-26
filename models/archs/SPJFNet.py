from models.archs.model_utils import *
from models.archs.wavelet_normlize import *

class SPJFNet(nn.Module):
    def __init__(self, y_nf=16, f_nf=16, s_nf=32):
        super(SPJFNet, self).__init__()

        # Model parameters
        self.y_nf = y_nf
        self.f_nf = f_nf
        self.s_nf = s_nf

        # Fourier stage processing
        self.fourier_process1 = DFGFLow(nf=self.s_nf, numblocks=2)
        self.fourier_process2 = DFGFLow(nf=self.s_nf, numblocks=4)
        self.fourier_process3 = DFGFLow(nf=self.s_nf, numblocks=6)

        # Down-sampling and up-sampling groups
        self.down_group1 = DownFRG()
        self.down_group2 = DownFRG()
        self.down_group3 = DownFRG()

        self.up_group3 = UpFRG()
        self.up_group2 = UpFRG()
        self.up_group1 = UpFRG()

        # Wavelet transform
        self.dwt = arch_util.DWT()  # Discrete Wavelet Transform

        # High-frequency processing
        self.high_process3 = DFGFHigh(in_nf=3, out_if=3, nf=self.s_nf)
        self.high_process2 = DFGFHigh(in_nf=3, out_if=3, nf=self.s_nf)
        self.high_process1 = DFGFHigh(in_nf=3, out_if=3, nf=self.s_nf)

        # Post-processing with attention
        self.postprocess3 = WTConvAttentionModel(nf=self.s_nf)
        self.postprocess2 = WTConvAttentionModel(nf=self.s_nf)
        self.postprocess1 = SpatialAugmentModel(nf=self.s_nf)

        # Gamma correction network
        self.gammanet = GammaNet(input_channels=3, feature_channels=self.s_nf)

        # Activation function
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def pad_to_multiple(self, x, multiple=8):
        _, _, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def forward(self, x, gt=None):
        outputs = {}

        # --- Input Processing ---
        x_LL1, x_H1 = self.down_group1(x)
        x_LL2, x_H2 = self.down_group2(x_LL1)
        x_LL3, x_H3 = self.down_group3(x_LL2)

        gamma = self.gammanet(x_LL3)
        x_LL3_gamma = torch.pow(x_LL3, gamma)
        gradient3 = arch_util.compute_gradient(x_LL3_gamma)

        # 记录中间结果用于 Loss 计算
        outputs['x_LL3_gamma'] = x_LL3_gamma
        outputs['gradient3'] = gradient3

        # --- GT Processing (仅在训练且提供gt时执行) ---
        if gt is not None:
            x_LLgt1, x_HGT1 = self.down_group1(gt)
            x_LLgt2, x_HGT2 = self.down_group2(x_LLgt1)
            x_LLgt3, x_HGT3 = self.down_group3(x_LLgt2)
            gradient3_gt = arch_util.compute_gradient(x_LLgt3)

            # 将 GT 相关特征存入字典
            outputs.update({
                'x_LLgt3': x_LLgt3, 'x_LLgt2': x_LLgt2, 'x_LLgt1': x_LLgt1,
                'x_HGT3': x_HGT3, 'x_HGT2': x_HGT2, 'x_HGT1': x_HGT1,
                'gradient3_gt': gradient3_gt
            })

        # --- Stage 3 ---
        x_LL3, _, _ = self.fourier_process3(x_LL3, gradient3, x_LL3_gamma)
        x_H3 = self.high_process1(x_H3, gradient3)
        inverted_image2 = self.up_group3(x_LL3_gamma, x_H3)
        outputs['x_H3'] = x_H3  # 记录高频特征

        # --- Stage 2 ---
        x_l2 = self.up_group3(x_LL3, x_H3)
        x_l2 = self.postprocess3(x_l2)
        outputs['x_l2'] = x_l2  # 记录中间输出

        gradient2 = arch_util.compute_gradient(x_l2)
        x_l2_four, _, _ = self.fourier_process2(x_l2, gradient2, inverted_image2)
        x_H2 = self.high_process2(x_H2, gradient2)
        inverted_image1 = self.up_group3(inverted_image2, x_H2)
        outputs['x_H2'] = x_H2

        # --- Stage 1 ---
        x_l1 = self.up_group2(x_l2_four, x_H2)
        x_l1 = self.postprocess2(x_l1)
        outputs['x_l1'] = x_l1

        gradient1 = arch_util.compute_gradient(x_l1)
        x_l1_four, _, _ = self.fourier_process1(x_l1, gradient1, inverted_image1)
        x_H1 = self.high_process3(x_H1, gradient1)
        outputs['x_H1'] = x_H1

        # --- Final ---
        x_four = self.up_group1(x_l1_four, x_H1)
        x_four = self.pad_to_multiple(x_four, multiple=8)
        out_noise = self.postprocess1(x_four)

        outputs['out'] = out_noise  # 最终输出

        return outputs

class GammaNet(nn.Module):
    def __init__(self, input_channels=3, feature_channels=16):
        super(GammaNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channels, feature_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.gamma_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels * 2, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        gamma = self.gamma_pred(features)
        gamma = gamma.view(-1, 1, 1, 1)
        return gamma

class DFGFLow(nn.Module):
    def __init__(self, nf=16, numblocks = 6):
        super(DFGFLow, self).__init__()
        self.s_nf = nf
        self.processblock = LowFrequencyProcessing(nf=self.s_nf, num_blocks=numblocks, input_channels=3)
        # Initial convolution layers for Fourier features
        self.conv_first_fr = nn.Conv2d(3, self.s_nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_first_map = nn.Conv2d(3, self.s_nf, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, gradient, x_light):
        x_light_fre = torch.fft.rfft2(x_light, norm='backward')
        x_light_mag = torch.abs(x_light_fre)
        x_light_pha = torch.angle(x_light_fre)
        x_light_mag = self.conv_first_fr(x_light_mag)
        x_light_pha = self.conv_first_map(x_light_pha)

        # 幅值增强
        x_amplitude = self.processblock(x, x_light_pha, x_light_mag)
        return x_amplitude, x, x_light_pha

class DFGFHigh(nn.Module):
    def __init__(self, in_nf=3, out_if=3, nf=16):
        super(DFGFHigh,self).__init__()

        self.conv1_first = nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2_first = nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(nf,nf,3,1,1,groups=4),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1,groups=4),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1,groups=4),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
        )

        self.convf1_first = nn.Sequential(
            nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            Depth_conv(in_ch=nf, out_ch=nf)
        )
        self.convf2_first = nn.Sequential(
            nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            Depth_conv(in_ch=nf, out_ch=nf)
        )
        self.convf3_first = nn.Sequential(
            nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            Depth_conv(in_ch=nf, out_ch=nf)
        )


        self.convf1 = nn.Sequential(
            Depth_conv(in_ch=nf, out_ch=nf),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
            nn.Conv2d(nf, nf, 3, 1, 1, groups=4)
        )
        self.convf2 = nn.Sequential(
            Depth_conv(in_ch=nf, out_ch=nf),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
            nn.Conv2d(nf, nf, 3, 1, 1, groups=4)
        )

        self.convf3 = nn.Sequential(
            Depth_conv(in_ch=nf, out_ch=nf),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
            nn.Conv2d(nf, nf, 3, 1, 1, groups=4)
        )


        self.conv1_out = nn.Conv2d(nf, out_if, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2_out = nn.Conv2d(nf, out_if, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3_out = nn.Conv2d(nf, out_if, kernel_size=1, stride=1, padding=0, bias=True)


        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, f, fg):
        f1, f2, f3 = f
        f_add = f1+f2+f3
        fg = self.conv1_first(fg)
        f_add = self.conv2_first(f_add)
        attention1 = self.sigm(self.conv1(fg))
        attention2 = self.sigm(self.conv2(f_add))
        attention = fg*attention1 + f_add*attention2
        attention = self.sigm(self.conv3(attention))

        f1 = self.convf1_first(f1)
        f1 = f1 + attention * f1
        f1 = self.conv1_out(self.convf1(f1))

        f2 = self.convf1_first(f2)
        f2 = f2 + attention * f2
        f2 = self.conv2_out(self.convf2(f2))

        f3 = self.convf1_first(f3)
        f3 = f3 + attention * f3
        f3 = self.conv3_out(self.convf3(f3))

        return (f1, f2, f3)