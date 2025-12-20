import torch

def normalize_wavelet_coeffs_tensor(coeffs):
    """归一化小波系数并记录值域 (Tensor 格式)"""
    normalized_coeffs = []
    ranges = []
    for coeff in coeffs:
        if isinstance(coeff, tuple):  # 高频分量 (LH, HL, HH)
            normalized_subbands = []
            subband_ranges = []
            for subband in coeff:  # subband 是 bchw 格式的 Tensor
                c_min = subband.amin(dim=(2, 3), keepdim=True)  # 最小值 (b, c, 1, 1)
                c_max = subband.amax(dim=(2, 3), keepdim=True)  # 最大值 (b, c, 1, 1)
                subband_ranges.append((c_min, c_max))
                normalized_subbands.append((subband - c_min) / (c_max - c_min + 1e-8))  # 归一化到 [0, 1]
            normalized_coeffs.append(tuple(normalized_subbands))
            ranges.append(tuple(subband_ranges))
        else:  # 低频分量 (LL)
            c_min = coeff.amin(dim=(2, 3), keepdim=True)  # 最小值 (b, c, 1, 1)
            c_max = coeff.amax(dim=(2, 3), keepdim=True)  # 最大值 (b, c, 1, 1)
            ranges.append((c_min, c_max))
            normalized_coeffs.append((coeff - c_min) / (c_max - c_min + 1e-8))  # 归一化到 [0, 1]
    return normalized_coeffs, ranges

def normalize_wavelet_ll(ll):
    c_min = ll.amin(dim=(2, 3), keepdim=True)
    c_max = ll.amax(dim=(2, 3), keepdim=True)
    # 归一化
    normalized_ll = (ll - c_min) / (c_max - c_min + 1e-8)
    # 返回归一化结果和值域范围
    return normalized_ll, (c_min, c_max)


def denormalize_wavelet_ll(normalized_ll, ranges):
    c_min, c_max = ranges  # 解包最小值和最大值
    # 反归一化
    ll = normalized_ll * (c_max - c_min + 1e-8) + c_min
    # 返回反归一化结果
    return ll

def denormalize_wavelet_coeffs_tensor(normalized_coeffs, ranges):
    """反归一化小波系数 (Tensor 格式)"""
    denormalized_coeffs = []
    for coeff, coeff_range in zip(normalized_coeffs, ranges):
        if isinstance(coeff, tuple):  # 高频分量 (LH, HL, HH)
            denormalized_subbands = []
            for subband, (c_min, c_max) in zip(coeff, coeff_range):
                denormalized_subbands.append(subband * (c_max - c_min + 1e-8) + c_min)  # 反归一化
            denormalized_coeffs.append(tuple(denormalized_subbands))
        else:  # 低频分量 (LL)
            c_min, c_max = coeff_range
            denormalized_coeffs.append(coeff * (c_max - c_min + 1e-8) + c_min)  # 反归一化
    return denormalized_coeffs


def multi_scale_loss(original_coeffs, reconstructed_coeffs, ranges, alpha=1.0, beta=1.0):
    """
    计算多尺度损失，包括高低频分量的损失
    :param original_coeffs: 原始小波分量 (归一化前)
    :param reconstructed_coeffs: 重建的小波分量 (归一化前)
    :param ranges: 原始小波分量的值域
    :param alpha: 低频分量的损失权重
    :param beta: 高频分量的损失权重
    :return: 总损失
    """
    loss = 0.0
    for orig, recon, coeff_range in zip(original_coeffs, reconstructed_coeffs, ranges):
        if isinstance(orig, tuple):  # 高频分量 (LH, HL, HH)
            for o, r, (c_min, c_max) in zip(orig, recon, coeff_range):
                # 归一化后计算损失
                o_norm = (o - c_min) / (c_max - c_min + 1e-8)
                r_norm = (r - c_min) / (c_max - c_min + 1e-8)
                loss += beta * torch.nn.functional.mse_loss(o_norm, r_norm)
        else:  # 低频分量 (LL)
            c_min, c_max = coeff_range
            o_norm = (orig - c_min) / (c_max - c_min + 1e-8)
            r_norm = (recon - c_min) / (c_max - c_min + 1e-8)
            loss += alpha * torch.nn.functional.mse_loss(o_norm, r_norm)
    return loss