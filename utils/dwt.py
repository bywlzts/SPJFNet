import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))

    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

def plot_wavelet_components(ll, hl, lh, hh, title):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(ll.squeeze().cpu().numpy(), cmap='gray')
    axs[0].set_title(f'{title} - LL')
    axs[1].imshow(hl.squeeze().cpu().numpy(), cmap='gray')
    axs[1].set_title(f'{title} - HL')
    axs[2].imshow(lh.squeeze().cpu().numpy(), cmap='gray')
    axs[2].set_title(f'{title} - LH')
    axs[3].imshow(hh.squeeze().cpu().numpy(), cmap='gray')
    axs[3].set_title(f'{title} - HH')
    for ax in axs:
        ax.axis('off')
    plt.show()

def apply_four_level_dwt(image_tensor):
    dwt = DWT()

    # 进行四次小波变换
    ll1, hl1, lh1, hh1 = dwt(image_tensor)
    print(ll1.size())
    ll2, hl2, lh2, hh2 = dwt(ll1)
    print(ll2.size())
    ll3, hl3, lh3, hh3 = dwt(ll2)
    print(ll3.size())
    ll4, hl4, lh4, hh4 = dwt(ll3)
    print(ll4.size())

    # 可视化每层变换后的高低频分量
    plot_wavelet_components(ll1, hl1, lh1, hh1, 'Level 1')
    plot_wavelet_components(ll2, hl2, lh2, hh2, 'Level 2')
    plot_wavelet_components(ll3, hl3, lh3, hh3, 'Level 3')
    plot_wavelet_components(ll4, hl4, lh4, hh4, 'Level 4')

# 示例使用
# 假设我们有一张单通道输入图像（1, 1, H, W），如下所示
image_array = np.random.rand(1, 1, 64, 64).astype(np.float32) # 使用随机生成图像作为示例，高度和宽度为64
image_tensor = torch.tensor(image_array)

# 进行四次小波变换并可视化结果
apply_four_level_dwt(image_tensor)


