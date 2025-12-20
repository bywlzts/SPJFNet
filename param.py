import torch
from torchvision.models import resnet18
from thop import profile
from models.archs.SPJFNet import SPJFNet
model = SPJFNet(y_nf=8, f_nf=8, s_nf=16)
input = torch.randn(1, 3, 256, 256)
input2 = torch.randn(1, 3, 256, 256)
input3 = torch.randn(1, 1, 256, 256)
input4 = torch.randn(1, 1, 256, 56)
flops, params = profile(model, inputs=(input,))
print('flops:{}'.format(flops))
print('params:{}'.format(params))
