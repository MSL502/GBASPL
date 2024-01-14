import torch.nn as nn
import torch
import torchvision.utils as vutils
from models.layers import *

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class Grad_Group(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Grad_Group, self).__init__()
        modules = [Block(conv, dim * 2, kernel_size)]
        modules.append(conv(dim * 2, dim * 2, kernel_size))
        self.gp = nn.Sequential(*modules)
        # change channel number
        self.conv_block = conv(dim * 2, dim, kernel_size)

    def forward(self, x):
        res = self.gp(x)
        res += x
        res = self.conv_block(res)
        return res

class V_Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(V_Group, self).__init__()
        modules = [Block(conv, dim * 2, kernel_size)]
        modules.append(conv(dim * 2 , dim * 2, kernel_size))
        self.gp = nn.Sequential(*modules)
        # change channel number
        self.conv_block = conv(dim * 2, dim, kernel_size)

    def forward(self, x):
        res = self.gp(x)
        res += x
        res = self.conv_block(res)
        return res


class Fusion_Group(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Fusion_Group, self).__init__()
        modules = [Block(conv, dim * 3, kernel_size)]
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class Network(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(Network, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim * 3, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

        self.get_g_nopadding = Get_gradient_nopadding()
        self.grad_g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.grad_g2 = Grad_Group(conv, self.dim, kernel_size)
        self.grad_g3 = Grad_Group(conv, self.dim, kernel_size)
        self.grad_ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * 2, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * 2, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.v_pre_process = conv(1, self.dim, kernel_size)
        self.v_g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.v_g2 = V_Group(conv, self.dim, kernel_size, blocks=blocks)
        self.v_g3 = V_Group(conv, self.dim, kernel_size, blocks=blocks)
        self.v_ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * 2, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * 2, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.fusion_block = Fusion_Group(conv, self.dim, kernel_size)

        self.output_grad = conv(self.dim, 3, kernel_size=1)
        self.v_out = conv(self.dim, 1, kernel_size)

        self.gamma_max = 0.1
        self.gamma_min = 0.05
        self.grad_gamma_max = 0.03
        self.grad_gamma_min = 0.01
        self.grad_default_conv = default_conv(3, self.dim, kernel_size)
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self, x1, m_v_getatt=None, m_grad_getatt=None, l_v_mask=0.00001, l_grad_mask=0.00001):
        # print(x1.shape)
        x = self.pre(x1)

        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        print(w.shape)
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        w = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        enhanced_out = self.palayer(w)

        # grid
        # gramma = 0.5
        x_gamma = x1.pow(1 / 2.0)
        x_grad = self.get_g_nopadding(x_gamma)
        x_grad_pre = self.pre(x_grad)

        x_grad_res1 = self.grad_g1(x_grad_pre)
        x_grad_res2 = self.grad_g2(torch.cat([x_grad_res1, res1], dim=1))
        x_grad_res3 = self.grad_g3(torch.cat([x_grad_res2, res2], dim=1))

        x_grad_w = self.grad_ca(torch.cat([x_grad_res3, res3], dim=1))
        x_grad_w = x_grad_w.view(-1, 2, self.dim)[:, :, :, None, None]
        grad_w = x_grad_w[:, 0, ::] * x_grad_res3 + x_grad_w[:, 1, ::] * res3
        grad_fea = self.palayer(grad_w)
        grad_image = self.output_grad(grad_fea)


        if l_grad_mask >= self.grad_gamma_max:
            alpha = 1.0
        elif l_grad_mask >= self.grad_gamma_min:
            alpha = (l_grad_mask - self.grad_gamma_min) / (self.grad_gamma_max - self.grad_gamma_min)
        else:
            alpha = 0.0

        if m_grad_getatt is None:
            grad_fusion_fea = grad_fea
        else:
            m_grad_getatt = self.pre(m_grad_getatt)
            grad_fusion_fea = alpha * m_grad_getatt + (1 - alpha) * grad_fea

        x_v = x.max(1, keepdim=True)[0]
        x_v_pre = self.v_pre_process(x_v)
        x_v_res1 = self.v_g1(x_v_pre)
        x_v_res2 = self.v_g2(torch.cat([x_v_res1, res1], dim=1))
        x_v_res3 = self.v_g3(torch.cat([x_v_res2, res2], dim=1))
        x_v_w = self.v_ca(torch.cat([x_v_res3, res3], dim=1))
        x_v_w = x_v_w.view(-1, 2, self.dim)[:, :, :, None, None]
        v_w = x_v_w[:, 0, ::] * x_v_res3 + x_v_w[:, 1, ::] * res3
        v_fea = self.palayer(v_w)
        v_img = self.v_out(v_fea)

        if l_v_mask >= self.gamma_max:
            beta = 1.0
        elif l_v_mask >= self.gamma_min:
            beta = (l_v_mask - self.gamma_min) / (self.gamma_max - self.gamma_min)
        else:
            beta = 0.0
        if m_v_getatt is None:
            v_fusion_fea = v_fea
        else:
            v_fusion_fea = beta * m_v_getatt + (1 - beta) * v_fea

        fusion_fea = self.fusion_block(torch.cat([grad_fusion_fea, v_fusion_fea, enhanced_out],dim=1))

        x = self.post(fusion_fea)
        enhanced_img = x + x1
        return enhanced_img, grad_image, v_img