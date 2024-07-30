import torch
from Utils.utils import *
from Utils.option import *
import torch.nn as nn

def CircleFilter(H, W, r, alpha):
    """
        --image size (H,W)
        --r : radius
    """
    x_center = int(W // 2)
    y_center = int(H // 2)
    X, Y = torch.meshgrid(torch.arange(0, H, 1), torch.arange(0, W, 1))
    circle = torch.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2)

    lp_F = (circle < r).clone().to(torch.float32)
    hp_F = (circle > r).clone().to(torch.float32)

    combined_Filter = alpha * lp_F + 0 * hp_F # (H, W)
    combined_Filter[~(circle < r) & ~(circle > r)] = 1 / 2  # cutoff

    return combined_Filter


def getHighLowFre(image):
    img_fft = torch.fft.fft2(image, dim=(2,3), norm='ortho')
    img_fft = torch.fft.fftshift(img_fft)
    _, _, in_h, in_w = image.shape
    alpha = 1
    mask = nn.Parameter(CircleFilter(in_h, in_w, 2, alpha).unsqueeze(0), requires_grad=False).to(args.device)
    low_freq = torch.fft.ifft2(torch.fft.ifftshift(img_fft * mask), s=(in_h, in_w), dim=(2, 3), norm='ortho').real
    high_freq = image - low_freq

    return high_freq, low_freq


class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.omega_sp = 1
        self.omega_lp = 0.5
        self.weig_func = lambda x, y: torch.exp((x - x.min()) / (x.max() - x.min()) * y)

    def forward(self, pred, gt):
        gt_high_freq, gt_low_freq = getHighLowFre(gt)

        y_sp = torch.abs(gt_high_freq)
        w_y_sp = self.weig_func(y_sp, self.omega_sp).detach()

        y_lp = torch.abs(gt_low_freq)
        w_y_lp = self.weig_func(y_lp, self.omega_lp).detach()

        y_hat = gt - pred
        loss = torch.mean(w_y_sp * w_y_lp * torch.abs(y_hat))

        return loss
