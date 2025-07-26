import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from math import exp

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class IoULoss(nn.Module):
    def __init__(self,smooth=1):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        intersection = pred_sigmoid * target
        iou_loss = 1 - (intersection.sum() + self.smooth) / (pred_sigmoid.sum() + target.sum() - intersection.sum() + self.smooth)
        return iou_loss

def temperature_weighted_infonce_loss(F_fore, F_back, M_fore, M_back, T_fore, T_back, temperature=0.07, alpha=2.0):
    """
    计算新的加权 InfoNCE 对比损失
    Args:
        F_fore:    [N, C] - MLP生成的前景特征
        F_back:    [M, C] - MLP生成的背景特征
        M_fore:    [N, C] - 真值图对应的前景特征
        M_back:    [M, C] - 真值图对应的背景特征
        T_fore:    [N] - 前景温度
        T_back:    [M] - 背景温度
        temperature: float
        alpha:      float

    Returns:
        scalar loss
    """

    fore_pos_sim = torch.matmul(F_fore, M_fore.T) / temperature
    back_pos_sim = torch.matmul(F_back, M_back.T) / temperature
    fore_back_sim = torch.matmul(F_fore, F_back.T) / temperature

    delta_T = torch.abs(T_fore.unsqueeze(1) - T_back.unsqueeze(0))
    weights = torch.exp(alpha * delta_T).detach()

    fore_numerator = torch.exp(fore_pos_sim)
    fore_denominator = fore_numerator + (torch.exp(fore_back_sim) * weights).sum(dim=1, keepdim=True)
    fore_loss = -torch.log(fore_numerator / fore_denominator)

    back_numerator = torch.exp(back_pos_sim)
    back_denominator = back_numerator + (torch.exp(fore_back_sim.T) * weights.T).sum(dim=1, keepdim=True)
    back_loss = -torch.log(back_numerator / back_denominator)

    return 0.01*(fore_loss.mean() + back_loss.mean())


def FinalLoss(pred, target, grad_pred1, grad_pred2, grad_pred3, grad_target,
              F_fore, F_back, M_fore, M_back, T_fore, T_back,
              edge_pred1,edge_pred2,edge_pred3,edge_pred4,edge_target):

    iou_loss=IoULoss()
    bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    dice_loss = DiceLoss()
    grad_loss = SSIMLoss()

    grad_target1 = F.interpolate(grad_target, size=grad_pred1.shape[2:], mode='bilinear', align_corners=False)
    grad_target2 = F.interpolate(grad_target, size=grad_pred2.shape[2:], mode='bilinear', align_corners=False)
    grad_target3 = F.interpolate(grad_target, size=grad_pred3.shape[2:], mode='bilinear', align_corners=False)

    edge_target1 = F.interpolate(edge_target, size=edge_pred1.shape[2:], mode='bilinear', align_corners=False)
    edge_target2 = F.interpolate(edge_target, size=edge_pred2.shape[2:], mode='bilinear', align_corners=False)
    edge_target3 = F.interpolate(edge_target, size=edge_pred3.shape[2:], mode='bilinear', align_corners=False)

    grad_loss1 = grad_loss(grad_pred1, grad_target1)
    grad_loss2 = grad_loss(grad_pred2, grad_target2)
    grad_loss3 = grad_loss(grad_pred3, grad_target3)
    final_grad=(grad_loss1+grad_loss2+grad_loss3)/3

    edge_loss1 = F.binary_cross_entropy_with_logits(edge_pred1, edge_target1)
    edge_loss2 = F.binary_cross_entropy_with_logits(edge_pred2, edge_target2)
    edge_loss3 = F.binary_cross_entropy_with_logits(edge_pred3, edge_target3)
    edge_loss4 = F.binary_cross_entropy_with_logits(edge_pred4, edge_target)
    final_edge=(edge_loss1+edge_loss2+edge_loss3+edge_loss4)/4

    return (iou_loss(pred,target)+bce_loss + dice_loss(pred, target) + 0.5*final_grad + 0.5*final_edge +
             temperature_weighted_infonce_loss(F_fore, F_back, M_fore, M_back, T_fore, T_back))
