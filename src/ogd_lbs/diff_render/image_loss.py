# filepath: src/ogd_lbs/diff_render/image_loss.py
import torch
import torch.nn as nn
from typing import Tuple, Dict

from ogd_lbs.diff_render.config import DiffRenderConfig
from ogd_lbs.utils.common import log, warn

try:
    import lpips
    from pytorch_msssim import ssim
    HAS_ADVANCED_LOSS = True
except ImportError:
    HAS_ADVANCED_LOSS = False


class CompositeImageLoss(nn.Module):
    """
    高层复合图像 Loss：L1 + MSE + SSIM (结构) + LPIPS (感知纹理)
    支持接收空间掩码 (weight_mask) 进行加权计算。
    """
    def __init__(self, config: DiffRenderConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        self.l1_weight = config.loss_l1_weight
        self.mse_weight = config.loss_mse_weight
        self.ssim_weight = config.loss_ssim_weight
        self.lpips_weight = config.loss_lpips_weight

        self.lpips_net = None

        if (self.ssim_weight > 0 or self.lpips_weight > 0) and not HAS_ADVANCED_LOSS:
            warn("未安装 lpips 或 pytorch-msssim，将强制回退到纯 L1/MSE Loss。请运行: pip install lpips pytorch-msssim")
            self.ssim_weight = 0.0
            self.lpips_weight = 0.0

        # 🌟 修复：如果开启了深度评估，即使 weight 为 0 也强制加载 LPIPS 模型用于测试
        if (self.lpips_weight > 0 or config.eval_interval > 0) and HAS_ADVANCED_LOSS:
            log("正在加载 VGG-LPIPS 预训练模型 (用于训练或深度评估)...")
            self.lpips_net = lpips.LPIPS(net='vgg').to(device)
            for param in self.lpips_net.parameters():
                param.requires_grad = False

        log(f"复合 Image Loss 初始化完成 | L1: {self.l1_weight} | MSE: {self.mse_weight} | SSIM: {self.ssim_weight} | LPIPS: {self.lpips_weight}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight_mask: torch.Tensor = None, eval_mode: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 转换为 PyTorch 视觉标准格式 [B, C, H, W]
        p = pred[..., :3].permute(0, 3, 1, 2).contiguous()
        t = target[..., :3].permute(0, 3, 1, 2).contiguous()

        loss_dict = {}
        total_loss = torch.zeros((), dtype=torch.float32, device=self.device)

        # 🌟 修复：在 eval_mode 下强制计算所有指标
        compute_l1 = self.l1_weight > 0 or eval_mode
        compute_mse = self.mse_weight > 0 or eval_mode
        compute_ssim = self.ssim_weight > 0 or eval_mode
        compute_lpips = (self.lpips_weight > 0 or eval_mode) and self.lpips_net is not None

        # 1. L1 Loss
        if compute_l1:
            l1_diff = torch.abs(p - t)
            if weight_mask is not None:
                wm = weight_mask.permute(0, 3, 1, 2).contiguous()
                l1_loss = torch.sum(l1_diff * wm) / (torch.sum(wm) * 3 + 1e-8)
            else:
                l1_loss = l1_diff.mean()
            loss_dict['l1'] = l1_loss.item()
            if self.l1_weight > 0:
                total_loss += self.l1_weight * l1_loss

        # 2. MSE Loss
        if compute_mse:
            mse_diff = (p - t) ** 2
            if weight_mask is not None:
                wm = weight_mask.permute(0, 3, 1, 2).contiguous()
                mse_loss = torch.sum(mse_diff * wm) / (torch.sum(wm) * 3 + 1e-8)
            else:
                mse_loss = mse_diff.mean()
            loss_dict['mse'] = mse_loss.item()
            if self.mse_weight > 0:
                total_loss += self.mse_weight * mse_loss

        # 3. SSIM Loss
        if compute_ssim:
            ssim_val = ssim(p, t, data_range=1.0, size_average=True)
            ssim_loss = 1.0 - ssim_val
            loss_dict['ssim'] = ssim_loss.item()
            if self.ssim_weight > 0:
                total_loss += self.ssim_weight * ssim_loss

        # 4. LPIPS Loss
        if compute_lpips:
            p_lpips = p * 2.0 - 1.0
            t_lpips = t * 2.0 - 1.0
            lpips_map = self.lpips_net(p_lpips, t_lpips)

            if weight_mask is not None:
                wm = weight_mask.permute(0, 3, 1, 2).contiguous()
                wm = torch.nn.functional.interpolate(wm, size=lpips_map.shape[2:], mode='bilinear', align_corners=False)
                lpips_loss = torch.sum(lpips_map * wm) / (torch.sum(wm) + 1e-8)
            else:
                lpips_loss = lpips_map.mean()

            loss_dict['lpips'] = lpips_loss.item()
            if self.lpips_weight > 0:
                total_loss += self.lpips_weight * lpips_loss

        return total_loss, loss_dict
