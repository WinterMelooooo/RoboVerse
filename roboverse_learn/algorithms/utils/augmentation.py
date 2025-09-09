import torch
import torch.nn as nn
import torchvision

class OptionalAugment(nn.Module):
    """
    Train 模式下按概率对图像做轻量增强；Eval 模式或禁用时直接透传。
    设计原则：
      - 仅 photometric：亮度/对比度/饱和度、灰度、模糊、噪声、遮挡。
      - 不做水平翻转/旋转，避免机器人场景左右/坐标语义被破坏。
    """
    def __init__(
        self,
        enabled: bool = False,
        color_jitter_strength: float = 0.3,   # 亮度/对比度/饱和度/色相幅度
        blur_prob: float = 0.2,
        grayscale_prob: float = 0.1,
        erase_prob: float = 0.2,
        noise_std: float = 0.02               # 高斯噪声标准差（相对[0,1]）
    ):
        super().__init__()
        self.enabled = enabled
        # torchvision 的这些算子支持 Tensor，逐样本调用最稳妥
        self.t_color = torchvision.transforms.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=min(0.1, color_jitter_strength)  # 避免过强色偏
        )
        # 高斯模糊核大小用奇数，sigma 轻微抖动
        self.t_blur = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
        self.t_gray = torchvision.transforms.RandomGrayscale(p=1.0)  # 外层控制概率
        self.t_erase = torchvision.transforms.RandomErasing(
            p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random', inplace=False
        )
        self.blur_prob = blur_prob
        self.grayscale_prob = grayscale_prob
        self.erase_prob = erase_prob
        self.noise_std = noise_std

    def _augment_one(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W) in [0,1]
        # 1) 颜色抖动
        x = self.t_color(x)
        # 2) 随机灰度
        if torch.rand(()) < self.grayscale_prob:
            x = self.t_gray(x)
        # 3) 轻微模糊（模拟运动/对焦误差）
        if torch.rand(()) < self.blur_prob:
            x = self.t_blur(x)
        # 4) 加性高斯噪声（模拟感光噪声）
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = (x + noise).clamp(0.0, 1.0)
        # 5) 随机擦除（模拟小遮挡）
        if torch.rand(()) < self.erase_prob:
            x = self.t_erase(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) 且值域在 [0,1]（在 Normalize 之前调用）
        """
        if (not self.enabled) or (not self.training):
            return x
        # 逐样本增强，最大程度兼容不同 torchvision 版本
        xs = []
        for i in range(x.shape[0]):
            xi = x[i]
            xi = self._augment_one(xi)
            xs.append(xi)
        return torch.stack(xs, dim=0)
