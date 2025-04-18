import torch


class Preprocessor(object):
    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = torch.tensor(norm_mean).view(1,3,1,1)
        self.norm_std = torch.tensor(norm_std).view(1,3,1,1)
        self.sign = sign

    def __call__(self, rgbd):
        """
        Args:
            rgbd: [B, 4, H, W], RGBD format, [-1, 1] range
        Returns:
            rgb: [B, 3, H, W], BGR format, [0, 1] range and z-scored to ave = 0, std ~ 1
            depth: [B, 3, H, W], BGR format, [0, 1] range and z-scored to ave = 0, std ~ 1
        """
        assert rgbd.shape[1] == 4, f"Input should be RGBD format, but got {rgbd.shape}"
        nrgb = rgbd[:, 0:3, :, :].clone()
        nrgb = nrgb[:, [2, 1, 0], :, :] # BGR [B, 3, H, W] [-1, 1]
        nrgb = (nrgb + 1) / 2.0 # BGR [B, 3, H, W] [0, 1]
        modal_x = rgbd[:, 3:4, :, :].clone() # [B, 1, H, W]
        modal_x = modal_x.repeat(1, 3, 1, 1) # [B, 3, H, W] [-1, 1]
        modal_x = (modal_x + 1) / 2.0 # [B, 3, H, W] [0, 1]

        device = nrgb.device
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        rgb = (nrgb - self.norm_mean) / self.norm_std
        modal_x = (modal_x - 0.48) / 0.28
        return rgb, modal_x
