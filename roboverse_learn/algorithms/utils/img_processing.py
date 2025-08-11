import numpy as np
import torch
import torch.nn.functional as F


def _center_crop_and_resize(
    img: torch.Tensor | np.ndarray, target_width: int, target_height: int
) -> torch.Tensor | np.ndarray:
    """
    Args:
        img (torch.Tensor): Input image tensor of shape (N, H, W, C), range [0, 255], dtype uint8.
        target_width (int): Target width.
        target_height (int): Target height.
    Returns:
        torch.Tensor: Resized image tensor of shape (N, target_height, target_width, C),
                      range [0, 255], dtype uint8.
    """
    is_np = False
    need_squeeze = False
    if isinstance(img, np.ndarray):
        is_np = True
        img = torch.from_numpy(img)
    img_dtype = img.dtype
    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # Add batch dimension for consistency
        need_squeeze = True
    N, H, W, C = img.shape
    target_ratio = target_width / target_height
    orig_ratio = W / H
    K = None
    # determine crop size
    if orig_ratio > target_ratio:
        # input is wider → crop width
        new_h = H
        new_w = int(target_ratio * H)
    else:
        # input is taller → crop height
        new_w = W
        new_h = int(W / target_ratio)

    # compute crop coordinates
    left = (W - new_w) // 2
    top = (H - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    # center crop
    img_cropped = img[:, top:bottom, left:right, :]  # (N, new_h, new_w, C)
    # prepare for interpolation: to NCHW, float
    img_nchw = img_cropped.permute(0, 3, 1, 2).to(torch.float32)

    # resize with antialiasing for better quality
    img_resized = F.interpolate(
        img_nchw,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    # back to original shape and type
    img_out = img_resized.permute(0, 2, 3, 1).to(
        img_dtype
    )  # (N, target_height, target_width, C)

    if is_np:
        img_out = img_out.cpu().numpy()
        K = K.cpu().numpy() if K is not None else None
    if need_squeeze:
        img_out = img_out.squeeze(0)
    return img_out
