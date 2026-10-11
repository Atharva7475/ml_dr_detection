"""
Illumination-normalization module using Gaussian-blur subtraction (Ben Graham method).

Origin:
    Introduced by Ben Graham in the Kaggle Diabetic Retinopathy Detection competition (2015).
    Retinal fundus images suffer from substantial illumination variance, shadow gradients,
    and equipment-dependent lighting artifacts. This transform subtracts local average
    intensity via Gaussian filtering and rescales local contrast:

        I_norm = alpha * I + beta * GaussianBlur(I, sigma) + gamma

    Standard formulation:
        alpha = 4.0, beta = -4.0, gamma = 128.0
        sigma = image_width / 30.0 (or configurable scalar)

    A circular retinal boundary mask is applied so background pixels outside the retina
    remain 0 (black), preventing gray boundary bleeding artifacts.

Features:
    - Pure RGB preserving (does not permute channels).
    - Works with both PIL Image and NumPy ndarray inputs.
    - Configurable sigma (absolute pixels or proportional ratio).
    - Configurable circular boundary mask with smooth radius ratio.
    - Zero mutation of input arrays or images.
    - Deterministic across training, validation, testing, and inference.
    - Uses SciPy ndimage gaussian_filter by default, with OpenCV acceleration if available.
"""

from typing import Optional, Union, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

import scipy.ndimage


def ben_graham_filter(
    image: Union[Image.Image, np.ndarray],
    sigma: Optional[float] = None,
    sigma_ratio: float = 1.0 / 30.0,
    alpha: float = 4.0,
    beta: float = -4.0,
    gamma: float = 128.0,
    mask_background: bool = True,
    mask_radius_ratio: float = 0.98,
    return_pil: bool = True,
) -> Union[Image.Image, np.ndarray]:
    """
    Apply Ben Graham illumination normalization to an RGB image.

    Args:
        image (PIL.Image or np.ndarray): Input RGB image.
        sigma (float | None): Explicit Gaussian filter standard deviation in pixels.
            If None, computed as max(1.0, min(height, width) * sigma_ratio).
        sigma_ratio (float): Ratio of image dimension used to compute sigma if sigma is None.
            Default: 1/30 (0.0333), the standard Kaggle Ben Graham ratio.
        alpha (float): Scaling factor for original image. Default: 4.0.
        beta (float): Scaling factor for blurred image. Default: -4.0.
        gamma (float): Offset value for contrast normalization. Default: 128.0.
        mask_background (bool): If True, pixels outside circular retinal mask are zeroed.
            Default: True.
        mask_radius_ratio (float): Radius of circular mask as fraction of half min dimension.
            Default: 0.98.
        return_pil (bool): If True, returns PIL.Image. If False, returns np.ndarray (uint8).

    Returns:
        PIL.Image or np.ndarray: Normalized RGB image in range [0, 255] as uint8.
    """
    # 1. Convert input to float32 NumPy array without mutating original
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.float32)
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32, copy=True)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> RGB
            arr = arr[:, :, :3]
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image or np.ndarray.")

    height, width = arr.shape[:2]

    # 2. Determine blur scale (sigma)
    if sigma is None:
        computed_sigma = max(1.0, float(min(height, width)) * float(sigma_ratio))
    else:
        computed_sigma = max(0.5, float(sigma))

    # 3. Apply Gaussian blur
    # Note: OpenCV operates in spatial coordinates on each channel independently
    # preserving RGB channel ordering. SciPy operates along (H, W, 0).
    if _HAS_CV2:
        # cv2.GaussianBlur preserves channel order; sigmaX=sigma, sigmaY=sigma
        blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=computed_sigma, sigmaY=computed_sigma)
    else:
        blur = scipy.ndimage.gaussian_filter(arr, sigma=(computed_sigma, computed_sigma, 0))

    # 4. Illumination subtraction & local contrast enhancement
    enhanced = alpha * arr + beta * blur + gamma

    # 5. Circular boundary mask to prevent gray background bleeding
    if mask_background:
        cy, cx = height / 2.0, width / 2.0
        radius = (min(height, width) / 2.0) * mask_radius_ratio
        y_indices, x_indices = np.ogrid[:height, :width]
        dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
        inside_mask = dist_sq <= (radius ** 2)
        # Retain original black background outside circular retina boundary
        enhanced[~inside_mask] = 0.0

    # 6. Clip to valid [0, 255] range and cast to uint8
    enhanced = np.clip(enhanced, 0.0, 255.0).astype(np.uint8)

    # 7. Return desired type
    if return_pil:
        return Image.fromarray(enhanced, mode="RGB")
    return enhanced


class BenGrahamTransform:
    """
    Torchvision-compatible transform callable for Ben Graham preprocessing.

    Usage:
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            BenGrahamTransform(sigma_ratio=1.0/30.0, mask_background=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        sigma_ratio: float = 1.0 / 30.0,
        alpha: float = 4.0,
        beta: float = -4.0,
        gamma: float = 128.0,
        mask_background: bool = True,
        mask_radius_ratio: float = 0.98,
    ) -> None:
        self.sigma = sigma
        self.sigma_ratio = sigma_ratio
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mask_background = mask_background
        self.mask_radius_ratio = mask_radius_ratio

    def __call__(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        return ben_graham_filter(
            image=image,
            sigma=self.sigma,
            sigma_ratio=self.sigma_ratio,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            mask_background=self.mask_background,
            mask_radius_ratio=self.mask_radius_ratio,
            return_pil=True,
        )

    def __repr__(self) -> str:
        return (
            f"BenGrahamTransform(sigma={self.sigma}, sigma_ratio={self.sigma_ratio:.4f}, "
            f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, "
            f"mask_background={self.mask_background})"
        )
