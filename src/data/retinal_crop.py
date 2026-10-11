"""
Retinal field circular cropping and margin removal for fundus photography.

Detects the illuminated retinal aperture in fundus images and eliminates
uninformative black rectangular margins produced by camera sensors.
Preserves RGB channel ordering, avoids cutting off peripheral lesions,
and provides safe fallback to the original image if detection fails.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image


def detect_retina_bbox(
    image_np: np.ndarray,
    tolerance: int = 15,
    min_coverage: float = 0.01,
    pad_percent: float = 0.01,
) -> Tuple[Tuple[int, int, int, int], bool, str]:
    """
    Detect the bounding box of the illuminated retinal field.

    Args:
        image_np (np.ndarray): Input RGB image array of shape (H, W, 3) or grayscale (H, W).
        tolerance (int): Baseline intensity threshold for foreground detection (default: 15).
        min_coverage (float): Minimum active fraction of a row/column to qualify as retina (default: 0.01).
        pad_percent (float): Margin padding added around detected bounding box (default: 0.01).

    Returns:
        tuple: (
            (xmin, ymin, xmax, ymax): Bounding box coordinates,
            is_cropped (bool): True if black margins were detected and cropped,
            reason (str): Diagnostic message or status.
        )
    """
    if not isinstance(image_np, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image_np)}")

    if image_np.ndim == 2:
        gray = image_np
        h, w = gray.shape
    elif image_np.ndim == 3:
        h, w = image_np.shape[:2]
        # Maximum across RGB channels ensures sensitivity to faint red/green retinal peripheries
        gray = np.max(image_np, axis=2)
    else:
        return (0, 0, 0, 0), False, "invalid_dimensions"

    if h < 10 or w < 10:
        return (0, 0, w, h), False, "image_too_small"

    # 1. Estimate background noise from the four corners (sample 3% of minimum dimension)
    corner_size = max(4, int(min(h, w) * 0.03))
    corners = np.concatenate([
        gray[:corner_size, :corner_size].flatten(),
        gray[:corner_size, -corner_size:].flatten(),
        gray[-corner_size:, :corner_size].flatten(),
        gray[-corner_size:, -corner_size:].flatten(),
    ])
    bg_p95 = float(np.percentile(corners, 95))
    effective_thresh = max(float(tolerance), bg_p95 + 8.0)

    # 2. Binary mask
    mask = gray > effective_thresh

    # Check foreground volume: if less than 5% of pixels are active, image is likely black/corrupted
    total_active = np.sum(mask)
    if total_active < 0.05 * h * w:
        return (0, 0, w, h), False, "insufficient_foreground_pixels"

    # 3. Row and column coverage aggregation (immune to single-pixel sensor specks)
    row_coverage = np.sum(mask, axis=1) / float(w)
    col_coverage = np.sum(mask, axis=0) / float(h)

    valid_rows = np.where(row_coverage > min_coverage)[0]
    valid_cols = np.where(col_coverage > min_coverage)[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return (0, 0, w, h), False, "no_valid_retinal_bands"

    ymin, ymax = int(valid_rows[0]), int(valid_rows[-1] + 1)
    xmin, xmax = int(valid_cols[0]), int(valid_cols[-1] + 1)

    # 4. Add safety padding (1% of detected dimensions) to guarantee no peripheral lesions are clipped
    pad_h = int((ymax - ymin) * pad_percent)
    pad_w = int((xmax - xmin) * pad_percent)

    ymin = max(0, ymin - pad_h)
    ymax = min(h, ymax + pad_h)
    xmin = max(0, xmin - pad_w)
    xmax = min(w, xmax + pad_w)

    crop_w = xmax - xmin
    crop_h = ymax - ymin

    # 5. Degeneracy check: crop must not collapse below 20% of original dimensions
    if crop_w < 0.2 * w or crop_h < 0.2 * h:
        return (0, 0, w, h), False, "degenerate_detected_bbox"

    # 6. Unnecessary crop check: if bounding box covers >= 98% of both axes, keep as-is
    if crop_w >= 0.98 * w and crop_h >= 0.98 * h:
        return (0, 0, w, h), False, "already_tightly_framed"

    return (xmin, ymin, xmax, ymax), True, "successful_retinal_crop"


def pad_to_square(image: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
    """
    Symmetrically pad an image with black pixels (zeros) to a square aspect ratio.
    Prevents geometric distortion of the circular retina during subsequent resizing.

    Args:
        image (Image.Image | np.ndarray): Input image in PIL or NumPy format.

    Returns:
        Image.Image | np.ndarray: Square-padded image in same format as input.
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        arr = np.array(image)
    else:
        arr = image

    h, w = arr.shape[:2]
    if h == w:
        return image

    max_dim = max(h, w)
    pad_h = max_dim - h
    pad_w = max_dim - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if arr.ndim == 3:
        pad_spec = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        pad_spec = ((pad_top, pad_bottom), (pad_left, pad_right))

    padded_arr = np.pad(arr, pad_spec, mode="constant", constant_values=0)

    if is_pil:
        return Image.fromarray(padded_arr)
    return padded_arr


def crop_retina(
    image: Union[Image.Image, np.ndarray],
    tolerance: int = 15,
    min_coverage: float = 0.01,
    pad_percent: float = 0.01,
    preserve_aspect_ratio: bool = True,
    return_metadata: bool = False,
) -> Union[Union[Image.Image, np.ndarray], Tuple[Union[Image.Image, np.ndarray], Dict]]:
    """
    Crop extraneous black rectangular margins from a fundus image.

    Features:
    - Automatically handles variable camera resolutions and sensor aspect ratios.
    - Preserves RGB channel order.
    - Symmetrically pads crop to square if preserve_aspect_ratio=True.
    - Completely safe fallback: returns original image unmodified if detection fails.
    - Does not modify any disk files.

    Args:
        image (Image.Image | np.ndarray): Input fundus image.
        tolerance (int): Foreground detection tolerance (default: 15).
        min_coverage (float): Minimum active row/col coverage fraction (default: 0.01).
        pad_percent (float): Safety padding percent around detected crop (default: 0.01).
        preserve_aspect_ratio (bool): If True, pad cropped box to square with black borders.
        return_metadata (bool): If True, returns (cropped_image, metadata_dict).

    Returns:
        cropped_image or (cropped_image, metadata_dict)
    """
    is_pil = isinstance(image, Image.Image)
    try:
        if is_pil:
            arr = np.array(image)
        else:
            arr = np.array(image, copy=False)

        orig_h, orig_w = arr.shape[:2]
        bbox, is_cropped, reason = detect_retina_bbox(
            arr,
            tolerance=tolerance,
            min_coverage=min_coverage,
            pad_percent=pad_percent,
        )

        if not is_cropped:
            metadata = {
                "is_cropped": False,
                "bbox": (0, 0, orig_w, orig_h),
                "orig_size": (orig_w, orig_h),
                "new_size": (orig_w, orig_h),
                "reason": reason,
            }
            if return_metadata:
                return image, metadata
            return image

        xmin, ymin, xmax, ymax = bbox
        cropped_arr = arr[ymin:ymax, xmin:xmax]

        if preserve_aspect_ratio:
            cropped_arr = pad_to_square(cropped_arr)

        if is_pil:
            out_img = Image.fromarray(cropped_arr)
        else:
            out_img = cropped_arr

        new_h, new_w = cropped_arr.shape[:2]
        metadata = {
            "is_cropped": True,
            "bbox": bbox,
            "orig_size": (orig_w, orig_h),
            "new_size": (new_w, new_h),
            "reason": reason,
        }

        if return_metadata:
            return out_img, metadata
        return out_img

    except Exception as e:
        # Failsafe fallback: return original image untouched on any unhandled error
        metadata = {
            "is_cropped": False,
            "bbox": (0, 0, getattr(image, "size", (0, 0))[0], getattr(image, "size", (0, 0))[1]),
            "orig_size": getattr(image, "size", (0, 0)),
            "new_size": getattr(image, "size", (0, 0)),
            "reason": f"exception_fallback: {str(e)}",
        }
        if return_metadata:
            return image, metadata
        return image


class RetinalCropTransform:
    """
    Torchvision-compatible callable transform for retinal cropping.
    Can be inserted directly into torchvision.transforms.Compose.
    """

    def __init__(
        self,
        tolerance: int = 15,
        min_coverage: float = 0.01,
        pad_percent: float = 0.01,
        preserve_aspect_ratio: bool = True,
    ):
        self.tolerance = tolerance
        self.min_coverage = min_coverage
        self.pad_percent = pad_percent
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        return crop_retina(
            image=img,
            tolerance=self.tolerance,
            min_coverage=self.min_coverage,
            pad_percent=self.pad_percent,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
            return_metadata=False,
        )

    def __repr__(self) -> str:
        return (
            f"RetinalCropTransform(tolerance={self.tolerance}, "
            f"min_coverage={self.min_coverage}, pad_percent={self.pad_percent}, "
            f"preserve_aspect_ratio={self.preserve_aspect_ratio})"
        )
