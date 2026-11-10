"""Fast unique-color and per-color mask extraction for color-coded segmentation masks."""

from importlib.metadata import (
    PackageNotFoundError,
    version,
)

from fastmask_seg import functional
from fastmask_seg.bbox import (
    mask_to_bbox,
    masks_to_bboxes,
)
from fastmask_seg.fast_mask import (
    Backend,
    ColorFormat,
    ColorMasks,
    FastMask,
    color_ids_to_colors,
    colors_to_color_ids,
    sort_colors,
)
from fastmask_seg.functional import (
    UniqueColorMasks,
    color_id_image,
    get_color_mask,
    get_color_masks,
    iter_color_masks,
    label_image,
    unique_color_counts,
    unique_color_ids,
    unique_color_masks,
    unique_colors,
)


try:
    __version__ = version("fastmask-seg")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

__all__ = [
    "Backend",
    "ColorFormat",
    "ColorMasks",
    "FastMask",
    "UniqueColorMasks",
    "__version__",
    "color_id_image",
    "color_ids_to_colors",
    "colors_to_color_ids",
    "functional",
    "get_color_mask",
    "get_color_masks",
    "iter_color_masks",
    "label_image",
    "mask_to_bbox",
    "masks_to_bboxes",
    "sort_colors",
    "unique_color_counts",
    "unique_color_ids",
    "unique_color_masks",
    "unique_colors",
]
