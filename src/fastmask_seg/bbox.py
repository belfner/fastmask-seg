"""Bounding boxes of boolean masks.

A box is ``(rmin, rmax, cmin, cmax)`` with exclusive ends, so ``mask[rmin:rmax, cmin:cmax]`` crops a mask to its
True pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["mask_to_bbox", "masks_to_bboxes"]


def masks_to_bboxes(masks: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
    """Compute the bounding box of every mask in a stack.

    One ``np.any`` over each row of the whole stack finds the occupied rows, and the occupied columns of each mask come
    from an ``np.any`` over its occupied rows alone.

    Args:
      masks (npt.NDArray[np.bool_]): Bool stack of shape ``(K, H, W)``.

    Returns:
      npt.NDArray[np.int64]: Boxes of shape ``(K, 4)``, one ``(rmin, rmax, cmin, cmax)`` row per mask with exclusive
        ends.

    Raises:
      ValueError: A mask has no True pixel; the message names its index.
    """
    num_masks, height, width = masks.shape
    out = np.empty((num_masks, 4), dtype=np.int64)
    if num_masks == 0:
        return out

    occupied_rows = masks.any(axis=2)
    has_pixels = occupied_rows.any(axis=1)
    if not has_pixels.all():
        raise ValueError(f"mask {int(np.flatnonzero(~has_pixels)[0])} has no True pixels")
    out[:, 0] = occupied_rows.argmax(axis=1)
    out[:, 1] = height - occupied_rows[:, ::-1].argmax(axis=1)

    for k in range(num_masks):
        occupied_cols = masks[k, out[k, 0] : out[k, 1]].any(axis=0)
        out[k, 2] = occupied_cols.argmax()
        out[k, 3] = width - occupied_cols[::-1].argmax()
    return out


def mask_to_bbox(mask: npt.NDArray[np.bool_]) -> tuple[int, int, int, int]:
    """Compute the bounding box of one mask.

    Args:
      mask (npt.NDArray[np.bool_]): Bool mask of shape ``(H, W)``.

    Returns:
      tuple[int, int, int, int]: ``(rmin, rmax, cmin, cmax)`` with exclusive ends.

    Raises:
      ValueError: ``mask`` has no True pixel.
    """
    rmin, rmax, cmin, cmax = masks_to_bboxes(mask[np.newaxis])[0].tolist()
    return rmin, rmax, cmin, cmax
