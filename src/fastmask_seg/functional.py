"""Functional interface to FastMask.

Every function takes an image or an existing ``FastMask`` as its first argument and returns what the ``FastMask``
attribute or method of the same name returns. Passing a ``FastMask`` reuses its cached color ID image, run table, and
label image across calls; passing an image analyzes it from scratch on each call.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
)

from fastmask_seg.fast_mask import (
    ColorMasks,
    FastMask,
)


if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import numpy.typing as npt

    from fastmask_seg.fast_mask import Backend

__all__ = [
    "UniqueColorMasks",
    "color_id_image",
    "get_color_mask",
    "get_color_masks",
    "iter_color_masks",
    "label_image",
    "unique_color_counts",
    "unique_color_ids",
    "unique_color_masks",
    "unique_colors",
]


class UniqueColorMasks(NamedTuple):
    """Every unique color of an image together with its mask.

    Attributes:
      colors (npt.NDArray[np.uint8]): Unique colors in the image's channel layout, ordered by color ID.
      color_ids (npt.NDArray[Any]): Sorted unique color IDs.
      color_counts (npt.NDArray[np.int64]): Pixel count of each color.
      masks (npt.NDArray[np.bool_]): Masks of shape ``(num_masks, H, W)``, one per color in the same order. When the
        caller supplied an ``out`` buffer this is a view of its first ``num_masks`` entries.
      num_masks (int): Number of masks written, equal to the number of unique colors.
    """

    colors: npt.NDArray[np.uint8]
    color_ids: npt.NDArray[Any]
    color_counts: npt.NDArray[np.int64]
    masks: npt.NDArray[np.bool_]
    num_masks: int


def _as_fast_mask(
    source: npt.NDArray[np.uint8] | FastMask,
    color_id_buffer: npt.NDArray[np.uint32] | None,
    backend: Backend = "auto",
) -> FastMask:
    """Return ``source`` when it is a FastMask, or wrap the image in a new one.

    Args:
      source (npt.NDArray[np.uint8] | FastMask): Image or existing FastMask. A FastMask keeps its own
        ``color_id_buffer`` and ``backend``.
      color_id_buffer (npt.NDArray[np.uint32] | None): Color ID storage for a new FastMask.
      backend (Backend = "auto"): Mask backend for a new FastMask.

    Returns:
      FastMask: The FastMask to query.
    """
    if isinstance(source, FastMask):
        return source
    return FastMask(source, color_id_buffer=color_id_buffer, backend=backend)


def color_id_image(
    source: npt.NDArray[np.uint8] | FastMask,
    /,
    *,
    color_id_buffer: npt.NDArray[np.uint32] | None = None,
) -> npt.NDArray[Any]:
    """Return the image with every pixel replaced by its color ID.

    Args:
      source (npt.NDArray[np.uint8] | FastMask): Image or existing FastMask.
      color_id_buffer (npt.NDArray[np.uint32] | None = None): ``(H, W)`` uint32 storage for the color IDs of a
        3-channel image source, see ``FastMask``.

    Returns:
      npt.NDArray[Any]: ``(H, W)`` color IDs, uint8 for grayscale and uint32 for color images.
    """
    return _as_fast_mask(source, color_id_buffer).color_id_image


def unique_color_ids(
    source: npt.NDArray[np.uint8] | FastMask,
    /,
    *,
    color_id_buffer: npt.NDArray[np.uint32] | None = None,
) -> npt.NDArray[Any]:
    """Return the sorted unique color IDs present in an image.

    Args:
      source (npt.NDArray[np.uint8] | FastMask): Image or existing FastMask.
      color_id_buffer (npt.NDArray[np.uint32] | None = None): ``(H, W)`` uint32 storage for the color IDs of a
        3-channel image source, see ``FastMask``.

    Returns:
      npt.NDArray[Any]: Sorted unique color IDs.
    """
    return _as_fast_mask(source, color_id_buffer).unique_color_ids


def unique_colors(
    source: npt.NDArray[np.uint8] | FastMask,
    /,
    *,
    color_id_buffer: npt.NDArray[np.uint32] | None = None,
) -> npt.NDArray[np.uint8]:
    """Return the unique colors present in an image, ordered by color ID.

    Args:
      source (npt.NDArray[np.uint8] | FastMask): Image or existing FastMask.
      color_id_buffer (npt.NDArray[np.uint32] | None = None): ``(H, W)`` uint32 storage for the color IDs of a
        3-channel image source, see ``FastMask``.

    Returns:
      npt.NDArray[np.uint8]: Unique colors in the image's channel layout; a 1-D array of gray values for grayscale.
    """
    return _as_fast_mask(source, color_id_buffer).unique_colors


def unique_color_counts(
    source: npt.NDArray[np.uint8] | FastMask,
    /,
    *,
    color_id_buffer: npt.NDArray[np.uint32] | None = None,
) -> npt.NDArray[np.int64]:
    """Return the pixel count of each unique color, ordered by color ID.

    Args:
      source (npt.NDArray[np.uint8] | FastMask): Image or existing FastMask.
      color_id_buffer (npt.NDArray[np.uint32] | None = None): ``(H, W)`` uint32 storage for the color IDs of a
        3-channel image source, see ``FastMask``.

    Returns:
      npt.NDArray[np.int64]: Pixel counts aligned with ``unique_color_ids``.
    """
    return _as_fast_mask(source, color_id_buffer).unique_color_counts


def label_image(
    source: npt.NDArray[np.uint8] | FastMask,
    /,
    *,
    color_id_buffer: npt.NDArray[np.uint32] | None = None,
) -> npt.NDArray[Any]:
    """Return the image with every pixel replaced by its index into the sorted unique color IDs.

    Args:
      source (npt.NDArray[np.uint8] | FastMask): Image or existing FastMask.
      color_id_buffer (npt.NDArray[np.uint32] | None = None): ``(H, W)`` uint32 storage for the color IDs of a
        3-channel image source, see ``FastMask``.

    Returns:
      npt.NDArray[Any]: ``(H, W)`` label indices in the narrowest unsigned dtype that holds them.
    """
    return _as_fast_mask(source, color_id_buffer).label_image


def get_color_masks(
    source: npt.NDArray[np.uint8] | FastMask,
    /,
    *,
    colors: npt.NDArray[np.uint8] | None = None,
    color_ids: npt.NDArray[np.integer] | None = None,
    out: npt.NDArray[np.bool_] | None = None,
    color_id_buffer: npt.NDArray[np.uint32] | None = None,
    backend: Backend = "auto",
) -> ColorMasks:
    """Generate a boolean mask for each requested color or color ID.

    Args:
      source (npt.NDArray[np.uint8] | FastMask): Image or existing FastMask.
      colors (npt.NDArray[np.uint8] | None = None): uint8 colors in the image's channel layout. When both ``colors`` and
        ``color_ids`` are None, masks are generated for every unique color.
      color_ids (npt.NDArray[np.integer] | None = None): 1-D integer color IDs.
      out (npt.NDArray[np.bool_] | None = None): Bool buffer of shape ``(N, H, W)`` with ``N`` at least the number of
        requested masks; masks are written to its first entries. None allocates a new array.
      color_id_buffer (npt.NDArray[np.uint32] | None = None): ``(H, W)`` uint32 storage for the color IDs of a
        3-channel image source, see ``FastMask``.
      backend (Backend = "auto"): Mask backend for an image source, see ``FastMask``. A FastMask source uses its own
        ``backend``.

    Returns:
      ColorMasks: The masks, shape ``(num_masks, H, W)`` in request order, and ``num_masks``.
    """
    return _as_fast_mask(source, color_id_buffer, backend).get_color_masks(colors=colors, color_ids=color_ids, out=out)


def get_color_mask(
    source: npt.NDArray[np.uint8] | FastMask,
    /,
    *,
    color: npt.NDArray[np.uint8] | None = None,
    color_id: int | None = None,
    out: npt.NDArray[np.bool_] | None = None,
    color_id_buffer: npt.NDArray[np.uint32] | None = None,
    backend: Backend = "auto",
) -> npt.NDArray[np.bool_]:
    """Generate the boolean mask of one color or color ID.

    Args:
      source (npt.NDArray[np.uint8] | FastMask): Image or existing FastMask.
      color (npt.NDArray[np.uint8] | None = None): One uint8 color in the image's channel layout.
      color_id (int | None = None): One color ID.
      out (npt.NDArray[np.bool_] | None = None): Bool buffer of shape ``(H, W)`` that receives the mask and is
        returned. None allocates a new array.
      color_id_buffer (npt.NDArray[np.uint32] | None = None): ``(H, W)`` uint32 storage for the color IDs of a
        3-channel image source, see ``FastMask``.
      backend (Backend = "auto"): Mask backend for an image source, see ``FastMask``. A FastMask source uses its own
        ``backend``.

    Returns:
      npt.NDArray[np.bool_]: Mask of shape ``(H, W)``, all False when the color is absent from the image.
    """
    return _as_fast_mask(source, color_id_buffer, backend).get_color_mask(color=color, color_id=color_id, out=out)


def iter_color_masks(
    source: npt.NDArray[np.uint8] | FastMask,
    /,
    *,
    colors: npt.NDArray[np.uint8] | None = None,
    color_ids: npt.NDArray[np.integer] | None = None,
    out: npt.NDArray[np.bool_] | None = None,
    color_id_buffer: npt.NDArray[np.uint32] | None = None,
    backend: Backend = "auto",
) -> Iterator[npt.NDArray[np.bool_]]:
    """Return an iterator over the mask of each requested color, one at a time.

    Args:
      source (npt.NDArray[np.uint8] | FastMask): Image or existing FastMask.
      colors (npt.NDArray[np.uint8] | None = None): uint8 colors in the image's channel layout. When both ``colors`` and
        ``color_ids`` are None, masks are generated for every unique color.
      color_ids (npt.NDArray[np.integer] | None = None): 1-D integer color IDs.
      out (npt.NDArray[np.bool_] | None = None): Bool buffer of shape ``(H, W)`` yielded for every mask, overwritten
        in place. None allocates each mask.
      color_id_buffer (npt.NDArray[np.uint32] | None = None): ``(H, W)`` uint32 storage for the color IDs of a
        3-channel image source, see ``FastMask``.
      backend (Backend = "auto"): Mask backend for an image source, see ``FastMask``. A FastMask source uses its own
        ``backend``.

    Returns:
      Iterator[npt.NDArray[np.bool_]]: The ``(H, W)`` mask for each requested color, in request order.
    """
    return _as_fast_mask(source, color_id_buffer, backend).iter_color_masks(colors=colors, color_ids=color_ids, out=out)


def unique_color_masks(
    source: npt.NDArray[np.uint8] | FastMask,
    /,
    *,
    out: npt.NDArray[np.bool_] | None = None,
    color_id_buffer: npt.NDArray[np.uint32] | None = None,
    backend: Backend = "auto",
) -> UniqueColorMasks:
    """Return every unique color of an image with its pixel count and mask, from one analysis.

    Args:
      source (npt.NDArray[np.uint8] | FastMask): Image or existing FastMask.
      out (npt.NDArray[np.bool_] | None = None): Bool buffer of shape ``(N, H, W)`` with ``N`` at least the number of
        unique colors; masks are written to its first entries. None allocates a new array.
      color_id_buffer (npt.NDArray[np.uint32] | None = None): ``(H, W)`` uint32 storage for the color IDs of a
        3-channel image source, see ``FastMask``.
      backend (Backend = "auto"): Mask backend for an image source, see ``FastMask``. A FastMask source uses its own
        ``backend``.

    Returns:
      UniqueColorMasks: Colors, color IDs, pixel counts, masks, and the number of masks.
    """
    fast_mask = _as_fast_mask(source, color_id_buffer, backend)
    color_masks = fast_mask.get_color_masks(out=out)
    return UniqueColorMasks(
        colors=fast_mask.unique_colors,
        color_ids=fast_mask.unique_color_ids,
        color_counts=fast_mask.unique_color_counts,
        masks=color_masks.masks,
        num_masks=color_masks.num_masks,
    )
