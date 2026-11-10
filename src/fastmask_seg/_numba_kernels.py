"""Numba kernels for the optional ``numba`` extra.

Each kernel compiles once per argument signature and caches the machine code on disk. Kernels run serially on the
calling thread and release the GIL while they run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit


if TYPE_CHECKING:
    import numpy.typing as npt


@njit(cache=True, nogil=True)
def rgb_masks(
    image: npt.NDArray[np.uint8],
    color_ids: npt.NDArray[np.uint32],
    masks: npt.NDArray[np.bool_],
) -> None:
    """Write one boolean mask per color ID of a 3-channel image in a single pass over the image.

    Each row is packed into color IDs in a row-sized buffer that stays in cache, and every requested ID is compared
    against that buffer before the next row is read. The image is read once and each mask is written once.

    Args:
      image (npt.NDArray[np.uint8]): Image of shape ``(H, W, 3)``.
      color_ids (npt.NDArray[np.uint32]): 1-D color IDs, channel 0 in the least significant byte. IDs above
        ``0xFFFFFF`` give all-False masks.
      masks (npt.NDArray[np.bool_]): Output of shape ``(len(color_ids), H, W)``.
    """
    height, width, _ = image.shape
    row_ids = np.empty(width, dtype=np.uint32)
    for y in range(height):
        for x in range(width):
            row_ids[x] = (
                np.uint32(image[y, x, 0]) | (np.uint32(image[y, x, 1]) << 8) | (np.uint32(image[y, x, 2]) << 16)
            )
        for j in range(color_ids.size):
            color_id = color_ids[j]
            for x in range(width):
                masks[j, y, x] = row_ids[x] == color_id


_BLOCK_PIXELS = 32
_BLOCK_BYTES = 3 * _BLOCK_PIXELS
_RELABEL_BLOCK = 256


@njit(cache=True, nogil=True)
def _is_single_color(block: npt.NDArray[np.uint8]) -> bool:
    """Test whether a block of ``_BLOCK_PIXELS`` 3-channel pixels holds one color.

    Every byte is compared with the byte 3 positions later, its counterpart in the next pixel, in one branch-free
    loop that compiles to SIMD code.

    Args:
      block (npt.NDArray[np.uint8]): 1-D contiguous array of ``_BLOCK_BYTES`` channel values.

    Returns:
      bool: True when all pixels of the block are equal.
    """
    difference = np.uint8(0)
    for i in range(_BLOCK_BYTES - 3):
        difference |= block[i] ^ block[i + 3]
    return bool(difference == 0)


@njit(cache=True, nogil=True)
def _is_single_value(block: npt.NDArray[np.uint8]) -> bool:
    """Test whether every element of a block equals its first element.

    Args:
      block (npt.NDArray[np.uint8]): 1-D contiguous array with at least one element.

    Returns:
      bool: True when all elements are equal.
    """
    difference = np.uint8(0)
    for i in range(block.size):
        difference |= block[i] ^ block[0]
    return bool(difference == 0)


@njit(cache=True, nogil=True)
def _find_color(color_ids: npt.NDArray[np.uint32], num_colors: int, color_id: np.uint32) -> int:
    """Return the position of ``color_id`` among the first ``num_colors`` entries, or ``num_colors`` when new.

    Args:
      color_ids (npt.NDArray[np.uint32]): Color IDs found so far, in order of first appearance.
      num_colors (int): Number of valid entries of ``color_ids``.
      color_id (np.uint32): Color ID to look up.

    Returns:
      int: Index of ``color_id``, or ``num_colors`` when it has yet to appear.
    """
    index = 0
    while index < num_colors and color_ids[index] != color_id:
        index += 1
    return index


@njit(cache=True, nogil=True)
def rgb_analyze(
    image: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    color_ids: npt.NDArray[np.uint32],
    counts: npt.NDArray[np.int64],
) -> int:
    """Find the colors of a 3-channel image, count their pixels, and label every pixel in a single pass.

    The image is read in blocks of ``_BLOCK_PIXELS`` pixels. A block of one color, the common case in a segmentation
    mask, is recognized with one SIMD test and handled as a unit; other blocks are handled pixel by pixel. Colors are
    numbered in order of first appearance.

    Args:
      image (npt.NDArray[np.uint8]): C-contiguous image of shape ``(H, W, 3)``.
      labels (npt.NDArray[np.uint8]): C-contiguous output of shape ``(H, W)`` that receives, for every pixel, the index
        of its color in ``color_ids``.
      color_ids (npt.NDArray[np.uint32]): 1-D output that receives the color IDs in order of first appearance, channel
        0 in the least significant byte. Its size, at most 256, is the largest number of colors the call handles.
      counts (npt.NDArray[np.int64]): 1-D output with the size of ``color_ids`` that receives each color's pixel count.

    Returns:
      int: Number of colors found, or -1 when the image holds more colors than ``color_ids`` has entries, in which
        case the outputs are incomplete.
    """
    flat_image = image.reshape(-1)
    flat_labels = labels.reshape(-1)
    num_pixels = flat_labels.size
    num_colors = 0
    current_id = np.uint32(0xFFFFFFFF)  # above every 24-bit color ID, so the first pixel starts a new color
    current = 0
    for block in range((num_pixels + _BLOCK_PIXELS - 1) // _BLOCK_PIXELS):
        start = block * _BLOCK_PIXELS
        stop = min(start + _BLOCK_PIXELS, num_pixels)
        block_bytes = flat_image[3 * start : 3 * stop]
        # A single-color block needs one lookup; the first pixel of any other block is looked up per pixel.
        is_single = stop - start == _BLOCK_PIXELS and _is_single_color(block_bytes)
        for pixel in range(1 if is_single else stop - start):
            color_id = (
                np.uint32(block_bytes[3 * pixel])
                | (np.uint32(block_bytes[3 * pixel + 1]) << 8)
                | (np.uint32(block_bytes[3 * pixel + 2]) << 16)
            )
            if color_id != current_id:
                current = _find_color(color_ids, num_colors, color_id)
                if current == num_colors:
                    if num_colors == color_ids.size:
                        return -1
                    color_ids[current] = color_id
                    counts[current] = 0
                    num_colors += 1
                current_id = color_id
            if is_single:
                counts[current] += _BLOCK_PIXELS
                flat_labels[start:stop] = current
            else:
                counts[current] += 1
                flat_labels[start + pixel] = current
    return num_colors


@njit(cache=True, nogil=True)
def relabel(labels: npt.NDArray[np.uint8], new_labels: npt.NDArray[np.uint8]) -> None:
    """Replace every label by its entry in a lookup table, in place.

    Blocks holding one label are recognized with one SIMD test and rewritten as a unit.

    Args:
      labels (npt.NDArray[np.uint8]): C-contiguous label image, rewritten in place.
      new_labels (npt.NDArray[np.uint8]): New label for each current label.
    """
    flat_labels = labels.reshape(-1)
    for block in range((flat_labels.size + _RELABEL_BLOCK - 1) // _RELABEL_BLOCK):
        block_labels = flat_labels[block * _RELABEL_BLOCK : (block + 1) * _RELABEL_BLOCK]
        if _is_single_value(block_labels):
            block_labels[:] = new_labels[block_labels[0]]
        else:
            for i in range(block_labels.size):
                block_labels[i] = new_labels[block_labels[i]]
