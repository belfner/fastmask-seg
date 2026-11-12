"""Unique-color and per-color mask extraction for color-coded segmentation masks.

A segmentation mask stored as a color image marks each class or instance with its own color. Each pixel's channels are
packed into a single uint32 color ID (channel 0 in the least significant byte), so every color comparison is one
integer comparison. Masks consist of long runs of identical pixels, and the unique colors, their pixel counts, and a
compact label image are all derived from those runs.
"""

from __future__ import annotations

import functools
import importlib
import sys
import warnings
from enum import (
    Enum,
    auto,
)
from functools import (
    cache,
    cached_property,
)
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    cast,
)

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    import numpy.typing as npt

__all__ = [
    "Backend",
    "ColorFormat",
    "ColorMasks",
    "FastMask",
    "color_ids_to_colors",
    "colors_to_color_ids",
    "sort_colors",
]

# Files whose frames a warning skips to reach its caller: this package, and functools, whose cached_property runs
# between the caller and this package.
_WARNING_SKIP_PREFIXES = (str(Path(__file__).parent), functools.__file__)

# Backend: "auto" uses the numba kernels where they apply when numba is installed, and "numpy" uses NumPy alone.
type Backend = Literal["auto", "numpy"]

# Color IDs place channel 0 in the least significant byte on every platform, which lets a contiguous 4-channel uint8
# array be reinterpreted as color IDs in place.
_ID_DTYPE = np.dtype("<u4")

# Number of requested masks at which building the label image costs less than comparing every mask against the uint32
# color ID image.
_LABEL_PATH_MIN_MASKS = 4

# Largest number of colors the numba analysis kernel handles, the capacity of its uint8 label image.
_KERNEL_MAX_COLORS = 256

# Largest color ID of a 3-channel image. The numba kernel receives a larger ID for requests that cannot match.
_RGB_ID_MAX = 0xFFFFFF
_NO_MATCH_ID = 0xFFFFFFFF


class ColorFormat(Enum):
    """Channel layout of an image or color array.

    Attributes:
      SINGLE_CHANNEL: Single channel of any integer dtype (a grayscale or label image), where the pixel value is the
        color ID.
      COLOR: Three channels (RGB or BGR).
      COLOR_ALPHA: Four channels (RGBA or BGRA).
    """

    SINGLE_CHANNEL = auto()
    COLOR = auto()
    COLOR_ALPHA = auto()


class ColorMasks(NamedTuple):
    """Boolean masks for a set of requested colors.

    Attributes:
      masks (npt.NDArray[np.bool_]): Masks of shape ``(num_masks, H, W)`` in request order. When the caller supplied an
        ``out`` buffer this is a view of its first ``num_masks`` entries.
      num_masks (int): Number of masks written.
    """

    masks: npt.NDArray[np.bool_]
    num_masks: int


def _pack_contiguous_rgb(flat_colors: npt.NDArray[np.uint8], ids: npt.NDArray[np.uint32]) -> None:
    """Pack contiguous 3-channel colors into color IDs with one strided read.

    Pixel ``i`` occupies bytes ``3i`` to ``3i + 2``. Reading 4 bytes at offset ``3i`` also picks up byte ``3i + 3``,
    the first byte of pixel ``i + 1``, which the ``& 0xFFFFFF`` mask clears. Every such read stays inside the buffer for
    all pixels except the last, which is packed separately.

    Args:
      flat_colors (npt.NDArray[np.uint8]): C-contiguous 1-D array of ``3 * n`` channel values.
      ids (npt.NDArray[np.uint32]): 1-D output of ``n`` color IDs.
    """
    num_pixels = ids.size
    if num_pixels > 1:
        overlapping = np.ndarray(shape=(num_pixels - 1,), dtype=_ID_DTYPE, buffer=flat_colors, strides=(3,))
        np.bitwise_and(overlapping, 0xFFFFFF, out=ids[:-1])
    if num_pixels > 0:
        ids[-1] = int(flat_colors[-3]) | (int(flat_colors[-2]) << 8) | (int(flat_colors[-1]) << 16)


def colors_to_color_ids(
    colors: npt.NDArray[np.uint8],
    *,
    out: npt.NDArray[np.uint32] | None = None,
) -> npt.NDArray[np.uint32]:
    """Pack 3-channel or 4-channel colors into one uint32 ID per color.

    Channel 0 occupies the least significant byte of the ID, channel 1 the next byte, and so on, so an ID is the
    little-endian byte layout of its color. For 4-channel colors the alpha channel is part of the ID.

    Args:
      colors (npt.NDArray[np.uint8]): uint8 colors with shape ``(..., 3)`` or ``(..., 4)``. 4-channel colors with a
        contiguous last axis and ``out`` None give a zero-copy view of ``colors``.
      out (npt.NDArray[np.uint32] | None = None): C-contiguous little-endian uint32 buffer of shape
        ``colors.shape[:-1]``, in storage separate from ``colors``, that receives the IDs and is returned. None
        allocates the result.

    Returns:
      npt.NDArray[np.uint32]: Color IDs with the shape of ``colors`` minus its last axis.
    """
    num_channels = colors.shape[-1]

    # An RGBA/BGRA array already holds its color IDs byte for byte.
    if out is None and num_channels == 4 and colors.strides[-1] == 1:
        try:
            return colors.view(_ID_DTYPE)[..., 0]
        except ValueError:
            # Layouts that view() rejects take the copy path below.
            pass

    ids = np.empty(colors.shape[:-1], dtype=_ID_DTYPE) if out is None else out
    if num_channels == 3 and colors.flags.c_contiguous:
        _pack_contiguous_rgb(colors.reshape(-1), ids.reshape(-1))
        return ids

    # Fill the uint32 output byte by byte, one channel at a time.
    ids_bytes = ids.reshape(-1).view(np.uint8).reshape(-1, 4)
    flat_colors = colors.reshape(-1, num_channels)
    for channel in range(num_channels):
        ids_bytes[:, channel] = flat_colors[:, channel]
    if num_channels == 3:
        ids_bytes[:, 3] = 0
    return ids


def sort_colors(colors: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Sort colors into a canonical order.

    Colors are ordered by their color ID, which compares the last channel first and the first channel last. For RGB
    input the order is by B, then G, then R.

    Args:
      colors (npt.NDArray[np.uint8]): uint8 colors with shape ``(N, 3)`` or ``(N, 4)``.

    Returns:
      npt.NDArray[np.uint8]: Sorted, C-contiguous uint8 copy of ``colors``.
    """
    ids = np.sort(colors_to_color_ids(colors))
    color_format = ColorFormat.COLOR if colors.shape[-1] == 3 else ColorFormat.COLOR_ALPHA
    # Unpacking the sorted IDs costs less than an argsort followed by a gather.
    return np.ascontiguousarray(color_ids_to_colors(ids, color_format))


def color_ids_to_colors(
    color_ids: npt.NDArray[np.uint32],
    color_format: ColorFormat = ColorFormat.COLOR,
) -> npt.NDArray[np.uint8]:
    """Unpack color IDs into their channel values.

    Args:
      color_ids (npt.NDArray[np.uint32]): Little-endian uint32 color IDs of any shape with a contiguous last axis.
      color_format (ColorFormat = ColorFormat.COLOR): ``COLOR_ALPHA`` produces 4 channels and ``COLOR`` produces 3.

    Returns:
      npt.NDArray[np.uint8]: Colors with shape ``(*color_ids.shape, channels)``. The 3-channel result is a strided
        view over a uint32 buffer; ``np.ascontiguousarray`` gives a packed copy.
    """
    num_channels = 4 if color_format is ColorFormat.COLOR_ALPHA else 3
    return color_ids[..., np.newaxis].view(np.uint8)[..., :num_channels]


def _get_image_color_format(image: npt.NDArray[Any]) -> ColorFormat:
    """Determine the channel layout of an image from its shape.

    Args:
      image (npt.NDArray[Any]): 2-D image or 3-D image with 3 or 4 channels.

    Returns:
      ColorFormat: The image's channel layout.
    """
    if image.ndim == 2:
        return ColorFormat.SINGLE_CHANNEL
    return ColorFormat.COLOR if image.shape[2] == 3 else ColorFormat.COLOR_ALPHA


def _label_dtype(num_colors: int) -> np.dtype[Any]:
    """Return the narrowest unsigned dtype that holds a label index for each of ``num_colors`` colors.

    Args:
      num_colors (int): Number of distinct colors.

    Returns:
      np.dtype[Any]: uint8 for up to 256 colors, uint16 for up to 65536, uint32 above that.
    """
    return np.min_scalar_type(max(num_colors - 1, 0))


@cache
def _numba_kernels() -> ModuleType | None:
    """Load the numba kernel module of the ``numba`` extra on first use.

    Returns:
      ModuleType | None: ``fastmask_seg._numba_kernels``, or None when numba is unavailable.
    """
    try:
        return importlib.import_module("fastmask_seg._numba_kernels")
    except ImportError:
        return None


def _fill_mask(mask: npt.NDArray[np.bool_], source: npt.NDArray[Any], value: Any, is_present: bool) -> None:
    """Write ``source == value`` into ``mask``, or all False for a value absent from ``source``.

    Args:
      mask (npt.NDArray[np.bool_]): Output mask with the shape of ``source``.
      source (npt.NDArray[Any]): Color ID image or label image to compare against.
      value (Any): Scalar with the dtype of ``source``.
      is_present (bool): False marks a value that cannot occur in ``source``.
    """
    if is_present:
        np.equal(source, value, out=mask)
    else:
        mask.fill(False)


def _cast_requests(
    requested: npt.NDArray[np.integer],
    dtype: np.dtype[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
    """Cast requested color IDs to an integer dtype, flagging the requests that dtype can hold.

    Every later comparison then runs within one integer dtype. Mixed signed and unsigned 64-bit operands would promote
    to float64, which merges IDs above 2**53. The cast values form a new array, so they stay intact when the request
    array shares storage with an output buffer.

    Args:
      requested (npt.NDArray[np.integer]): 1-D color IDs.
      dtype (np.dtype[Any]): Integer dtype to cast to.

    Returns:
      tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]: The IDs in ``dtype``, with 0 in place of IDs outside its range,
        and whether each ID is within that range.
    """
    limits = np.iinfo(dtype)
    in_range = (requested >= limits.min) & (requested <= limits.max)
    return np.where(in_range, requested, 0).astype(dtype), in_range


def _iter_masks(
    source: npt.NDArray[Any],
    values: npt.NDArray[Any],
    present: list[bool],
    out: npt.NDArray[np.bool_] | None,
) -> Iterator[npt.NDArray[np.bool_]]:
    """Yield one mask per comparison value.

    Args:
      source (npt.NDArray[Any]): Color ID image or label image to compare against.
      values (npt.NDArray[Any]): One comparison value per mask, with the dtype of ``source``.
      present (list[bool]): Whether each value can occur in ``source``.
      out (npt.NDArray[np.bool_] | None): Buffer that receives every mask in turn. None allocates each mask.

    Yields:
      npt.NDArray[np.bool_]: The mask for each value, in order.
    """
    for value, is_present in zip(values, present, strict=True):
        mask = out if out is not None else np.empty(source.shape, dtype=np.bool_)
        _fill_mask(mask, source, value, is_present)
        yield mask


class _Runs(NamedTuple):
    """Run-length structure of an image's color IDs in row-major order.

    Attributes:
      unique_ids (npt.NDArray[Any]): Sorted unique color IDs.
      labels (npt.NDArray[np.intp]): Index into ``unique_ids`` of each run's color.
      lengths (npt.NDArray[np.intp]): Pixel count of each run.
    """

    unique_ids: npt.NDArray[Any]
    labels: npt.NDArray[np.intp]
    lengths: npt.NDArray[np.intp]


class _KernelAnalysis(NamedTuple):
    """Colors, pixel counts, and label image of a 3-channel image, computed in one pass by the numba kernel.

    Attributes:
      unique_ids (npt.NDArray[np.uint32]): Sorted unique color IDs.
      counts (npt.NDArray[np.int64]): Pixel count of each entry of ``unique_ids``.
      labels (npt.NDArray[np.uint8]): ``(H, W)`` image of indices into ``unique_ids``.
    """

    unique_ids: npt.NDArray[np.uint32]
    counts: npt.NDArray[np.int64]
    labels: npt.NDArray[np.uint8]


class FastMask:
    """Unique colors and per-color boolean masks of one color-coded segmentation image.

    Derived arrays (color ID image, unique colors, pixel counts, label image) are computed on first access and cached,
    so repeated queries on the same image pay only for the comparisons. The object borrows ``image``, and the color ID
    image of a 2-D or 4-channel image shares its storage, so the image contents stay unchanged while the object
    is in use; a changed image takes a new ``FastMask``.

    Attributes:
      image (npt.NDArray[np.integer]): The source image, in the layout it was given.
      color_format (ColorFormat): Channel layout of ``image``.
      backend (Backend): Whether the numba kernels are used, see ``__init__``.
    """

    def __init__(
        self,
        image: npt.NDArray[np.integer],
        *,
        color_id_buffer: npt.NDArray[np.uint32] | None = None,
        backend: Backend = "auto",
    ) -> None:
        """Wrap an image for mask queries.

        Args:
          image (npt.NDArray[np.integer]): 3-channel or 4-channel uint8 image, or 2-D image of any integer dtype: a
            grayscale image, or a label image such as an instance-ID or object-index pass. C-contiguous images take
            the fastest paths.
          color_id_buffer (npt.NDArray[np.uint32] | None = None): C-contiguous little-endian uint32 buffer of shape
            ``(H, W)``, in storage separate from ``image``, that holds the color ID image of a 3-channel image, so a
            loop over same-sized images reuses one allocation. ``color_id_image`` is then this buffer, and two live
            instances sharing it overwrite each other's IDs. 2-D and 4-channel images derive their color ID image
            from ``image`` itself and ignore this buffer. The numba kernels read the image directly, and a call they
            serve while a buffer is set emits a ``UserWarning``.
          backend (Backend = "auto"): ``"auto"`` uses the kernels of the ``numba`` extra when it is installed, on
            3-channel images: one kernel computes the unique colors, pixel counts, and label image together in a single
            pass over a C-contiguous image with up to 256 colors, and another writes the masks of ``get_color_masks``
            and ``get_color_mask`` requests that pass ``colors`` or ``color_ids`` while the color ID image and label
            image are still uncomputed. NumPy computes everything else. ``"numpy"`` computes everything with NumPy.
        """
        self.image = image
        self.color_format = _get_image_color_format(self.image)
        self._color_id_buffer = color_id_buffer if self.color_format is ColorFormat.COLOR else None
        self.backend: Backend = backend

    @property
    def is_single_channel(self) -> bool:
        """Whether the image is single-channel."""
        return self.color_format is ColorFormat.SINGLE_CHANNEL

    @cached_property
    def color_id_image(self) -> npt.NDArray[Any]:
        """The image with every pixel replaced by its color ID.

        A 2-D image is its own color ID image, in its own dtype. Color images give uint32 IDs; for a 4-channel image
        with a contiguous channel axis this is a zero-copy view of ``image``, and for a 3-channel image it is
        ``color_id_buffer`` when one was given.
        """
        if self.is_single_channel:
            return self.image
        # 3-channel and 4-channel images are uint8.
        return colors_to_color_ids(cast("npt.NDArray[np.uint8]", self.image), out=self._color_id_buffer)

    @cached_property
    def _kernel_analysis(self) -> _KernelAnalysis | None:
        """Unique color IDs, pixel counts, and label image from the numba kernel, or None when NumPy computes them.

        The kernel serves C-contiguous 3-channel images with up to ``_KERNEL_MAX_COLORS`` colors when ``backend`` is
        ``"auto"`` and ``color_id_image`` is still uncomputed, so every derived array describes one snapshot of the
        image. It reads the image once, and its label image becomes the cached ``label_image``.
        """
        # cached_property stores a computed value in the instance __dict__ under the property's name.
        if (
            self.backend == "numpy"
            or self.color_format is not ColorFormat.COLOR
            or not self.image.flags.c_contiguous
            or "color_id_image" in self.__dict__
        ):
            return None
        kernels = _numba_kernels()
        if kernels is None:
            return None

        labels = np.empty(self.image.shape[:2], dtype=np.uint8)
        found_ids = np.empty(_KERNEL_MAX_COLORS, dtype=_ID_DTYPE)
        found_counts = np.empty(_KERNEL_MAX_COLORS, dtype=np.int64)
        num_colors = kernels.rgb_analyze(self.image, labels, found_ids, found_counts)
        if num_colors < 0:
            return None

        # The kernel numbers colors in order of first appearance; labels index the sorted IDs.
        order = np.argsort(found_ids[:num_colors])
        if np.any(order[1:] < order[:-1]):
            new_labels = np.empty(_KERNEL_MAX_COLORS, dtype=np.uint8)
            new_labels[order] = np.arange(num_colors, dtype=np.uint8)
            kernels.relabel(labels, new_labels)
        self.__dict__["label_image"] = labels
        self._warn_idle_color_id_buffer()
        return _KernelAnalysis(
            unique_ids=found_ids[:num_colors][order],
            counts=found_counts[:num_colors][order],
            labels=labels,
        )

    @cached_property
    def _runs(self) -> _Runs:
        """Run-length structure of the color ID image.

        Segmentation masks consist of few long runs of identical pixels, so deduplicating one value per run replaces
        sorting every pixel. The per-run results also give the pixel counts and the label image directly.
        """
        flat = self.color_id_image.reshape(-1)
        starts = np.flatnonzero(flat[1:] != flat[:-1])
        starts += 1
        if flat.size > 0:
            starts = np.concatenate((np.zeros(1, dtype=np.intp), starts))

        unique_ids, labels = np.unique(flat[starts], return_inverse=True)
        lengths = np.diff(starts, append=flat.size)
        return _Runs(unique_ids=unique_ids, labels=labels, lengths=lengths)

    @property
    def unique_color_ids(self) -> npt.NDArray[Any]:
        """The sorted unique color IDs present in the image, with the dtype of ``color_id_image``."""
        analysis = self._kernel_analysis
        return self._runs.unique_ids if analysis is None else analysis.unique_ids

    @cached_property
    def unique_colors(self) -> npt.NDArray[np.integer]:
        """The unique colors present in the image, in the image's channel layout and in ``unique_color_ids`` order.

        2-D images give a 1-D array of pixel values.
        """
        if self.is_single_channel:
            return self.unique_color_ids
        return color_ids_to_colors(self.unique_color_ids, self.color_format)

    @cached_property
    def unique_color_counts(self) -> npt.NDArray[np.int64]:
        """The pixel count of each entry of ``unique_color_ids``.

        Summed from the run lengths, which costs one pass over the runs. Useful for dropping the background or
        filtering out tiny instances before generating masks.
        """
        analysis = self._kernel_analysis
        if analysis is not None:
            return analysis.counts
        runs = self._runs
        counts = np.bincount(runs.labels, weights=runs.lengths, minlength=runs.unique_ids.size)
        return counts.astype(np.int64)

    @cached_property
    def label_image(self) -> npt.NDArray[Any]:
        """The image with every pixel replaced by its index into ``unique_color_ids``.

        Uses the narrowest unsigned dtype that holds every index (uint8 for up to 256 colors), so comparing against it
        reads a quarter of the bytes per pixel of the uint32 color ID image.
        """
        analysis = self._kernel_analysis
        if analysis is not None:
            return analysis.labels
        runs = self._runs
        labels = np.repeat(runs.labels.astype(_label_dtype(runs.unique_ids.size)), runs.lengths)
        return labels.reshape(self.image.shape[:2])

    def get_color_masks(
        self,
        *,
        color_ids: npt.NDArray[np.integer] | None = None,
        colors: npt.NDArray[np.integer] | None = None,
        out: npt.NDArray[np.bool_] | None = None,
    ) -> ColorMasks:
        """Generate a boolean mask for each requested color or color ID.

        Args:
          color_ids (npt.NDArray[np.integer] | None = None): 1-D integer color IDs to generate masks for.
          colors (npt.NDArray[np.integer] | None = None): Colors to generate masks for: uint8 channel values in the
            image's channel layout for color images, where a single color may be given as shape ``(3,)`` or ``(4,)``,
            or integer pixel values for 2-D images. Pass at most one of ``colors`` and ``color_ids``; when both are
            None, masks are generated for every entry of ``unique_color_ids``.
          out (npt.NDArray[np.bool_] | None = None): Bool buffer of shape ``(N, H, W)`` with ``N`` at least the number
            of requested masks, in storage separate from the image and the arrays derived from it. Masks are written to
            its first entries, so one buffer sized for the largest request serves a loop over images. None allocates a
            new array.

        Returns:
          ColorMasks: The masks, shape ``(num_masks, H, W)`` in request order, and ``num_masks``. A requested color
            absent from the image gives an all-False mask.
        """
        requested = self._resolve_color_ids(color_ids, colors)
        num_masks = requested.size
        if out is None:
            masks = np.empty((num_masks, *self.image.shape[:2]), dtype=np.bool_)
        else:
            masks = out[:num_masks]

        if self._write_kernel_masks(requested, masks):
            return ColorMasks(masks=masks, num_masks=num_masks)
        source, values, present = self._mask_source(requested)
        for mask, value, is_present in zip(masks, values, present, strict=True):
            _fill_mask(mask, source, value, is_present)
        return ColorMasks(masks=masks, num_masks=num_masks)

    def get_color_mask(
        self,
        *,
        color_id: int | None = None,
        color: npt.NDArray[np.integer] | np.integer | None = None,
        out: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Generate the boolean mask of one color or color ID.

        Args:
          color_id (int | None = None): Color ID to generate the mask for.
          color (npt.NDArray[np.integer] | np.integer | None = None): Color to generate the mask for: uint8 channel
            values of shape ``(3,)`` or ``(4,)`` in the image's channel layout for color images, or an integer pixel
            value (NumPy scalar or 0-d array) for 2-D images. Pass exactly one of ``color`` and ``color_id``.
          out (npt.NDArray[np.bool_] | None = None): Bool buffer of shape ``(H, W)``, with the same storage
            requirement as ``get_color_masks``, that receives the mask and is returned. None allocates a new array.

        Returns:
          npt.NDArray[np.bool_]: Mask of shape ``(H, W)``, all False when the color is absent from the image.
        """
        color_ids = None if color_id is None else np.array([color_id])
        colors = None if color is None else np.asarray(color)
        masks = self.get_color_masks(color_ids=color_ids, colors=colors, out=None if out is None else out[np.newaxis])
        return masks.masks[0] if out is None else out

    def iter_color_masks(
        self,
        *,
        color_ids: npt.NDArray[np.integer] | None = None,
        colors: npt.NDArray[np.integer] | None = None,
        out: npt.NDArray[np.bool_] | None = None,
    ) -> Iterator[npt.NDArray[np.bool_]]:
        """Return an iterator over the mask of each requested color, one at a time.

        Peak memory stays at one ``(H, W)`` mask, where ``get_color_masks`` holds all of them at once. A 4K image with
        40 instances needs about 330 MB as a single boolean stack. Masks are computed with NumPy.

        Args:
          color_ids (npt.NDArray[np.integer] | None = None): 1-D integer color IDs to generate masks for.
          colors (npt.NDArray[np.integer] | None = None): Colors to generate masks for: uint8 channel values in the
            image's channel layout for color images, or integer pixel values for 2-D images. Pass at most one of
            ``colors`` and ``color_ids``; when both are None, masks are generated for every entry of
            ``unique_color_ids``.
          out (npt.NDArray[np.bool_] | None = None): Bool buffer of shape ``(H, W)``, with the same storage
            requirement as ``get_color_masks``, that receives every mask in turn and is yielded each time, overwritten
            in place, so each mask must be consumed before the next is requested. None allocates each mask.

        Returns:
          Iterator[npt.NDArray[np.bool_]]: The ``(H, W)`` mask for each requested color, in request order.
        """
        source, values, present = self._mask_source(self._resolve_color_ids(color_ids, colors))
        return _iter_masks(source, values, present, out)

    def _resolve_color_ids(
        self,
        color_ids: npt.NDArray[np.integer] | None,
        colors: npt.NDArray[np.integer] | None,
    ) -> npt.NDArray[Any]:
        """Turn a mask request given as colors or as color IDs into a 1-D array of color IDs.

        Args:
          color_ids (npt.NDArray[np.integer] | None): 1-D integer color IDs, used when given.
          colors (npt.NDArray[np.integer] | None): uint8 colors in the image's channel layout, or pixel
            values of a 2-D image.

        Returns:
          npt.NDArray[Any]: 1-D color IDs in request order; ``unique_color_ids`` when both arguments are None.
        """
        if color_ids is not None:
            return color_ids
        if colors is None:
            return self.unique_color_ids
        if self.is_single_channel:
            return colors.reshape(-1)
        # Color images take uint8 channel values.
        return colors_to_color_ids(cast("npt.NDArray[np.uint8]", colors)).reshape(-1)

    def _write_kernel_masks(self, requested: npt.NDArray[Any], masks: npt.NDArray[np.bool_]) -> bool:
        """Write masks straight from a 3-channel image with the numba kernel when ``backend`` is ``"auto"``.

        The kernel reads the image once and skips building the uint32 color ID image. It serves 3-channel images whose
        color ID image and label image are both still uncomputed, since comparing against either one once it exists is
        cheaper.

        Args:
          requested (npt.NDArray[Any]): 1-D color IDs.
          masks (npt.NDArray[np.bool_]): Output of shape ``(requested.size, H, W)``.

        Returns:
          bool: True when the kernel wrote ``masks``, False when the caller computes them.
        """
        # cached_property stores a computed value in the instance __dict__ under the property's name.
        if (
            self.backend == "numpy"
            or self.color_format is not ColorFormat.COLOR
            or "color_id_image" in self.__dict__
            or "label_image" in self.__dict__
            # The kernel writes without bounds checks, so a mismatched buffer goes to NumPy, which raises on it.
            or masks.shape != (requested.size, *self.image.shape[:2])
        ):
            return False
        kernels = _numba_kernels()
        if kernels is None:
            return False

        in_range = (requested >= 0) & (requested <= _RGB_ID_MAX)
        kernel_ids = np.full(requested.shape, _NO_MATCH_ID, dtype=np.uint32)
        kernel_ids[in_range] = requested[in_range]
        kernels.rgb_masks(self.image, kernel_ids, masks)
        self._warn_idle_color_id_buffer()
        return True

    def _warn_idle_color_id_buffer(self) -> None:
        """Warn that a numba kernel computed a result straight from the image while a ``color_id_buffer`` was given.

        The kernels read the image directly, so the buffer stays untouched. The warning is attributed to the first
        calling frame outside this package, and the default warning filter shows it once per calling line.
        """
        if self._color_id_buffer is not None:
            frame = sys._getframe()
            stacklevel = 1
            while frame.f_back is not None and frame.f_code.co_filename.startswith(_WARNING_SKIP_PREFIXES):
                frame = frame.f_back
                stacklevel += 1
            warnings.warn(
                "color_id_buffer was left untouched: the numba kernel computed this result directly from the image. "
                'The buffer serves backend="numpy", images with more than 256 colors, and color_id_image.',
                UserWarning,
                stacklevel=stacklevel,
            )

    def _use_label_image(self, num_requested: int) -> bool:
        """Choose whether a mask query compares against the label image or the color ID image.

        One-byte 2-D images always compare against the image itself. Other images use the label image once it exists,
        and otherwise build it when its dtype is narrower than the color ID image and the query amortizes the build.
        For color images that holds when the run table is already computed (expanding it then costs less than one
        uint32 comparison) or enough masks are requested to pay for the run scan as well. For wider 2-D images, whose
        own comparisons are cheaper, it takes both: a computed run table and enough masks. Images with more than 65536
        colors have uint32 label images.

        Args:
          num_requested (int): Number of masks in the query.

        Returns:
          bool: True to compare against ``label_image``, False to compare against ``color_id_image``.
        """
        if self.is_single_channel and self.image.dtype.itemsize == 1:
            return False
        # cached_property stores a computed value in the instance __dict__ under the property's name.
        if "label_image" in self.__dict__:
            return True
        runs_computed = "_runs" in self.__dict__
        many_masks = num_requested >= _LABEL_PATH_MIN_MASKS
        if self.is_single_channel:
            build = runs_computed and many_masks
        else:
            build = runs_computed or many_masks
        if not build:
            return False
        num_colors = self.unique_color_ids.size
        # The numba analysis kernel caches its label image while computing the unique colors.
        if "label_image" in self.__dict__:
            return True
        return _label_dtype(num_colors).itemsize < self.color_id_image.dtype.itemsize

    def _mask_source(self, requested: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any], list[bool]]:
        """Return the image to compare against, the per-request comparison value, and which requests can match.

        Comparison values carry the source image's dtype, because comparing a narrow image against wider values makes
        numpy widen the whole image first. A request that cannot occur in the image is flagged so its mask is filled
        with False.

        Args:
          requested (npt.NDArray[Any]): 1-D color IDs.

        Returns:
          tuple[npt.NDArray[Any], npt.NDArray[Any], list[bool]]: The source image, one comparison value per request
            with the source's dtype, and one presence flag per request.
        """
        if self._use_label_image(requested.size):
            source = self.label_image
            unique_ids = self.unique_color_ids
            ids, present = _cast_requests(requested, unique_ids.dtype)
            ranks = np.searchsorted(unique_ids, ids)
            present &= ranks < unique_ids.size
            present[present] = unique_ids[ranks[present]] == ids[present]
            values = np.where(present, ranks, 0).astype(source.dtype)
            return source, values, present.tolist()

        source = self.color_id_image
        values, in_range = _cast_requests(requested, source.dtype)
        return source, values, in_range.tolist()
