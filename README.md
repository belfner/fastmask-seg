# fastmask-seg

Unique colors, pixel counts, per-color boolean masks, and bounding boxes from segmentation masks stored as images.

```python
import cv2
import numpy as np
import fastmask_seg as fms

image = cv2.imread("instance_mask.png")  # (H, W, 3) uint8, one solid color per instance

result = fms.unique_color_masks(image)
result.colors        # (K, 3) uint8 colors present, in BGR order because OpenCV loads BGR
result.color_counts  # (K,) int64 pixel count per color
result.masks         # (K, H, W) bool, one mask per color

road = fms.get_color_mask(image, color=np.array([128, 64, 128], np.uint8))  # (H, W) bool
boxes = fms.masks_to_bboxes(result.masks)  # (K, 4) int64 rows of (rmin, rmax, cmin, cmax)
```

The library targets instance and semantic segmentation masks, where each class or instance is one solid color. Its fast paths rely on what those images look like: few distinct colors in long runs of identical pixels.

## Install

```bash
pip install fastmask-seg            # NumPy only
pip install "fastmask-seg[numba]"   # adds compiled kernels for 3-channel images
```

Requires Python 3.12+ and NumPy 2.0+. The `numba` extra gives the same results as NumPy, with color analysis 1.3-2x faster and masks of named colors 2-4x faster ([details](https://github.com/belfner/fastmask-seg/blob/main/docs/performance.md)).

## API

```python
fm = fms.FastMask(image)             # computes on first access and caches, so later queries reuse the analysis
fm.unique_colors                     # (K, C) uint8 colors in the image's channel layout, ordered by color ID
fm.unique_color_ids                  # (K,) sorted color IDs
fm.unique_color_counts               # (K,) int64 pixel counts
fm.label_image                       # (H, W) index of each pixel's color in unique_color_ids
fm.color_id_image                    # (H, W) color ID of each pixel
fm.get_color_masks(colors=palette)   # ColorMasks(masks, num_masks), one mask per requested color
fm.get_color_masks()                 # every color in the image
fm.get_color_mask(color=c)           # one (H, W) bool mask
fm.iter_color_masks()                # one (H, W) mask at a time, for many colors on large images

# Each FastMask member is also a function that takes an image or a FastMask:
fms.unique_colors(image)
fms.get_color_masks(image, colors=palette)
fms.unique_color_masks(image)        # UniqueColorMasks(colors, color_ids, color_counts, masks, num_masks)

fms.masks_to_bboxes(masks)           # (K, 4) boxes of a (K, H, W) bool stack
fms.mask_to_bbox(mask)               # (rmin, rmax, cmin, cmax) of one mask

fms.colors_to_color_ids(colors)      # (..., 3) or (..., 4) uint8 colors to uint32 IDs
fms.color_ids_to_colors(ids, fms.ColorFormat.COLOR)
fms.sort_colors(colors)              # colors in color ID order
```

- **Images** are uint8 arrays of shape `(H, W)`, `(H, W, 3)`, or `(H, W, 4)`, as `cv2.imread` or `np.asarray(PIL.Image.open(...))` give them. In a grayscale image the gray value is the color.
- **Colors** follow the image's channel order: BGR for OpenCV, RGB for Pillow. A color ID packs a color into one uint32 with channel 0 in the lowest byte.
- **Mask requests** take `colors=` or `color_ids=`, and with neither they cover every color in the image. A color absent from the image gives an all-False mask.
- **Bounding boxes** have exclusive ends, so `mask[rmin:rmax, cmin:cmax]` crops a mask to its pixels. An empty mask raises `ValueError`.
- **Functions and `FastMask`:** a function given an image analyzes it from scratch on each call. Wrap the image in a `FastMask` to ask several questions about one image, and pass that object to the functions when you prefer their form.
- **Arguments are used as given.** The docstrings state each required shape and dtype, and other inputs surface as NumPy errors or wrong results.

## Processing many images

Allocate the mask buffer once and pass it as `out=`, so each image writes into memory the process already holds:

```python
masks_buf = np.empty((64, H, W), dtype=bool)  # room for up to 64 colors per image

for path in paths:
    result = fms.unique_color_masks(cv2.imread(path), out=masks_buf)
    ...  # result.masks is masks_buf[:result.num_masks], overwritten by the next image
```

For a fixed class palette, pass the palette straight to `get_color_masks(image, colors=palette, out=...)`; with the `numba` extra this is the fastest path. [docs/performance.md](https://github.com/belfner/fastmask-seg/blob/main/docs/performance.md) covers the backends, the NumPy-path `color_id_buffer`, caching, and benchmarks.

## Development

```bash
make dev     # uv sync --dev
make check   # ruff, isort, mypy --strict
make format  # isort + ruff format
```

## License

MIT
