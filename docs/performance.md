# Performance and backends

## Benchmarks

Median time per image on six 4096x3072 (12.6 MP) instance masks with 2-11 colors each, one pinned core of an Intel i7-12850HX, a new `FastMask` per image. NumPy reuses a `color_id_buffer`; the masks go into a reused `out=`.

| Operation | NumPy | `numba` extra |
| --- | --- | --- |
| `unique_colors` and `unique_color_counts` | 10.7 ms | 7.8 ms |
| `label_image` | 11.6 ms | 7.7 ms |
| Masks of every color | 18.5 ms | 14.2 ms |
| Mask of 1 named color | 8.8 ms | 4.2 ms |
| Masks of 2 named colors | 11.2 ms | 4.8 ms |

Images with up to 256 colors run the analysis kernel. Above 256 colors the analysis falls back to NumPy after the kernel's attempt, which adds about 4 ms on such an image.

## Backend

`FastMask(image, backend=...)` and the functional mask calls take `backend`:

| `backend` | Behavior |
| --- | --- |
| `"auto"` (default) | The numba kernels below on 3-channel images when the `numba` extra is installed; NumPy for everything else. |
| `"numpy"` | NumPy for everything, and numba stays unimported. |

| Kernel | Used for | Applies when |
| --- | --- | --- |
| Analysis | `unique_colors`, `unique_color_ids`, `unique_color_counts`, `label_image`, and through them masks of every color and `unique_color_masks` | The image is C-contiguous and has up to 256 colors. |
| Masks | `get_color_masks` and `get_color_mask` calls that pass `colors` or `color_ids` | The color ID image and label image are still uncached, as on a new `FastMask` or a functional call on an image. |

Once the label image is cached, masks compare against it with NumPy at about 1 ms per mask. The functional analysis calls (`unique_colors(image)` and the like) run with `"auto"`; pass `FastMask(image, backend="numpy")` to keep them on NumPy. A functional call given a `FastMask` uses that object's `backend` and `color_id_buffer`. Each kernel compiles on first use and caches its machine code on disk for later processes.

## Usage patterns

- **Fixed palette:** `get_color_masks(image, colors=palette, out=...)` on each new image. One call covers the whole palette in one pass over the image, where separate `get_color_mask` calls on a new `FastMask` each pay that pass.
- **Every color (instance masks):** `unique_color_masks(image, out=...)`.
- **Inspect, then choose:** read `fm.unique_color_counts`, then request the masks you want from the same `fm`; they compare against the cached label image.
- **More than 256 colors:** `backend="numpy"` skips the kernel attempt.

## Buffers

| Argument | Shape and dtype | Behavior |
| --- | --- | --- |
| `get_color_masks(out=...)`, `unique_color_masks(out=...)` | `(N, H, W)` bool, `N` at least the number of masks | Masks fill `out[:num_masks]`; the result's `masks` is that view. |
| `get_color_mask(out=...)` | `(H, W)` bool | Receives the mask and is returned. |
| `iter_color_masks(out=...)` | `(H, W)` bool | Yielded for every mask and overwritten in place; consume each mask before requesting the next. |
| `color_id_buffer=` | `(H, W)` uint32, C-contiguous | Holds the color ID image NumPy builds from a 3-channel image. |
| `colors_to_color_ids(out=...)` | `colors.shape[:-1]` uint32, C-contiguous | Receives the IDs and is returned. |

Buffers occupy storage separate from the image and from each other. Results written into a buffer are views of it and stay valid until the buffer is reused.

`color_id_buffer` serves the NumPy path: `backend="numpy"`, NumPy-only installs, and images with more than 256 colors. 2-D and 4-channel images derive their IDs from the image itself and ignore it. The numba kernels read the image directly, and a call they serve while a `color_id_buffer` is set emits a `UserWarning`. Two live `FastMask` objects sharing one `color_id_buffer` overwrite each other's IDs.

```python
ids_buf = np.empty((H, W), dtype=np.uint32)
result = fms.unique_color_masks(image, out=masks_buf, color_id_buffer=ids_buf, backend="numpy")
```

## Caching

`FastMask(image)` records the image and its channel layout and computes nothing else. Each derived array is computed on first access and cached on the instance; the analysis kernel computes the unique colors, counts, and label image together. A `FastMask` borrows its image, and the color ID image of a 2-D or 4-channel image shares its storage, so keep the image unchanged while the object is in use and wrap a changed image in a new `FastMask`.

A function given an image builds a temporary `FastMask`, computes what that call needs, and discards it. Given a `FastMask`, it uses and fills that object's cache. `iter_color_masks` prepares its comparison image when called and computes each mask as the iterator reaches it.
