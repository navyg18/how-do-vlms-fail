"""
Utilities for mapping bounding boxes to vision tokens in Qwen2.5-VL.

Key facts for Qwen2.5-VL (HF Transformers):
- The processor produces input_ids that contain a contiguous block of <|image_pad|> tokens.
- The processor also outputs image_grid_thw = [t, h, w] describing the *pre-merge* vision grid.
- The model uses spatial merging (e.g., 2x2) so the number of <|image_pad|> tokens equals:
    (h // spatial_merge_size) * (w // spatial_merge_size)

Therefore:
- DO NOT compute grids from raw image.size // patch_size.
- DO compute merged grid from image_grid_thw and spatial_merge_size.
- Map bbox pixels to merged grid using proportional mapping (image width/height -> grid_w/grid_h).
"""

from __future__ import annotations

from typing import List, Tuple, Set, Optional
import numpy as np
from PIL import Image


def _add_buffer(indices: Set[int],
                grid_w: int,
                grid_h: int,
                buffer_amount: int = 1) -> Set[int]:
    """Add 8-connected neighbor rings around indices."""
    if buffer_amount <= 0:
        return indices

    new_indices = set(indices)
    for _ in range(buffer_amount):
        buffer_indices = set()
        for idx in new_indices:
            row = idx // grid_w
            col = idx % grid_w
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = row + dr
                    cc = col + dc
                    if 0 <= rr < grid_h and 0 <= cc < grid_w:
                        buffer_indices.add(rr * grid_w + cc)
        new_indices |= buffer_indices

    return new_indices


def get_merged_grid_from_thw(
    image_grid_thw: List[int] | Tuple[int, int, int],
    spatial_merge_size: int = 2
) -> Tuple[int, int]:
    """
    Convert image_grid_thw=[t,h,w] (pre-merge grid) to merged grid (grid_h, grid_w).

    Example:
      image_grid_thw = [1, 42, 58], spatial_merge_size=2
      merged_h = 42//2 = 21
      merged_w = 58//2 = 29
      merged tokens = 21*29 = 609
    """
    if image_grid_thw is None or len(image_grid_thw) != 3:
        raise ValueError(f"image_grid_thw must be [t,h,w], got: {image_grid_thw}")

    _, h, w = image_grid_thw
    if spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be > 0, got {spatial_merge_size}")

    merged_h = h // spatial_merge_size
    merged_w = w // spatial_merge_size

    if merged_h <= 0 or merged_w <= 0:
        raise ValueError(
            f"Invalid merged grid from thw={image_grid_thw}, spatial_merge_size={spatial_merge_size}"
        )

    return merged_h, merged_w


def bbox_to_merged_patch_indices(
    bbox: List[int],
    image_size: Tuple[int, int],
    image_grid_thw: List[int] | Tuple[int, int, int],
    spatial_merge_size: int = 2,
    buffer: int = 0,
) -> List[int]:
    """
    Map bbox [x1,y1,x2,y2] in raw image pixels -> merged patch indices [0..grid_h*grid_w-1]
    using proportional mapping to merged grid derived from image_grid_thw.

    This assumes bbox coordinates correspond to the loaded PIL image (before processor resizing),
    which is what you have in Visual Genome.
    """
    x1, y1, x2, y2 = bbox
    W, H = image_size

    # Guard
    if W <= 0 or H <= 0:
        return []

    # Clamp bbox to image bounds (robustness)
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))

    # Ensure proper ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    grid_h, grid_w = get_merged_grid_from_thw(image_grid_thw, spatial_merge_size=spatial_merge_size)

    # Pixel size per merged cell
    cell_w = W / grid_w
    cell_h = H / grid_h

    # Convert bbox -> grid coords
    gx1 = int(x1 / cell_w)
    gx2 = int(x2 / cell_w)
    gy1 = int(y1 / cell_h)
    gy2 = int(y2 / cell_h)

    # Clamp to grid bounds
    gx1 = max(0, min(gx1, grid_w - 1))
    gx2 = max(0, min(gx2, grid_w - 1))
    gy1 = max(0, min(gy1, grid_h - 1))
    gy2 = max(0, min(gy2, grid_h - 1))

    patches = set()
    for gy in range(gy1, gy2 + 1):
        for gx in range(gx1, gx2 + 1):
            patches.add(gy * grid_w + gx)

    if buffer > 0:
        patches = _add_buffer(patches, grid_w=grid_w, grid_h=grid_h, buffer_amount=buffer)

    return sorted(patches)


def get_object_patch_indices_qwenvl(
    image: Image.Image,
    bbox: List[int],
    start_end_indices: List[Tuple[int, int]],
    image_grid_thw: List[int] | Tuple[int, int, int],
    spatial_merge_size: int = 2,
    buffer: int = 0,
) -> List[int]:
    """
    Return *actual token indices* (in input_ids / hidden_states sequence positions)
    corresponding to bbox patches for Qwen2.5-VL.

    - start_end_indices: [(vision_start, vision_end)] for the contiguous <|image_pad|> block
      (inclusive indices).
    - image_grid_thw: from processor outputs (inputs["image_grid_thw"][0].tolist()).
    """
    if not start_end_indices:
        return []

    start, end = start_end_indices[0]
    total_tokens = end - start + 1

    # Compute merged grid size and sanity-check against token count
    grid_h, grid_w = get_merged_grid_from_thw(image_grid_thw, spatial_merge_size=spatial_merge_size)
    expected_tokens = grid_h * grid_w

    # If mismatch, still proceed but warn via return trimming.
    # (You can turn this into an assert once stable.)
    # assert expected_tokens == total_tokens, f"grid tokens {expected_tokens} != pad tokens {total_tokens}"

    patch_indices = bbox_to_merged_patch_indices(
        bbox=bbox,
        image_size=image.size,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=spatial_merge_size,
        buffer=buffer,
    )

    # Map merged patch indices -> actual token indices
    actual = [start + p for p in patch_indices]

    # Clamp into [start,end]
    actual = [i for i in actual if start <= i <= end]

    return actual


def visualize_patches_on_image(
    image: Image.Image,
    merged_patch_indices: List[int],
    image_grid_thw: List[int] | Tuple[int, int, int],
    spatial_merge_size: int = 2,
    output_path: Optional[str] = None
):
    """
    Debug visualization: draw merged grid cells (as rectangles) that correspond to patch indices.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    grid_h, grid_w = get_merged_grid_from_thw(image_grid_thw, spatial_merge_size=spatial_merge_size)
    W, H = image.size
    cell_w = W / grid_w
    cell_h = H / grid_h

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    for p in merged_patch_indices:
        r = p // grid_w
        c = p % grid_w
        x = c * cell_w
        y = r * cell_h

        rect = mpatches.Rectangle(
            (x, y),
            cell_w,
            cell_h,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.axis("off")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved patch visualization to {output_path}")
    else:
        plt.show()
    plt.close()


def get_random_patch_token_indices(
    start_end_indices: List[Tuple[int, int]],
    n_patches: int,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Control: sample random token indices inside the <|image_pad|> span.
    """
    if not start_end_indices:
        return []

    start, end = start_end_indices[0]
    total = end - start + 1
    n = min(n_patches, total)

    rng = np.random.default_rng(seed)
    offsets = rng.choice(total, size=n, replace=False)
    offsets.sort()

    return [start + int(o) for o in offsets]
