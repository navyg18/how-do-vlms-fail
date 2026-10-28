"""
bbox_utils.py — Bounding box to vision-token mapping.
Requires qwen_vl_patch_utils.py (in this folder).
"""

try:
    from qwen_vl_patch_utils import get_object_patch_indices_qwenvl
    HAS_BBOX_UTILS = True
except ImportError:
    HAS_BBOX_UTILS = False
    print("Warning: qwen_vl_patch_utils not found. Bbox patching disabled.")


def bbox_to_token_positions(image, bbox, vision_idx, image_grid_thw):
    """
    Map [x, y, w, h] to (offsets_in_vision_span, absolute_seq_positions).
    Returns ([], []) if mapping fails.
    """
    if not HAS_BBOX_UTILS or bbox is None or not vision_idx:
        return [], []

    try:
        # Convert [x, y, w, h] (GQA/VG format) to [x1, y1, x2, y2]
        x, y, w, h = bbox
        bbox_xyxy = [x, y, x + w, y + h]
        indices = list(get_object_patch_indices_qwenvl(
            image=image,
            bbox=bbox_xyxy,
            start_end_indices=[(vision_idx[0], vision_idx[-1])],
            image_grid_thw=image_grid_thw,
            spatial_merge_size=2,
            buffer=0,
        ))
    except Exception:
        return [], []

    if not indices:
        return [], []

    vision_set = set(vision_idx)
    valid = [pos for pos in indices if pos in vision_set]
    if not valid:
        return [], []

    offsets = [vision_idx.index(pos) for pos in valid]
    return offsets, valid