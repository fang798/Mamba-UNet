"""
Generate ring masks and ring-masked images from existing masks.

Example:
python /home/hit/fx/Mamba-UNet/code/gen_ring_mask_and_image.py \
  --radius 17 --mask_dir /home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/mask_all_255
"""

import argparse
import os
from typing import List

import cv2
import numpy as np

LABELED_IMG_DIR = "/home/hit/fx/network/thyroid/nodule_compare/classifier/first_batch_roi2/image"
UNLABELED_IMG_DIR = "/home/hit/fx/network/thyroid/nodule_compare/classifier/second_batch/img_roi"
IMG_EXTS = [".png", ".jpg", ".jpeg"]


def _find_image_path(base_name: str, search_dirs: List[str]) -> str:
    for root in search_dirs:
        for ext in IMG_EXTS:
            candidate = os.path.join(root, base_name + ext)
            if os.path.exists(candidate):
                return candidate
    return ""


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_ring(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask_u8 > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    ring = cv2.subtract(dilated, eroded)
    ring[ring > 0] = 255
    return ring


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Directory containing 0/255 masks",
    )
    parser.add_argument(
        "--image_dirs",
        type=str,
        default="",
        help="Comma separated image dirs; defaults to batch1+batch2 dirs",
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--radius", type=int, default=13)
    parser.add_argument("--out_mask_dir", type=str, default="")
    parser.add_argument("--out_image_dir", type=str, default="")
    args = parser.parse_args()

    if args.image_dirs:
        image_dirs = [p.strip() for p in args.image_dirs.split(",") if p.strip()]
    else:
        image_dirs = [LABELED_IMG_DIR, UNLABELED_IMG_DIR]

    base_out = os.path.dirname(os.path.abspath(args.mask_dir))
    mask_ring_dir = args.out_mask_dir or os.path.join(base_out, "mask_ring")
    image_ring_dir = args.out_image_dir or os.path.join(base_out, "image_ring")
    _ensure_dir(mask_ring_dir)
    _ensure_dir(image_ring_dir)

    mask_files = [
        f
        for f in os.listdir(args.mask_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]

    for mask_name in mask_files:
        base = os.path.splitext(mask_name)[0]
        mask_path = os.path.join(args.mask_dir, mask_name)
        img_path = _find_image_path(base, image_dirs)
        if not img_path:
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if mask is None or image is None:
            continue

        mask = cv2.resize(mask, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_LINEAR)

        mask = (mask > 0).astype(np.uint8) * 255
        ring = _make_ring(mask, args.radius)

        ring_mask_path = os.path.join(mask_ring_dir, f"{base}.png")
        cv2.imwrite(ring_mask_path, ring)

        ring_image = cv2.bitwise_and(image, image, mask=ring)
        ring_image_path = os.path.join(image_ring_dir, f"{base}.png")
        cv2.imwrite(ring_image_path, ring_image)

    print(f"Saved ring masks to: {mask_ring_dir}")
    print(f"Saved ring images to: {image_ring_dir}")


if __name__ == "__main__":
    main()
