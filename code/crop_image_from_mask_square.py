"""
Crop images using a square bbox from ring masks (dilate/erode).

Example:
python /home/hit/fx/Mamba-UNet/code/crop_image_from_mask_square.py \
  --mask_dir /home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/mask_all_255 \
  --radius 17 \
  --out_dir /home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/crop_image

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


def _make_ring(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask_u8 > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    ring = cv2.subtract(dilated, eroded)
    ring[ring > 0] = 255
    return ring


def _square_bbox(x_min: int, y_min: int, x_max: int, y_max: int, w: int, h: int) -> List[int]:
    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1
    side = max(box_w, box_h)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side

    if x0 < 0:
        x1 = min(w, x1 - x0)
        x0 = 0
    if y0 < 0:
        y1 = min(h, y1 - y0)
        y0 = 0
    if x1 > w:
        x0 = max(0, x0 - (x1 - w))
        x1 = w
    if y1 > h:
        y0 = max(0, y0 - (y1 - h))
        y1 = h

    return [x0, y0, x1, y1]


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
    parser.add_argument("--radius", type=int, default=17)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/crop_image",
        help="Directory to save cropped images",
    )
    args = parser.parse_args()

    if args.image_dirs:
        image_dirs = [p.strip() for p in args.image_dirs.split(",") if p.strip()]
    else:
        image_dirs = [LABELED_IMG_DIR, UNLABELED_IMG_DIR]

    os.makedirs(args.out_dir, exist_ok=True)

    mask_files = [
        f for f in os.listdir(args.mask_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]

    total = 0
    saved = 0
    missing_img = 0
    empty_mask = 0

    for mask_name in mask_files:
        total += 1
        base = os.path.splitext(mask_name)[0]
        mask_path = os.path.join(args.mask_dir, mask_name)
        img_path = _find_image_path(base, image_dirs)
        if not img_path:
            missing_img += 1
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if mask is None or image is None:
            missing_img += 1
            continue

        mask = cv2.resize(mask, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_LINEAR)

        mask = (mask > 0).astype(np.uint8) * 255
        ring = _make_ring(mask, args.radius)

        ys, xs = np.where(ring > 0)
        if len(xs) == 0 or len(ys) == 0:
            empty_mask += 1
            continue

        h, w = image.shape[:2]
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        x0, y0, x1, y1 = _square_bbox(x_min, y_min, x_max, y_max, w, h)
        crop = image[y0:y1, x0:x1]

        save_path = os.path.join(args.out_dir, f"{base}.png")
        cv2.imwrite(save_path, crop)
        saved += 1

    print(
        f"Done. total={total}, saved={saved}, missing_img={missing_img}, empty_mask={empty_mask}"
    )


if __name__ == "__main__":
    main()
