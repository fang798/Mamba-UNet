"""
Extract handcrafted features (compute_mask_features / compute_image_features / compute_unclear)
with data loading & augmentations matching train_vim_dual_cls_image_ring.py and
transforms matching train_all7_continue.py.

Each epoch重新做随机增广，因此同一张图会产生多行特征（更鲁棒）。

Example:
python /home/hit/fx/Mamba-UNet/code/extract_cranet_dense_features_v2.py \
  --epochs 50 \
  --batch_size 32 \
  --target_cols "4" \
  --num_workers 4 \
  --num_threads 4 \
  --mask_dir /home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/mask_all_255 \
  --out_csv /home/hit/fx/Mamba-UNet/run_result/cranet_dense_features.csv
"""

import argparse
import csv
import os
import random
import sys
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# --- sys.path hacks for custom modules ---
CRANET_DIR = "/home/hit/fx/network/thyroid/thyroid_compare/TRFE-Net-for-thyroid-nodule-segmentation-main"
if CRANET_DIR not in sys.path:
    sys.path.insert(0, CRANET_DIR)
TRAD_DIRS = [
    "/home/hit/fx/network/thyroid/thyroid_compare/TRFE-Net-for-thyroid-nodule-segmentation-main/Traditional_Feature_Extraction_main",
    "/home/hit/fx/network/thyroid/nodule_seg_compare/TRFE-Net-for-thyroid-nodule-segmentation-main/Traditional_Feature_Extraction_main",
    "/home/hit/fx/network/nodule_seg_class/Traditional_Feature_Extraction_main",
]
for _d in TRAD_DIRS:
    if os.path.isdir(_d):
        parent = os.path.dirname(_d)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        break

import importlib.util

trforms_path = os.path.join(CRANET_DIR, "dataloaders", "onehot_custom_transforms.py")
spec = importlib.util.spec_from_file_location("onehot_custom_transforms", trforms_path)
trforms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trforms)  # type: ignore

CRANET_DENSE_PATH = os.path.join(CRANET_DIR, "model", "CRANet_dense.py")
spec2 = importlib.util.spec_from_file_location("cranet_dense_module", CRANET_DENSE_PATH)
cranet_dense = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(cranet_dense)  # type: ignore
compute_mask_features = cranet_dense.compute_mask_features
compute_image_features = cranet_dense.compute_image_features
compute_unclear = cranet_dense.compute_unclear

ROOT_DIR = "/home/hit/fx/network/thyroid/nodule_compare/classifier"
LABEL_FILE1 = os.path.join(ROOT_DIR, "batch1_single_nodule_class2_with_result.xlsx")
LABEL_FILE2 = os.path.join(ROOT_DIR, "second_batch/batch2_single_nodule_class2_with_result.xlsx")
IMG_DIR1 = "/home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/image_ring"
IMG_DIR2 = "/home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/image_ring"


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_label_df(path: str, batch_id: int) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=1, header=None)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    df.set_index(df.columns[0], inplace=True)
    df["batch"] = batch_id
    return df


def _find_first_existing(base_dir: str, candidates: List[str]) -> str:
    for name in candidates:
        full_path = os.path.join(base_dir, name)
        if os.path.exists(full_path):
            return full_path
    return ""


def _find_mask_path(mask_dir: str, base: str) -> str:
    for ext in [".png", ".jpg", ".jpeg"]:
        cand = os.path.join(mask_dir, base + ext)
        if os.path.exists(cand):
            return cand
    return ""


class JointTransform:
    """Apply transforms to image & mask jointly using the class_* ops from train_all7_continue (224x224)."""

    def __init__(self):
        self.tf = transforms.Compose(
            [
                trforms.class_FixedResize(size=(224, 224)),
                trforms.class_RandomBrightnessContrast(
                    brightness_range=(-20, 20),
                    contrast_range=(0.9, 1.1),
                ),
                trforms.class_RandomRotate(30),
                trforms.class_RandomAffine(translate=(0.2, 0.2), scale=(0.8, 1.2)),
                trforms.class_RandomHorizontalFlip(),
                trforms.class_RandomVerticalFlip(),
                trforms.class_ToTensor(),
                trforms.class_Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = {"image": image, "label": mask, "cls_labels": torch.zeros(1), "mask_cls": torch.zeros(1)}
        out = self.tf(sample)
        return out["image"], out["label"]


class NoduleFeatureDataset(Dataset):
    def __init__(
        self,
        mask_dir: str,
        target_cols: Tuple[int, ...],
        seed: int = 42,
    ) -> None:
        self.mask_dir = mask_dir
        self.target_cols = target_cols
        self.transform = JointTransform()

        df1 = _read_label_df(LABEL_FILE1, batch_id=1)
        df2 = _read_label_df(LABEL_FILE2, batch_id=2)
        self.label_cols = [c for c in df1.columns if c != "batch"]
        if not self.label_cols:
            self.label_cols = [c for c in df2.columns if c != "batch"]
        self.label_dim = len(self.label_cols)

        all_files_1 = list(df1.index)
        all_files_2 = list(df2.index)

        self.imgs = []
        self.masks = []
        self.labels = []
        self.names = []

        for img_name in all_files_1:
            img_name_str = str(img_name)
            base_raw = img_name_str
            base_no_prefix = img_name_str.removeprefix("1_")
            candidates = [base_raw + ext for ext in [".png", ".jpg", ".jpeg"]] + [
                base_no_prefix + ext for ext in [".png", ".jpg", ".jpeg"]
            ]
            candidates = list(dict.fromkeys(candidates))
            img_path = _find_first_existing(IMG_DIR1, candidates)
            if not img_path:
                continue
            mask_path = _find_mask_path(mask_dir, os.path.splitext(os.path.basename(img_path))[0])
            if not mask_path:
                continue
            cls_labels = df1.loc[img_name_str].drop("batch", errors="ignore").values.astype(int)
            if max(self.target_cols) >= len(cls_labels):
                continue
            targets = cls_labels.astype(np.float32)
            self.imgs.append(img_path)
            self.masks.append(mask_path)
            self.labels.append(targets)
            self.names.append(os.path.basename(img_path))

        for img_name in all_files_2:
            img_name_str = str(img_name)
            base_raw = img_name_str
            base_no_prefix = img_name_str.removeprefix("2_")
            candidates = [base_raw + "_nroi" + ext for ext in [".png", ".jpg", ".jpeg"]] + [
                base_no_prefix + "_nroi" + ext for ext in [".png", ".jpg", ".jpeg"]
            ]
            candidates += [base_raw + ext for ext in [".png", ".jpg", ".jpeg"]] + [
                base_no_prefix + ext for ext in [".png", ".jpg", ".jpeg"]
            ]
            candidates = list(dict.fromkeys(candidates))
            img_path = _find_first_existing(IMG_DIR2, candidates)
            if not img_path:
                continue
            mask_path = _find_mask_path(mask_dir, os.path.splitext(os.path.basename(img_path))[0])
            if not mask_path:
                continue
            cls_labels = df2.loc[img_name_str].drop("batch", errors="ignore").values.astype(int)
            if max(self.target_cols) >= len(cls_labels):
                continue
            targets = cls_labels.astype(np.float32)
            self.imgs.append(img_path)
            self.masks.append(mask_path)
            self.labels.append(targets)
            self.names.append(os.path.basename(img_path))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)
        image_tensor, mask_tensor = self.transform(img, mask)
        # convert mask to logits (before sigmoid) to match compute_mask_features expectation
        mask_logits = (mask_tensor * 2.0 - 1.0) * 10.0  # logits ~ very confident 0/1
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        name = self.names[idx]
        return image_tensor, mask_logits, label, name


def _align_inputs_for_cranet(
    image_tensor: torch.Tensor, mask_logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align inputs to match train_all7_continue.py expectations."""
    image_tensor = torch.as_tensor(image_tensor).float()
    mask_logits = torch.as_tensor(mask_logits).float()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if mask_logits.dim() == 3:
        mask_logits = mask_logits.unsqueeze(0)

    if image_tensor.size(1) == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)
    if mask_logits.dim() == 4 and mask_logits.size(1) != 1:
        mask_logits = mask_logits[:, :1, ...]

    max_val = float(image_tensor.max().item())
    min_val = float(image_tensor.min().item())
    if max_val > 2.0:
        image_tensor = image_tensor / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
    elif min_val >= 0.0 and max_val <= 1.0:
        image_tensor = (image_tensor - 0.5) / 0.5

    if mask_logits.min().item() >= 0.0 and mask_logits.max().item() <= 1.0:
        mask_logits = (mask_logits * 2.0 - 1.0) * 10.0

    return image_tensor, mask_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--target_cols", type=str, default="4")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="/home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/mask_all_255",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="/home/hit/fx/Mamba-UNet/run_result/cranet_dense_features.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_seed(args.seed)
    target_cols = tuple(int(x) for x in args.target_cols.split(",") if x.strip() != "")

    dataset = NoduleFeatureDataset(
        mask_dir=args.mask_dir,
        target_cols=target_cols,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    header = ["epoch", "image_name"]
    header += [f"label{i + 1}" for i in range(dataset.label_dim)]
    header += [f"mask_f{i}" for i in range(6)]
    header += [f"image_f{i}" for i in range(11)]
    header += [f"unclear_f{i}" for i in range(3)]

    write_header = not os.path.exists(args.out_csv)
    bad_log = os.path.splitext(args.out_csv)[0] + "_bad.txt"
    with open(args.out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        def _extract_one(idx: int, epoch: int, names, images, masks, labels):
            err_msgs = []
            img_i = images[idx : idx + 1]
            mask_i = masks[idx : idx + 1]

            try:
                mask_feat_list = compute_mask_features(mask_i)
                if isinstance(mask_feat_list, list) and len(mask_feat_list) > 0:
                    mask_feat = mask_feat_list[0]
                else:
                    mask_feat = [0.0] * 6
                    err_msgs.append(f"[mask_features] epoch={epoch} name={names[idx]} err=invalid_return")
            except Exception as exc:
                mask_feat = [0.0] * 6
                err_msgs.append(f"[mask_features] epoch={epoch} name={names[idx]} err={repr(exc)}")

            if not isinstance(mask_feat, (list, tuple)) or len(mask_feat) != 6:
                mask_feat = [0.0] * 6

            try:
                img_feat = compute_image_features(img_i, mask_i)[0]
            except Exception as exc:
                img_feat = [0.0] * 11
                err_msgs.append(f"[image_features] epoch={epoch} name={names[idx]} err={repr(exc)}")

            try:
                unc_feat = compute_unclear(img_i, mask_i)[0]
            except Exception as exc:
                unc_feat = [0.0] * 3
                err_msgs.append(f"[unclear_features] epoch={epoch} name={names[idx]} err={repr(exc)}")

            label_vals = labels[idx].detach().cpu().view(-1).tolist()
            row = [epoch, names[idx]]
            row += [float(x) for x in label_vals]
            row += [float(x) for x in mask_feat]
            row += [float(x) for x in img_feat]
            row += [float(x) for x in unc_feat]
            return row, err_msgs

        with ThreadPoolExecutor(max_workers=max(1, args.num_threads)) as executor:
            for epoch in range(1, args.epochs + 1):
                for image_tensor, mask_logits, label, name in tqdm(
                    loader,
                    desc=f"Epoch {epoch}/{args.epochs}",
                    total=len(loader),
                ):
                    image_tensor, mask_logits = _align_inputs_for_cranet(
                        image_tensor, mask_logits
                    )

                    futures = [
                        executor.submit(
                            _extract_one,
                            i,
                            epoch,
                            name,
                            image_tensor,
                            mask_logits,
                            label,
                        )
                        for i in range(len(name))
                    ]
                    for fut in futures:
                        row, err_msgs = fut.result()
                        if err_msgs:
                            with open(bad_log, "a") as bf:
                                for msg in err_msgs:
                                    bf.write(msg + "\n")
                        writer.writerow(row)

    print(f"Saved features to: {args.out_csv}")
    print(f"Bad log (if any errors): {bad_log}")


if __name__ == "__main__":
    main()
