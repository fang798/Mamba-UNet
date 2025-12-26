'''
# model1 = cranet
python /home/hit/fx/Mamba-UNet/code/test_Semi_Mamba_UNet_thyroid.py \
  --ckpt /home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_cranet_m2_vim/model1_iter_4800_dice_0.8715.pth \
  --use_model model1 --model1_type cranet --gpus "3"
python /home/hit/fx/Mamba-UNet/code/test_Semi_Mamba_UNet_thyroid.py \
  --ckpt /home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_cranet_m2_cranet/model1_iter_9400_dice_0.8811.pth \
  --use_model model1 --model1_type cranet --gpus "3"
# model2 = vim
python /home/hit/fx/Mamba-UNet/code/test_Semi_Mamba_UNet_thyroid.py \
  --ckpt /home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/model2_iter_10000_dice_0.8885.pth \
  --use_model model2 --model2_type vim --gpus "3" --load_pretrain
'''

import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import get_config
from networks.unet import UNet
from networks.vision_mamba import MambaUnet as ViM_seg
from networks.vision_transformer import SwinUnet as ViT_seg
import importlib.util

ROOT_DIR = "/home/hit/fx/network/thyroid/nodule_compare/classifier"
LABELED_EXCEL = os.path.join(ROOT_DIR, "batch1_single_nodule_class2_with_result.xlsx")
UNLABELED_EXCEL = os.path.join(ROOT_DIR, "second_batch/batch2_single_nodule_class2_with_result.xlsx")
LABELED_IMG_DIR = os.path.join(ROOT_DIR, "first_batch_roi2/image")
LABELED_MASK_DIR = os.path.join(ROOT_DIR, "first_batch_roi2/mask")
UNLABELED_IMG_DIR = os.path.join(ROOT_DIR, "second_batch/img_roi")


def _read_label_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=1, header=None)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    df.set_index(df.columns[0], inplace=True)
    return df


def _find_first_existing(base_dir: str, candidates: List[str]) -> str:
    for name in candidates:
        full_path = os.path.join(base_dir, name)
        if os.path.exists(full_path):
            return full_path
    return ""


def _find_mask_path(base_dir: str, base_name: str) -> str:
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(base_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return ""


def _collect_labeled_val(seed: int) -> List[Dict[str, str]]:
    labeled_df = _read_label_df(LABELED_EXCEL)
    all_files = [str(x) for x in labeled_df.index]
    _, val_files = train_test_split(all_files, test_size=0.2, random_state=seed)
    samples = []
    for img_name_str in val_files:
        candidates = [img_name_str.removeprefix("1_") + ext for ext in [".png", ".jpg", ".jpeg"]]
        img_path = _find_first_existing(LABELED_IMG_DIR, candidates)
        if not img_path:
            continue
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = _find_mask_path(LABELED_MASK_DIR, base)
        samples.append({"image": img_path, "mask": mask_path, "is_labeled": True})
    return samples


def _collect_unlabeled() -> List[Dict[str, str]]:
    unlabeled_df = _read_label_df(UNLABELED_EXCEL)
    samples = []
    for img_name in unlabeled_df.index:
        img_name_str = str(img_name)
        candidates = [img_name_str + "_nroi" + ext for ext in [".png", ".jpg", ".jpeg"]]
        img_path = _find_first_existing(UNLABELED_IMG_DIR, candidates)
        if not img_path:
            continue
        samples.append({"image": img_path, "mask": "", "is_labeled": False})
    return samples


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    # Handle DataParallel prefix.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def _dice_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    dice = (2.0 * inter) / (pred_sum + gt_sum + 1e-8)
    iou = inter / (pred_sum + gt_sum - inter + 1e-8)
    return float(dice), float(iou)


def _predict_mask(model: torch.nn.Module, image_gray: np.ndarray, patch_size: int, device: str) -> np.ndarray:
    h, w = image_gray.shape[:2]
    image_rs = cv2.resize(image_gray, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    image_rs = image_rs.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_rs).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
    pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return pred


def _draw_contour(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = image_bgr.copy()
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)
    return vis


def _load_cranet_class():
    cranet_path = "/home/hit/fx/network/thyroid/thyroid_compare/TRFE-Net-for-thyroid-nodule-segmentation-main/model/CRANet.py"
    spec = importlib.util.spec_from_file_location("cranet_module", cranet_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CRANet


def _build_model(model_type: str, args, config: "CfgNode") -> torch.nn.Module:
    if model_type == "unet":
        return UNet(in_chns=1, class_num=args.num_classes)
    if model_type == "cranet":
        CRANet = _load_cranet_class()

        class CRANetWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = CRANet()

            def forward(self, x):
                if x.size(1) == 1:
                    x = x.repeat(1, 3, 1, 1)
                _, _, _, lateral_map_2 = self.core(x)
                return torch.cat([-lateral_map_2, lateral_map_2], dim=1)

        return CRANetWrapper()
    if model_type == "swin":
        config.defrost()
        config.DATA.IMG_SIZE = args.patch_size
        config.freeze()
        model = ViT_seg(config, img_size=args.patch_size, num_classes=args.num_classes)
        if args.load_pretrain:
            model.load_from(config)
        return model
    model = ViM_seg(config, img_size=args.patch_size, num_classes=args.num_classes)
    if args.load_pretrain:
        model.load_from(config)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to trained checkpoint")
    parser.add_argument("--use_model", type=str, default="model1", choices=["model1", "model2"])
    parser.add_argument("--model1_type", type=str, default="vim", choices=["unet", "vim", "swin", "cranet"])
    parser.add_argument("--model2_type", type=str, default="vim", choices=["unet", "vim", "swin", "cranet"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default=None, help="Comma separated GPU ids, e.g. \"0,2\"")
    parser.add_argument("--cfg", type=str, default="../code/configs/vmamba_tiny.yaml")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"])
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation-steps", type=int, default=0)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", type=str, default="", choices=["", "O0", "O1", "O2"])
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--model", type=str, default="mambaunet", help="model name for snapshot path")
    parser.add_argument("--load_pretrain", action="store_true", help="load pretrain weights before ckpt")
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    ckpt_path = os.path.abspath(args.ckpt)
    snapshot_path = ckpt_path if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)
    os.makedirs(snapshot_path, exist_ok=True)
    save_root = os.path.join(snapshot_path, "test")
    os.makedirs(save_root, exist_ok=True)
    metrics_path = os.path.join(snapshot_path, "test_metrics.txt")

    config = get_config(args)

    model_type = args.model2_type if args.use_model == "model2" else args.model1_type
    model = _build_model(model_type, args, config)

    model = model.to(args.device)
    model.eval()
    _load_checkpoint(model, args.ckpt)

    labeled_samples = _collect_labeled_val(args.seed)
    unlabeled_samples = _collect_unlabeled()
    all_samples = labeled_samples + unlabeled_samples

    dice_list = []
    iou_list = []

    for item in all_samples:
        img_path = item["image"]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        pred = _predict_mask(model, img_gray, args.patch_size, args.device)
        vis = _draw_contour(img_bgr, pred)

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(save_root, f"vis_{img_name}.png")
        cv2.imwrite(save_path, vis)

        if item["is_labeled"] and item["mask"] and os.path.exists(item["mask"]):
            gt = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            gt = (gt >= 128).astype(np.uint8)
            dice, iou = _dice_iou(pred > 0, gt > 0)
            dice_list.append(dice)
            iou_list.append(iou)

    if dice_list:
        mean_dice = float(np.mean(dice_list))
        mean_iou = float(np.mean(iou_list))
    else:
        mean_dice = 0.0
        mean_iou = 0.0

    with open(metrics_path, "w") as f:
        f.write(f"mean_dice\t{mean_dice:.6f}\n")
        f.write(f"mean_iou\t{mean_iou:.6f}\n")

    print(f"Saved visualizations to: {save_root}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
