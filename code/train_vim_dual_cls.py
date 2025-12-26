'''
python /home/hit/fx/Mamba-UNet/code/train_vim_dual_cls.py \
  --ckpt /home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/model2_iter_10000_dice_0.8885.pth \
  --load_pretrain \
  --gpus "3" \
  --batch_size 32 \
  --image_size 224 \
  --epochs 120 \
  --warmup_epochs 5 \
  --freeze_epochs 70 \
  --head_lr 1e-3 \
  --backbone_lr 1e-5 \
  --filter_interval 5 \
  --base_drop 100 \
  --increase_per_round 20 \
  --save_dir "/home/hit/fx/Mamba-UNet/run_result/vim_dual_cls"

--filter_interval是每多少个epoch进行一次样本过滤
--base_drop是每次过滤时，基础丢弃样本数
--increase_per_round是每次过滤时，随着轮次增加的丢弃样本数
'''

import argparse
import math
import os
import random
import sys
from typing import List, Tuple

_gpus_arg = None
if "--gpus" in sys.argv:
    _idx = sys.argv.index("--gpus")
    if _idx + 1 < len(sys.argv):
        _gpus_arg = sys.argv[_idx + 1]
else:
    for _arg in sys.argv:
        if _arg.startswith("--gpus="):
            _gpus_arg = _arg.split("=", 1)[1]
            break
if _gpus_arg:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus_arg

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from networks.vision_mamba import MambaUnet as ViM_seg

ROOT_DIR = "/home/hit/fx/network/thyroid/nodule_compare/classifier"
LABEL_FILE1 = os.path.join(ROOT_DIR, "batch1_single_nodule_class2_with_result.xlsx")
LABEL_FILE2 = os.path.join(ROOT_DIR, "second_batch/batch2_single_nodule_class2_with_result.xlsx")
IMG_DIR1 = os.path.join(ROOT_DIR, "first_batch_roi2/image")
IMG_DIR2 = os.path.join(ROOT_DIR, "second_batch/img_roi")

def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


class NoduleDualDataset(Dataset):
    def __init__(
        self,
        mode: str,
        transform=None,
        seed: int = 42,
        target_cols: Tuple[int, int] = (4, 5),
    ) -> None:
        self.mode = mode
        self.transform = transform
        self.target_cols = target_cols

        df1 = _read_label_df(LABEL_FILE1, batch_id=1)
        df2 = _read_label_df(LABEL_FILE2, batch_id=2)

        all_files_1 = list(df1.index)
        all_files_2 = list(df2.index)
        train_1, val_1 = train_test_split(all_files_1, test_size=0.2, random_state=seed)
        train_2, val_2 = train_test_split(all_files_2, test_size=0.2, random_state=seed)

        if mode == "train":
            selected = [(name, 1) for name in train_1] + [(name, 2) for name in train_2]
        elif mode in ("val", "test"):
            selected = [(name, 1) for name in val_1] + [(name, 2) for name in val_2]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.imgs = []
        self.labels = []
        self.names = []

        for img_name, batch_id in selected:
            img_name_str = str(img_name)
            if batch_id == 1:
                candidates = [img_name_str.removeprefix("1_") + ext for ext in [".png", ".jpg", ".jpeg"]]
                img_path = _find_first_existing(IMG_DIR1, candidates)
            else:
                candidates = [img_name_str + "_nroi" + ext for ext in [".png", ".jpg", ".jpeg"]]
                img_path = _find_first_existing(IMG_DIR2, candidates)

            if not img_path:
                continue

            if batch_id == 1:
                cls_labels = df1.loc[img_name_str].drop("batch", errors="ignore").values.astype(int)
            else:
                cls_labels = df2.loc[img_name_str].drop("batch", errors="ignore").values.astype(int)
            if max(self.target_cols) >= len(cls_labels):
                continue
            targets = cls_labels[list(self.target_cols)].astype(np.float32)

            self.imgs.append(img_path)
            self.labels.append(targets)
            self.names.append(os.path.basename(img_path))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img_path = self.imgs[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        targets = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {"image": image, "targets": targets, "label_name": self.names[idx]}


class AttributeBranch(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class VimDualHead(nn.Module):
    def __init__(self, config, load_pretrain: bool = False) -> None:
        super().__init__()
        self.backbone = ViM_seg(config, img_size=config.DATA.IMG_SIZE, num_classes=2)
        if load_pretrain:
            self.backbone.load_from(config)
        self.feature_dim = self.backbone.mamba_unet.num_features
        self.head1 = AttributeBranch(self.feature_dim)
        self.head2 = AttributeBranch(self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        feat, _ = self.backbone.mamba_unet.forward_features(x)
        feat = feat.mean(dim=(1, 2))
        logit1 = self.head1(feat).view(-1)
        logit2 = self.head2(feat).view(-1)
        return torch.stack([logit1, logit2], dim=1)


def _load_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def _compute_pos_weights(labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_np = np.stack(labels, axis=0)
    pos_weights = []
    pos_counts = []
    neg_counts = []
    for col in range(labels_np.shape[1]):
        col_vals = labels_np[:, col]
        col_vals = col_vals[col_vals != -1]
        pos = int((col_vals == 1).sum())
        neg = int((col_vals == 0).sum())
        pos_counts.append(pos)
        neg_counts.append(neg)
        if pos == 0 or neg == 0:
            pos_weights.append(1.0)
        else:
            pos_weights.append(float(neg) / float(pos))
    return (
        np.array(pos_weights, dtype=np.float32),
        np.array(pos_counts, dtype=np.int64),
        np.array(neg_counts, dtype=np.int64),
    )


def compute_confusion(all_labels: list, all_preds: list) -> np.ndarray:
    if not all_labels:
        return np.zeros((2, 2), dtype=int)
    labels = np.concatenate(all_labels).astype(int)
    preds = np.concatenate(all_preds).astype(int)
    return confusion_matrix(labels, preds, labels=[0, 1])


def balanced_accuracy_from_conf(conf_matrix: np.ndarray) -> float:
    if conf_matrix.size < 4:
        return 0.0
    tn, fp, fn, tp = conf_matrix.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)


def save_accuracy_to_txt(
    total_accuracy: float,
    average_loss: float,
    balanced_accuracy: float,
    conf_matrix: np.ndarray,
    filename: str,
    learning_rate: float = None,
    prefix: str = "",
    roc_auc: float = None,
) -> None:
    with open(filename, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        if prefix:
            f.write(f"{prefix}\n")
        f.write(f"Accuracy: {total_accuracy * 100:.2f}%\n")
        f.write(f"Balanced accuracy: {balanced_accuracy:.4f}\n")
        f.write(f"Average loss: {average_loss:.4f}\n")
        if learning_rate is not None:
            f.write(f"Learning rate: {learning_rate:.6f}\n")
        if roc_auc is not None:
            f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, conf_matrix, fmt="%d")
        f.write("=" * 50 + "\n")


class LossFilter:
    def __init__(self, base_drop: int = 30, increase_per_round: int = 10, drop_range_ratio: float = 1.5):
        self.base_drop = base_drop
        self.increase_per_round = increase_per_round
        self.drop_range_ratio = drop_range_ratio
        self.round = 0
        self.all_losses: List[float] = []
        self.sample_names: List[str] = []

    def reset(self) -> None:
        self.all_losses.clear()
        self.sample_names.clear()

    def record_loss(self, sample_losses: torch.Tensor, image_names) -> None:
        if isinstance(image_names, (list, tuple)):
            names = [str(n) for n in image_names]
        elif image_names is None:
            names = []
        else:
            names = [str(image_names)]

        losses = sample_losses.detach().cpu().view(-1).numpy()
        if len(names) < len(losses):
            extra = [f"auto_{len(self.sample_names) + i}" for i in range(len(losses) - len(names))]
            names.extend(extra)
        elif len(names) > len(losses):
            names = names[: len(losses)]

        self.all_losses.extend(losses)
        self.sample_names.extend(names)

    def filter_out_bad_samples(self, dataset):
        if not self.all_losses:
            return dataset

        losses = np.array(self.all_losses)
        sample_names = np.array(self.sample_names)
        losses = np.nan_to_num(losses, nan=999.0, posinf=999.0, neginf=999.0)

        num_to_drop = self.base_drop + self.round * self.increase_per_round
        self.round += 1

        top_n = min(len(losses), int(num_to_drop * self.drop_range_ratio))
        if top_n <= 0:
            return dataset

        sorted_indices = np.argsort(losses)[::-1]
        top_indices = sorted_indices[:top_n]
        num_to_drop = min(num_to_drop, len(top_indices))
        drop_indices = np.random.choice(top_indices, size=num_to_drop, replace=False)
        bad_names = set(sample_names[drop_indices])

        if isinstance(dataset, Subset):
            original_dataset = dataset.dataset
            current_indices = dataset.indices
        else:
            original_dataset = dataset
            current_indices = list(range(len(dataset)))

        def _sample_name(idx: int) -> str:
            try:
                path = original_dataset.imgs[idx]
            except (AttributeError, IndexError):
                return str(idx)
            return os.path.basename(path)

        good_indices = [idx for idx in current_indices if _sample_name(idx) not in bad_names]
        if not good_indices:
            return dataset

        return Subset(original_dataset, good_indices)


def adjust_pos_weight_from_conf(
    conf_matrix: np.ndarray,
    base_pos_weight: float,
    min_scale: float = 0.8,
    max_scale: float = 1.2,
) -> float:
    if conf_matrix is None or conf_matrix.size < 4:
        return base_pos_weight
    tn, fp, fn, tp = conf_matrix.ravel()
    pos_total = tp + fn
    neg_total = tn + fp
    if pos_total <= 0 or neg_total <= 0:
        return base_pos_weight

    tpr = tp / pos_total
    tnr = tn / neg_total
    bias = tpr - tnr
    scale = 1.0 - 2 * bias
    scale = max(min_scale, min(max_scale, scale))
    return base_pos_weight * scale


def adjust_pos_weight_from_bias(
    mean_pred: float,
    mean_label: float,
    base_pos_weight: float,
    min_scale: float = 0.8,
    max_scale: float = 1.2,
) -> Tuple[float, float, float]:
    """Adjust pos_weight using probability bias (mean_pred - mean_label)."""
    if base_pos_weight <= 0:
        return base_pos_weight, 1.0, 0.0
    if mean_label < 0 or mean_pred < 0:
        return base_pos_weight, 1.0, 0.0
    bias = mean_pred - mean_label
    scale = 1.0 - 2.0 * bias
    scale = max(min_scale, min(max_scale, scale))
    return base_pos_weight * scale, scale, bias


def compute_pair_losses(
    scores: torch.Tensor,
    targets: torch.Tensor,
    margin_loss_fn: nn.MarginRankingLoss,
    pair_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pos_mask = targets > 0.5
    neg_mask = targets <= 0.5
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return None, None

    pos_expanded = pos_scores.unsqueeze(1).expand(-1, neg_scores.size(0)).reshape(-1)
    neg_expanded = neg_scores.unsqueeze(0).expand(pos_scores.size(0), -1).reshape(-1)
    margin_targets = torch.ones_like(pos_expanded)

    diff = pos_expanded - neg_expanded
    loss_rank = -torch.log(torch.sigmoid(diff)).mean()
    loss_margin = margin_loss_fn(pos_expanded, neg_expanded, margin_targets)

    if pair_weight is not None:
        loss_rank = loss_rank * pair_weight
        loss_margin = loss_margin * pair_weight

    return loss_rank, loss_margin


def evaluate_dataset_for_filter(
    model: nn.Module,
    dataset,
    loss_filters: List[LossFilter],
    pos_weight: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[List[np.ndarray], List[Tuple[float, float, int]]]:
    model.eval()
    for lf in loss_filters:
        lf.reset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    all_preds = [[], []]
    all_labels = [[], []]
    prob_sums = [0.0, 0.0]
    label_sums = [0.0, 0.0]
    label_counts = [0, 0]
    with torch.no_grad():
        for sample in tqdm(loader, desc="Loss filter eval", leave=False, file=sys.stdout):
            inputs = sample["image"].to(device)
            targets = sample["targets"].to(device)

            logits = model(inputs)
            names = sample.get("label_name")

            for idx in range(logits.size(1)):
                batch_size = logits.size(0)
                sample_losses = torch.zeros(batch_size, device=device)
                valid_mask = targets[:, idx] != -1
                if valid_mask.any():
                    scores = logits[valid_mask, idx]
                    targets_t = targets[valid_mask, idx]
                    loss_vals = F.binary_cross_entropy_with_logits(
                        scores, targets_t, pos_weight=pos_weight[idx], reduction="none"
                    )
                    sample_losses[valid_mask] = loss_vals
                loss_filters[idx].record_loss(sample_losses, names)

            probs = torch.sigmoid(logits)
            for idx in range(probs.size(1)):
                valid_mask = targets[:, idx] != -1
                if not valid_mask.any():
                    continue
                preds = (probs[valid_mask, idx] > 0.5).float()
                labels = targets[valid_mask, idx]
                all_preds[idx].append(preds.cpu().numpy())
                all_labels[idx].append(labels.cpu().numpy())
                prob_sums[idx] += probs[valid_mask, idx].sum().item()
                label_sums[idx] += labels.sum().item()
                label_counts[idx] += int(valid_mask.sum().item())

    confs = [compute_confusion(all_labels[i], all_preds[i]) for i in range(2)]
    stats = []
    for idx in range(2):
        count = label_counts[idx]
        if count > 0:
            mean_pred = prob_sums[idx] / count
            mean_label = label_sums[idx] / count
        else:
            mean_pred = 0.0
            mean_label = 0.0
        stats.append((mean_pred, mean_label, count))
    return confs, stats


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="VIM checkpoint for encoder")
    parser.add_argument("--gpus", type=str, default=None, help='Comma separated GPU ids, e.g. "0,2"')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cfg", type=str, default="../code/configs/vmamba_tiny.yaml")
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"])
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation-steps", type=int, default=0)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", type=str, default="", choices=["", "O0", "O1", "O2"])
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup_epochs", type=int, default=6)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--filter_interval", type=int, default=5)
    parser.add_argument("--base_drop", type=int, default=50)
    parser.add_argument("--increase_per_round", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="/home/hit/fx/Mamba-UNet/run_result/vim_dual_cls")
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--target_cols", type=str, default="4,5")
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    setup_seed(args.seed)

    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = args.image_size
    config.freeze()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    target_cols = tuple(int(x) for x in args.target_cols.split(","))

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    train_data = NoduleDualDataset(mode="train", transform=train_transform, seed=args.seed, target_cols=target_cols)
    val_data = NoduleDualDataset(mode="val", transform=val_transform, seed=args.seed, target_cols=target_cols)
    full_train_size = len(train_data)

    trainloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True
    )
    testloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    pos_weights, pos_counts, neg_counts = _compute_pos_weights(train_data.labels)
    base_pos_weight = pos_weights.copy()
    current_pos_weight = pos_weights.copy()
    pos_weight_tensor = torch.tensor(current_pos_weight, device=device)
    pair_weight = torch.tensor(base_pos_weight, device=device)

    margin_loss_fn = nn.MarginRankingLoss(margin=1.0, reduction="mean")
    alpha_bce = 0.5
    alpha_rank = 1.0
    alpha_margin = 0.8

    model = VimDualHead(config=config, load_pretrain=args.load_pretrain).to(device)
    _load_checkpoint(model.backbone, args.ckpt)

    head_params = list(model.head1.parameters()) + list(model.head2.parameters())
    backbone_params = list(model.backbone.parameters())
    optimizer = optim.Adam(
        [
            {"params": head_params, "lr": args.head_lr},
            {"params": backbone_params, "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "acc.txt")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("epoch,train_loss,val_accuracy,val_loss\n")
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Run args: {vars(args)}\n")
        f.write(
            "pos_counts: {}, neg_counts: {}, pos_weights: {}\n".format(
                pos_counts.tolist(), neg_counts.tolist(), pos_weights.tolist()
            )
        )
        f.write("=" * 50 + "\n")

    best_acc = [0.5, 0.5]
    best_balanced = [0.5, 0.5]
    loss_filters = [
        LossFilter(base_drop=args.base_drop, increase_per_round=args.increase_per_round),
        LossFilter(base_drop=args.base_drop, increase_per_round=args.increase_per_round),
    ]

    total_epochs = args.epochs
    warmup_epochs = args.warmup_epochs

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / warmup_epochs
        return 0.5 * (
            1
            + math.cos(math.pi * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    backbone_scheduler = None

    class BackboneScheduler:
        def __init__(self, optimizer, group_index: int, base_lr: float, total_epochs: int) -> None:
            self.optimizer = optimizer
            self.group_index = group_index
            self.base_lr = base_lr
            self.total_epochs = max(1, total_epochs)
            self.epoch = 0

        def _lr_scale(self, epoch: int) -> float:
            if self.total_epochs <= 1:
                return 1.0
            return 0.5 * (1 + math.cos(math.pi * epoch / (self.total_epochs - 1)))

        def step(self) -> None:
            lr = self.base_lr * self._lr_scale(self.epoch)
            self.optimizer.param_groups[self.group_index]["lr"] = lr
            self.epoch += 1

    for epoch in range(args.epochs):
        if epoch < args.freeze_epochs:
            _set_requires_grad(model.backbone, False)
            optimizer.param_groups[1]["lr"] = 0.0
        else:
            _set_requires_grad(model.backbone, True)
            if backbone_scheduler is None:
                backbone_total_epochs = max(1, args.epochs - args.freeze_epochs)
                backbone_scheduler = BackboneScheduler(
                    optimizer=optimizer,
                    group_index=1,
                    base_lr=args.backbone_lr,
                    total_epochs=backbone_total_epochs,
                )
            backbone_scheduler.step()

        head_lr_used = optimizer.param_groups[0]["lr"]
        backbone_lr_used = optimizer.param_groups[1]["lr"]

        model.train()
        running_loss = [0.0, 0.0]
        task_counts = [0, 0]
        processed_batches = 0
        train_bar = tqdm(trainloader, file=sys.stdout)
        for sample in train_bar:
            inputs = sample["image"].to(device)
            targets = sample["targets"].to(device)

            logits = model(inputs)
            task_losses = []
            for idx in range(logits.size(1)):
                valid_mask = targets[:, idx] != -1
                if not valid_mask.any():
                    continue
                scores = logits[valid_mask, idx]
                targets_t = targets[valid_mask, idx]
                loss_bce = F.binary_cross_entropy_with_logits(
                    scores, targets_t, pos_weight=pos_weight_tensor[idx], reduction="mean"
                )
                loss_rank, loss_margin = compute_pair_losses(
                    scores, targets_t, margin_loss_fn=margin_loss_fn, pair_weight=float(pair_weight[idx])
                )
                if loss_rank is None or loss_margin is None:
                    task_loss = alpha_bce * loss_bce
                else:
                    task_loss = alpha_bce * loss_bce + alpha_rank * loss_rank + alpha_margin * loss_margin
                task_losses.append(task_loss)
                running_loss[idx] += task_loss.item()
                task_counts[idx] += 1

            if not task_losses:
                continue
            loss = sum(task_losses) / len(task_losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            processed_batches += 1
            train_bar.set_description(f"[RankNet] Epoch {epoch + 1} Loss {loss.item():.4f}")

        if processed_batches == 0:
            print(f"[RankNet] Epoch {epoch + 1}: no valid samples in training data, skipping update.")
            scheduler.step()
            with open(log_file, "a") as f:
                f.write(
                    f"Epoch {epoch + 1}/{args.epochs}, skipped_no_valid=1, "
                    f"head_lr: {head_lr_used:.6f}, backbone_lr: {backbone_lr_used:.6f}\n"
                )
            continue

        train_loss = [
            running_loss[i] / max(1, task_counts[i]) if task_counts[i] else 0.0 for i in range(2)
        ]
        scheduler.step()

        model.eval()
        total_loss = [0.0, 0.0]
        val_counts = [0, 0]
        processed_batches = 0
        all_preds = [[], []]
        all_labels = [[], []]
        all_probs = [[], []]
        correct = [0, 0]
        total_samples = [0, 0]
        with torch.no_grad():
            for sample in testloader:
                inputs = sample["image"].to(device)
                targets = sample["targets"].to(device)

                logits = model(inputs)
                task_losses = []
                for idx in range(logits.size(1)):
                    valid_mask = targets[:, idx] != -1
                    if not valid_mask.any():
                        continue
                    scores = logits[valid_mask, idx]
                    targets_t = targets[valid_mask, idx]
                    loss_bce = F.binary_cross_entropy_with_logits(
                        scores, targets_t, pos_weight=pos_weight_tensor[idx], reduction="mean"
                    )
                    loss_rank, loss_margin = compute_pair_losses(
                        scores, targets_t, margin_loss_fn=margin_loss_fn, pair_weight=float(pair_weight[idx])
                    )
                    if loss_rank is None or loss_margin is None:
                        task_loss = alpha_bce * loss_bce
                    else:
                        task_loss = alpha_bce * loss_bce + alpha_rank * loss_rank + alpha_margin * loss_margin
                    task_losses.append(task_loss)
                    total_loss[idx] += task_loss.item()
                    val_counts[idx] += 1

                if not task_losses:
                    continue

                probs = torch.sigmoid(logits)
                processed_batches += 1

                for idx in range(probs.size(1)):
                    valid_mask = targets[:, idx] != -1
                    if not valid_mask.any():
                        continue
                    preds = (probs[valid_mask, idx] > 0.5).float()
                    labels = targets[valid_mask, idx]
                    correct[idx] += (preds == labels).sum().item()
                    total_samples[idx] += labels.numel()
                    all_preds[idx].append(preds.cpu().numpy())
                    all_labels[idx].append(labels.cpu().numpy())
                    all_probs[idx].append(probs[valid_mask, idx].cpu().numpy())

        if processed_batches == 0:
            val_loss = [0.0, 0.0]
            val_acc = [0.0, 0.0]
            val_balanced = [0.0, 0.0]
            val_roc = [0.0, 0.0]
            confs = [np.zeros((2, 2), dtype=int), np.zeros((2, 2), dtype=int)]
        else:
            val_loss = [
                total_loss[i] / max(1, val_counts[i]) if val_counts[i] else 0.0 for i in range(2)
            ]
            confs = [compute_confusion(all_labels[i], all_preds[i]) for i in range(2)]
            val_balanced = [balanced_accuracy_from_conf(c) for c in confs]
            val_acc = [
                (correct[i] / total_samples[i]) if total_samples[i] else 0.0 for i in range(2)
            ]
            val_roc = []
            for idx in range(2):
                try:
                    y_true = np.concatenate(all_labels[idx]).astype(int)
                    y_score = np.concatenate(all_probs[idx]).astype(float)
                    val_roc.append(float(roc_auc_score(y_true, y_score)))
                except Exception:
                    val_roc.append(0.0)

        with open(log_file, "a") as f:
            head_lr = optimizer.param_groups[0]["lr"]
            f.write(
                f"Epoch {epoch + 1}/{args.epochs}, "
                f"train_loss0: {train_loss[0]:.4f}, train_loss1: {train_loss[1]:.4f}, "
                f"val_loss0: {val_loss[0]:.4f}, val_loss1: {val_loss[1]:.4f}, "
                f"val_acc0: {val_acc[0]:.4f}, val_acc1: {val_acc[1]:.4f}, "
                f"val_bal0: {val_balanced[0]:.4f}, val_bal1: {val_balanced[1]:.4f}, "
                f"val_roc0: {val_roc[0]:.4f}, val_roc1: {val_roc[1]:.4f}, "
                f"head_lr: {head_lr_used:.6f}, backbone_lr: {backbone_lr_used:.6f}\n"
            )

        save_accuracy_to_txt(
            total_accuracy=val_acc[0],
            average_loss=val_loss[0],
            balanced_accuracy=val_balanced[0],
            conf_matrix=confs[0],
            filename=log_file,
            learning_rate=optimizer.param_groups[0]["lr"],
            prefix="Task 0",
            roc_auc=val_roc[0],
        )
        save_accuracy_to_txt(
            total_accuracy=val_acc[1],
            average_loss=val_loss[1],
            balanced_accuracy=val_balanced[1],
            conf_matrix=confs[1],
            filename=log_file,
            learning_rate=optimizer.param_groups[0]["lr"],
            prefix="Task 1",
            roc_auc=val_roc[1],
        )

        improved = False
        for idx in range(2):
            if val_acc[idx] > best_acc[idx] or val_balanced[idx] > best_balanced[idx]:
                best_acc[idx] = max(best_acc[idx], val_acc[idx])
                best_balanced[idx] = max(best_balanced[idx], val_balanced[idx])
                improved = True
        if improved:
            save_path = os.path.join(
                save_dir,
                f"vim_cls_epoch_{epoch + 1}_acc0_{val_acc[0]:.4f}_acc1_{val_acc[1]:.4f}.pth",
            )
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % args.filter_interval == 0:
            before_filter_size = len(trainloader.dataset)
            confs_train, bias_stats = evaluate_dataset_for_filter(
                model=model,
                dataset=train_data,
                loss_filters=loss_filters,
                pos_weight=pos_weight_tensor,
                device=device,
                batch_size=args.batch_size,
            )
            new_pos_weights = []
            scales = []
            biases = []
            counts = []
            for idx, stat in enumerate(bias_stats):
                mean_pred, mean_label, count = stat
                new_val, scale, bias = adjust_pos_weight_from_bias(
                    mean_pred, mean_label, base_pos_weight[idx], min_scale=0.8, max_scale=1.2
                )
                new_pos_weights.append(new_val)
                scales.append(scale)
                biases.append(bias)
                counts.append(count)

            current_pos_weight = np.array(new_pos_weights, dtype=np.float32)
            pos_weight_tensor = torch.tensor(current_pos_weight, device=device)

            filtered_dataset = train_data
            for lf in loss_filters:
                filtered_dataset = lf.filter_out_bad_samples(filtered_dataset)
            if len(filtered_dataset) == 0:
                filtered_dataset = train_data
            trainloader = DataLoader(
                filtered_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True,
                pin_memory=True,
            )
            with open(log_file, "a") as f:
                f.write(
                    f"[Filter] Epoch {epoch + 1}: train_size={len(filtered_dataset)}/{full_train_size}, "
                    f"prev_size={before_filter_size}, "
                    f"pos_weight={current_pos_weight.tolist()}, "
                    f"bias={['{:.4f}'.format(b) for b in biases]}, "
                    f"scale={['{:.4f}'.format(s) for s in scales]}, "
                    f"count={counts}\n"
                )


if __name__ == "__main__":
    main()
