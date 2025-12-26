"""
Train AttentionDeepMIL-style model on thyroid patches.

Example:
python /home/hit/fx/Mamba-UNet/code/train_mil_attention_deepmil.py \
  --gpus "2" \
  --epochs 100 \
  --lr 5e-3 \
  --reg 1e-4 \
  --image_size 224 \
  --patch_grid 4 \
  --target_cols "4"
"""

import argparse
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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

ROOT_DIR = "/home/hit/fx/network/thyroid/nodule_compare/classifier"
LABEL_FILE1 = os.path.join(ROOT_DIR, "batch1_single_nodule_class2_with_result.xlsx")
LABEL_FILE2 = os.path.join(ROOT_DIR, "second_batch/batch2_single_nodule_class2_with_result.xlsx")
IMG_DIR1 = "/home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/image_ring"
IMG_DIR2 = "/home/hit/fx/Mamba-UNet/run_result/mambaunet_m1_vim_m2_vim/image_ring"


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)

        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = torch.softmax(A, dim=1)

        Z = torch.mm(A, H)

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().data.item()
        return error, Y_hat

    def calculate_objective(self, X, Y, pos_weight: float = 1.0):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            pos_weight * Y * torch.log(Y_prob) + (1.0 - Y) * torch.log(1.0 - Y_prob)
        )
        return neg_log_likelihood, A


class NoduleMilBags(data_utils.Dataset):
    def __init__(
        self,
        mode: str,
        image_size: int = 224,
        patch_grid: int = 4,
        target_cols: Tuple[int, ...] = (4,),
        seed: int = 42,
    ) -> None:
        super().__init__()
        if len(target_cols) != 1:
            raise ValueError("Only one target column is supported.")
        self.image_size = image_size
        self.patch_grid = patch_grid
        self.target_cols = target_cols
        self.seed = seed

        df1 = self._read_label_df(LABEL_FILE1)
        df2 = self._read_label_df(LABEL_FILE2)

        all_files_1 = list(df1.index)
        all_files_2 = list(df2.index)
        train_1, val_1 = self._split(all_files_1)
        train_2, val_2 = self._split(all_files_2)

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
                base_raw = img_name_str
                base_no_prefix = img_name_str.removeprefix("1_")
                candidates = [
                    base_raw + ext for ext in [".png", ".jpg", ".jpeg"]
                ] + [
                    base_no_prefix + ext for ext in [".png", ".jpg", ".jpeg"]
                ]
                img_path = self._find_first_existing(IMG_DIR1, candidates)
                cls_labels = df1.loc[img_name_str].drop("batch", errors="ignore").values.astype(int)
            else:
                base_raw = img_name_str
                base_no_prefix = img_name_str.removeprefix("2_")
                candidates = [
                    base_raw + "_nroi" + ext for ext in [".png", ".jpg", ".jpeg"]
                ] + [
                    base_no_prefix + "_nroi" + ext for ext in [".png", ".jpg", ".jpeg"]
                ] + [
                    base_raw + ext for ext in [".png", ".jpg", ".jpeg"]
                ] + [
                    base_no_prefix + ext for ext in [".png", ".jpg", ".jpeg"]
                ]
                img_path = self._find_first_existing(IMG_DIR2, candidates)
                cls_labels = df2.loc[img_name_str].drop("batch", errors="ignore").values.astype(int)

            if not img_path:
                continue
            if max(self.target_cols) >= len(cls_labels):
                continue

            target = cls_labels[list(self.target_cols)].astype(np.float32)
            self.imgs.append(img_path)
            self.labels.append(target)
            self.names.append(os.path.basename(img_path))

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        bag = self._split_into_patches(img, self.patch_grid)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return bag, label

    def _read_label_df(self, path: str) -> pd.DataFrame:
        df = pd.read_excel(path, skiprows=1, header=None)
        df.iloc[:, 0] = df.iloc[:, 0].astype(str)
        df.set_index(df.columns[0], inplace=True)
        df["batch"] = 1
        return df

    def _split(self, names: List[str]) -> Tuple[List[str], List[str]]:
        rng = np.random.RandomState(self.seed)
        names = list(names)
        rng.shuffle(names)
        split = int(len(names) * 0.8)
        return names[:split], names[split:]

    def _find_first_existing(self, base_dir: str, candidates: List[str]) -> str:
        for name in candidates:
            full_path = os.path.join(base_dir, name)
            if os.path.exists(full_path):
                return full_path
        return ""

    def _split_into_patches(self, image_tensor: torch.Tensor, grid: int) -> torch.Tensor:
        c, h, w = image_tensor.shape
        if h % grid != 0 or w % grid != 0:
            raise ValueError(f"Image size ({h},{w}) not divisible by grid {grid}")
        patch_h = h // grid
        patch_w = w // grid
        patches = image_tensor.unfold(1, patch_h, patch_h).unfold(2, patch_w, patch_w)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, c, patch_h, patch_w)
        return patches


def main() -> None:
    parser = argparse.ArgumentParser(description="AttentionDeepMIL-style training on thyroid patches")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--gpus", type=str, default=None, help='Comma separated GPU ids, e.g. "0,2"')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_grid", type=int, default=4)
    parser.add_argument("--target_cols", type=str, default="4")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/hit/fx/Mamba-UNet/run_result/mil_attention_deepmil",
    )
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("This implementation matches AttentionDeepMIL and requires batch_size=1.")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print("\nGPU is ON!")

    target_cols = tuple(int(x) for x in args.target_cols.split(",") if x.strip() != "")

    print("Load Train and Test Set")
    loader_kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

    train_dataset = NoduleMilBags(
        mode="train",
        image_size=args.image_size,
        patch_grid=args.patch_grid,
        target_cols=target_cols,
        seed=args.seed,
    )
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    test_loader = data_utils.DataLoader(
        NoduleMilBags(
            mode="val",
            image_size=args.image_size,
            patch_grid=args.patch_grid,
            target_cols=target_cols,
            seed=args.seed,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    labels_np = np.array(train_dataset.labels, dtype=np.float32).reshape(-1)
    labels_np = labels_np[labels_np != -1]
    pos_count = int((labels_np == 1).sum())
    neg_count = int((labels_np == 0).sum())
    if pos_count > 0 and neg_count > 0:
        pos_weight = float(neg_count) / float(pos_count)
    else:
        pos_weight = 1.0

    print("Init Model")
    model = Attention()
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "log.txt")

    def train(epoch: int) -> None:
        model.train()
        train_loss = 0.0
        train_error = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for data, label in train_bar:
            bag_label = label[0]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)

            optimizer.zero_grad()
            loss, _ = model.calculate_objective(data, bag_label, pos_weight=pos_weight)
            train_loss += loss.data[0]
            error, _ = model.calculate_classification_error(data, bag_label)
            train_error += error
            loss.backward()
            optimizer.step()
            train_bar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

        train_loss /= len(train_loader)
        train_error /= len(train_loader)
        msg = (
            f"Epoch: {epoch}, Loss: {train_loss.cpu().numpy()[0]:.4f}, "
            f"Train error: {train_error:.4f}, pos_weight: {pos_weight:.4f}"
        )
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    def test(epoch: int) -> None:
        model.eval()
        test_loss = 0.0
        test_error = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for data, label in test_bar:
                bag_label = label[0]
                if args.cuda:
                    data, bag_label = data.cuda(), bag_label.cuda()
                data, bag_label = Variable(data), Variable(bag_label)
                loss, attention_weights = model.calculate_objective(data, bag_label, pos_weight=pos_weight)
                test_loss += loss.data[0]
                error, predicted_label = model.calculate_classification_error(data, bag_label)
                test_error += error
                test_bar.set_description(f"Test Loss {loss.item():.4f}")
                all_preds.append(predicted_label.cpu().numpy())
                all_labels.append(bag_label.cpu().numpy())

        test_error /= len(test_loader)
        test_loss /= len(test_loader)
        test_acc = 1.0 - test_error
        conf = confusion_matrix(
            np.concatenate(all_labels).astype(int),
            np.concatenate(all_preds).astype(int),
            labels=[0, 1],
        )
        msg = (
            f"Test Set, Loss: {test_loss.cpu().numpy()[0]:.4f}, "
            f"Test error: {test_error:.4f}, Test acc: {test_acc:.4f}"
        )
        print("\n" + msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
            f.write("Confusion Matrix:\n")
            np.savetxt(f, conf, fmt="%d")
            f.write("\n")
        # Confusion matrix is written to log.txt only.

    print(f"Logs saved to: {args.save_dir}")
    print("Start Training")
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        print("Start Testing")
        test(epoch)


if __name__ == "__main__":
    main()
