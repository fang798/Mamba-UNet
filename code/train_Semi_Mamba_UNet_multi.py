# python train_Semi_Mamba_UNet_multi.py --gpus "0,1,2,3" --exp Thyroid/Semi_Mamba_UNet --max_iterations 30000 --batch_size 16 --num_classes 2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_Semi_Mamba_UNet_multi.py --exp Thyroid/Semi_Mamba_UNet --max_iterations 30000 --batch_size 16 --num_classes 2
import argparse
import logging
import os
import random
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from networks.vision_mamba import MambaUnet as ViM_seg
from config import get_config
from dataloaders import utils
from dataloaders.dataset import RandomGenerator, TwoStreamBatchSampler
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, metrics, ramps
from val_2D import test_single_volume

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


def _find_first_existing(base_dir: str, candidates) -> str:
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


class ThyroidSemiDataset(Dataset):
    def __init__(self, split: str, transform=None, seed: int = 42, include_unlabeled: bool = True):
        self.split = split
        self.transform = transform
        self.labeled_samples = []
        self.unlabeled_samples = []

        labeled_df = _read_label_df(LABELED_EXCEL)
        all_files = [str(x) for x in labeled_df.index]
        train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=seed)
        if split == "train":
            selected_files = train_files
        elif split in ("val", "test"):
            selected_files = val_files
        else:
            raise ValueError(f"Unsupported split: {split}")

        for img_name_str in selected_files:
            candidates = [img_name_str.removeprefix("1_") + ext for ext in [".png", ".jpg", ".jpeg"]]
            img_path = _find_first_existing(LABELED_IMG_DIR, candidates)
            if not img_path:
                continue

            base = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = _find_mask_path(LABELED_MASK_DIR, base)
            cls_labels = labeled_df.loc[img_name_str].values.astype(int)

            self.labeled_samples.append(
                {"image": img_path, "mask": mask_path, "cls_labels": cls_labels, "is_labeled": True}
            )

        if split == "train" and include_unlabeled:
            unlabeled_df = _read_label_df(UNLABELED_EXCEL)
            for img_name in unlabeled_df.index:
                img_name_str = str(img_name)
                candidates = [img_name_str + "_nroi" + ext for ext in [".png", ".jpg", ".jpeg"]]
                img_path = _find_first_existing(UNLABELED_IMG_DIR, candidates)
                if not img_path:
                    continue

                cls_labels = unlabeled_df.loc[img_name_str].values.astype(int)
                self.unlabeled_samples.append(
                    {"image": img_path, "mask": "", "cls_labels": cls_labels, "is_labeled": False}
                )

        self.samples = self.labeled_samples + self.unlabeled_samples
        self.labeled_count = len(self.labeled_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = item["image"]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255.0

        mask_path = item["mask"]
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                if mask.shape != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        mask = (mask >= 128).astype(np.uint8)
        sample = {"image": image, "label": mask}

        if self.transform:
            sample = self.transform(sample)
            image = sample["image"]
            label = sample["label"]
        else:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(mask.astype(np.uint8))

        if torch.is_tensor(label):
            label = (label > 0).to(torch.uint8)

        return {
            "image": image,
            "label": label,
            "cls_labels": torch.tensor(item["cls_labels"]).long(),
            "is_labeled": item["is_labeled"],
            "name": os.path.basename(image_path),
        }

parser = argparse.ArgumentParser()
# parser.add_argument('--gpus', type=str, default="0",
#                     help='Comma separated GPU ids, e.g. "0,2"')
parser.add_argument('--root_path', type=str,
                    default=ROOT_DIR, help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='thyroid/Semi_Mamba_UNet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
    # '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )

parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
config = get_config(args)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # def create_model(ema=False):
    #     # Network definition
    #     model = net_factory(net_type='unet', in_chns=1,
    #                         class_num=num_classes)
    #     if ema:
    #         for param in model.parameters():
    #             param.detach_()
    #     return model

    model1 = net_factory(net_type='unet', in_chns=1, class_num=num_classes).cuda()

    # model2 = ViM_seg(config, img_size=args.patch_size,
    #                  num_classes=args.num_classes).cuda()
    # model2.load_from(config)

    model2 = ViM_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model2.load_from(config)

    if torch.cuda.device_count() > 1:
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)

    def _state_dict(model):
        return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()



    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_transform = transforms.Compose([RandomGenerator(args.patch_size)])
    train_dataset = ThyroidSemiDataset(
        split="train", transform=train_transform, seed=args.seed, include_unlabeled=True
    )
    val_dataset = ThyroidSemiDataset(
        split="val", transform=None, seed=args.seed, include_unlabeled=False
    )

    total_slices = len(train_dataset)
    labeled_slice = train_dataset.labeled_count
    unlabeled_slice = total_slices - labeled_slice
    if labeled_slice == 0:
        raise ValueError("No labeled samples found.")
    labeled_idxs = list(range(0, labeled_slice))
    if unlabeled_slice == 0:
        logging.warning("No unlabeled samples found; using labeled samples for both streams.")
        unlabeled_idxs = labeled_idxs
    else:
        unlabeled_idxs = list(range(labeled_slice, total_slices))

    print(
        "Total slices is: {}, labeled slices is: {}, unlabeled slices is: {}".format(
            total_slices, labeled_slice, unlabeled_slice
        )
    )
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs
    )

    trainloader = DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
    )

    model1.train()
    model2.train()

    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(
                iter_num // 150)

            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(
                outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(
                outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = dice_loss(
                outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(
                outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))

            from utils.losses import ConstraLoss
            con1 = ConstraLoss(outputs1,outputs2)

            # model1_loss = loss1 + consistency_weight * pseudo_supervision1
            # model2_loss = loss2 + consistency_weight * pseudo_supervision2

            model1_loss = loss1 + consistency_weight * pseudo_supervision1 +0.5*con1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2 +0.5*con1

            loss = model1_loss + model2_loss 

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(val_dataset)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice',
                                  performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95',
                                  mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(_state_dict(model1), save_mode_path)
                    torch.save(_state_dict(model1), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(val_dataset)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice',
                                  performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95',
                                  mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(_state_dict(model2), save_mode_path)
                    torch.save(_state_dict(model2), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(_state_dict(model1), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(_state_dict(model2), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
