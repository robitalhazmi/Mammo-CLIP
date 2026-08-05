"""
Standalone training script for Mammo-CLIP + CoCoOp.

Trains learnable image-conditional prompts (context vectors + Meta-Net) on top
of a frozen pre-trained Mammo-CLIP model for downstream classification on VinDr.

Usage:
    python train_cocoop.py \
        --clip_chk_pt_path /path/to/b5-model-best-epoch-7.tar \
        --label Mass \
        --batch-size 4 \
        --epochs 50
"""

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Ensure codebase is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cocoop.mammo_cocoop import MammoCoCoOp

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ============================================================
# Default class names per label (matching Mammo-CLIP ZS prompts)
# ============================================================
LABEL_CLASSNAMES = {
    "Mass": ["no mass", "mass"],
    "Suspicious_Calcification": [
        "no suspicious calcification",
        "suspicious calcification",
    ],
    "density": ["density a", "density b", "density c", "density d"],
}


# ============================================================
# Dataset
# ============================================================
class VinDrCoCoOpDataset(Dataset):
    """
    VinDr mammography dataset for CoCoOp training/evaluation.

    Loads images from: {data_dir}/{img_dir}/{study_id}/{image_id}.png
    Returns dict with 'image' tensor and 'label' int.
    """

    def __init__(self, df, data_dir, img_dir, label_col, transform=None,
                 mean=0.3089279, std=0.25053555):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.img_dir = img_dir
        self.label_col = label_col
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = str(row["study_id"])
        image_id = str(row["image_id"])

        img_path = self.data_dir / self.img_dir / study_id / f"{image_id}.png"
        img = np.array(Image.open(img_path).convert("RGB"))

        # Albumentations transform
        if self.transform:
            img = self.transform(image=img)["image"]

        # Min-max normalize to [0, 1], then standardize
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)
        img = (img - self.mean) / self.std

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        label = int(row[self.label_col])

        return {"image": img, "label": label}


def get_transforms(img_size, is_train=True):
    """Build albumentations transforms matching the Mammo-CLIP pipeline."""
    transforms = []

    default_size = (912, 1520)  # (height, width)
    if img_size != list(default_size):
        transforms.append(A.Resize(height=img_size[1], width=img_size[0]))

    if is_train:
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(rotate=(-20, 20), translate_percent=0.1, scale=(0.8, 1.2), shear=(-20, 20), p=0.5),
        ])

    return A.Compose(transforms)


# ============================================================
# Training / Evaluation
# ============================================================
def seed_all(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        loss = model(images, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            log.info(
                f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    scheduler.step()
    avg_loss = total_loss / max(n_batches, 1)
    logger.add_scalar("train/loss", avg_loss, epoch + 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device, n_cls):
    """Evaluate model and return predictions + ground truth."""
    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        logits = model(images)  # [B, n_cls]
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)  # [N, n_cls]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    probs = F.softmax(all_logits, dim=1)  # [N, n_cls]

    if n_cls == 2:
        # Binary: use probability of positive class for AUC-ROC
        pos_probs = probs[:, 1].numpy()
        gt = all_labels.numpy()
        try:
            auc = roc_auc_score(gt, pos_probs)
        except ValueError:
            auc = 0.0
        preds = (pos_probs >= 0.5).astype(int)
        acc = accuracy_score(gt, preds)
        return {"auc_roc": auc, "accuracy": acc, "predictions": pos_probs, "labels": gt}
    else:
        # Multi-class: accuracy
        preds = all_logits.argmax(dim=1).numpy()
        gt = all_labels.numpy()
        acc = accuracy_score(gt, preds)
        try:
            auc = roc_auc_score(gt, probs.numpy(), multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0
        return {"auc_roc": auc, "accuracy": acc, "predictions": preds, "labels": gt}


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Mammo-CLIP + CoCoOp Training")

    # Checkpoint
    parser.add_argument("--clip_chk_pt_path", required=True, type=str,
                        help="Path to pre-trained Mammo-CLIP checkpoint (.tar)")

    # Data
    parser.add_argument("--dataset", default="ViNDr", type=str)
    parser.add_argument("--label", default="Mass", type=str,
                        choices=["Mass", "Suspicious_Calcification", "density"],
                        help="Target label for classification")
    parser.add_argument("--data-dir", default="/data/nas07_new/PersonalData/robit/Mammo-CLIP/data", type=str)
    parser.add_argument("--img-dir", default="vindr/images_png", type=str)
    parser.add_argument("--csv-file", default="vindr/vindr_detection_v1_folds.csv", type=str)
    parser.add_argument("--data_frac", default=1.0, type=float,
                        help="Fraction of training data to use")

    # CoCoOp hyperparameters
    parser.add_argument("--n_ctx", default=4, type=int,
                        help="Number of learnable context tokens")
    parser.add_argument("--ctx_init", default="", type=str,
                        help="Context initialization string (empty = random init)")
    parser.add_argument("--meta_net_reduction", default=16, type=int,
                        help="Meta-Net bottleneck reduction factor")

    # Training
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.002, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--patience", default=10, type=int,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--seed", default=42, type=int)

    # Image
    parser.add_argument("--img-size", nargs=2, default=[1520, 912], type=int,
                        help="Image size [width, height]")
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555, type=float)

    # Output
    parser.add_argument("--output_path", required=True, type=str)

    # Misc
    parser.add_argument("--num_workers", default=4, type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Output directories ----
    output_path = Path(args.output_path)
    chk_pt_path = output_path / "checkpoints"
    tb_logs_path = output_path / "tb_logs"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)

    # Save config
    pickle.dump(vars(args), open(output_path / "train_config.pkl", "wb"))
    with open(output_path / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    logger = SummaryWriter(tb_logs_path)

    # ---- Class names ----
    classnames = LABEL_CLASSNAMES[args.label]
    n_cls = len(classnames)
    print(f"Label: {args.label} | Classes: {classnames} | n_cls: {n_cls}")

    # ---- Load data ----
    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / args.csv_file).fillna(0)

    train_df = df[df["split"] == "training"].copy()
    test_df = df[df["split"] == "test"].copy()

    if args.data_frac < 1.0:
        train_df = train_df.sample(frac=args.data_frac, random_state=args.seed).reset_index(drop=True)
        print(f"Using {args.data_frac*100:.0f}% of training data: {len(train_df)} samples")

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    train_transform = get_transforms(args.img_size, is_train=True)
    test_transform = get_transforms(args.img_size, is_train=False)

    train_dataset = VinDrCoCoOpDataset(
        df=train_df, data_dir=data_dir, img_dir=args.img_dir,
        label_col=args.label, transform=train_transform,
        mean=args.mean, std=args.std,
    )
    test_dataset = VinDrCoCoOpDataset(
        df=test_df, data_dir=data_dir, img_dir=args.img_dir,
        label_col=args.label, transform=test_transform,
        mean=args.mean, std=args.std,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Build model ----
    model = MammoCoCoOp(
        clip_ckpt_path=args.clip_chk_pt_path,
        classnames=classnames,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        meta_net_reduction=args.meta_net_reduction,
    )
    model = model.to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    # ---- Optimizer: only prompt_learner parameters ----
    prompt_learner = model.module.prompt_learner if isinstance(model, torch.nn.DataParallel) else model.prompt_learner
    optimizer = torch.optim.SGD(
        prompt_learner.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training loop ----
    best_metric = 0.0
    best_epoch = -1
    patience_counter = 0
    metric_name = "accuracy" if args.label == "density" else "auc_roc"

    print(f"\n{'='*60}")
    print(f"Starting CoCoOp training for {args.label}")
    print(f"Primary metric: {metric_name}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, logger)

        # Evaluate
        eval_results = evaluate(model, test_loader, device, n_cls)
        current_metric = eval_results[metric_name]

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"AUC: {eval_results['auc_roc']:.4f} | "
            f"Acc: {eval_results['accuracy']:.4f} | "
            f"LR: {lr:.6f} | "
            f"Time: {elapsed:.0f}s"
        )

        logger.add_scalar(f"val/{metric_name}", current_metric, epoch + 1)
        logger.add_scalar("val/auc_roc", eval_results["auc_roc"], epoch + 1)
        logger.add_scalar("val/accuracy", eval_results["accuracy"], epoch + 1)

        # Save best model
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0

            # Save prompt_learner state dict only
            save_dict = {
                "prompt_learner": prompt_learner.state_dict(),
                "epoch": epoch + 1,
                metric_name: current_metric,
                "auc_roc": eval_results["auc_roc"],
                "accuracy": eval_results["accuracy"],
                "config": vars(args),
            }
            torch.save(save_dict, chk_pt_path / "cocoop_best.pth")
            print(f"  -> New best {metric_name}: {best_metric:.4f} (saved)")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

    # ---- Final evaluation with best model ----
    print(f"\n{'='*60}")
    print(f"Training complete. Best {metric_name}: {best_metric:.4f} at epoch {best_epoch}")
    print(f"{'='*60}")

    # Reload best checkpoint
    best_ckpt = torch.load(chk_pt_path / "cocoop_best.pth", map_location="cpu")
    prompt_learner.load_state_dict(best_ckpt["prompt_learner"])

    final_results = evaluate(model, test_loader, device, n_cls)
    print(f"Final AUC-ROC: {final_results['auc_roc']:.4f}")
    print(f"Final Accuracy: {final_results['accuracy']:.4f}")

    # Save predictions CSV
    results_df = pd.DataFrame({
        "prediction": final_results["predictions"],
        "label": final_results["labels"],
    })
    results_df.to_csv(output_path / f"cocoop_{args.label}_predictions.csv", index=False)

    # Save summary
    summary = {
        "label": args.label,
        "classnames": classnames,
        "best_epoch": best_epoch,
        "best_metric_name": metric_name,
        "best_metric_value": best_metric,
        "final_auc_roc": final_results["auc_roc"],
        "final_accuracy": final_results["accuracy"],
        "n_ctx": args.n_ctx,
        "ctx_init": args.ctx_init,
        "clip_checkpoint": args.clip_chk_pt_path,
        "data_frac": args.data_frac,
        "batch_size": args.batch_size,
        "epochs_run": best_epoch,
        "lr": args.lr,
    }
    with open(output_path / f"cocoop_{args.label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {output_path}")
    logger.close()


if __name__ == "__main__":
    main()
