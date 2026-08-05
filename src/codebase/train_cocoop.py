"""
Standalone training script for Mammo-CoCoOp (Contrastive Training).

Trains learnable image-conditional prompts (context vectors + Meta-Net) on top
of a frozen pre-trained Mammo-CLIP model. 

1. Training: Contrastive InfoNCE loss aligning VinDr images with their 
   per-image LLM descriptions (from clip_vindr_final_prompts.csv).
2. Evaluation: Zero-shot classification using the trained Meta-Net and 
   fixed class-specific LLM templates (from vindr_detection_v1_folds.csv).
"""

import argparse
import ast
import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path

import albumentations as A
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
# Zero-Shot Class Templates (for downstream evaluation)
# ============================================================
LABEL_CLASSNAMES = {
    "Mass": [
        "A mammogram showing no discernible masses, calcifications, or architectural distortion, constituting a normal examination.",
        "A mammogram showing a discernible mass, requiring further clinical evaluation.",
    ],
    "Suspicious_Calcification": [
        "A mammogram showing no discernible masses, calcifications, or architectural distortion, constituting a normal examination.",
        "A mammogram showing suspicious calcifications, requiring further clinical evaluation.",
    ],
    "Malignancy": [
        "A mammogram showing no evidence of malignancy, consistent with benign or normal findings (BI-RADS 1 to 3).",
        "A mammogram showing suspicious findings highly suggestive of malignancy, requiring further clinical evaluation (BI-RADS 4 or 5).",
    ],
    "density": [
        "A mammogram showing almost entirely fatty tissue (ACR density A).",
        "A mammogram showing scattered areas of fibroglandular density (ACR density B).",
        "A mammogram showing heterogeneously dense fibroglandular tissue (ACR density C).",
        "A mammogram showing extremely dense tissue (ACR density D).",
    ],
}


# ============================================================
# Datasets
# ============================================================
class VinDrContrastiveTrainDataset(Dataset):
    """
    Loads images and their unique LLM prompts for Contrastive Learning.
    Uses `clip_vindr_final_prompts.csv`.
    """
    def __init__(self, df, data_dir, img_dir, transform=None, mean=0.3089279, std=0.25053555):
        self.data_dir = Path(data_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.mean = mean
        self.std = std
        
        # Flatten the dataframe (unpack CC and MLO lists)
        self.samples = []
        for _, row in df.iterrows():
            study_id = str(row["patient_id"])
            
            # CC images
            if isinstance(row["CC"], str) and row["CC"] != "[]":
                cc_imgs = ast.literal_eval(row["CC"])
                for img in cc_imgs:
                    img_id = img.replace(".png", "")
                    self.samples.append({
                        "study_id": study_id,
                        "image_id": img_id,
                        "text": str(row["cc_prompt"])
                    })
                    
            # MLO images
            if isinstance(row["MLO"], str) and row["MLO"] != "[]":
                mlo_imgs = ast.literal_eval(row["MLO"])
                for img in mlo_imgs:
                    img_id = img.replace(".png", "")
                    self.samples.append({
                        "study_id": study_id,
                        "image_id": img_id,
                        "text": str(row["mlo_prompt"])
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.data_dir / self.img_dir / sample["study_id"] / f"{sample['image_id']}.png"
        
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]

        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)
        img = (img - self.mean) / self.std

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        return {"image": img, "text": sample["text"]}


class VinDrZeroShotTestDataset(Dataset):
    """
    Loads images and ground-truth labels for Zero-Shot evaluation.
    Uses `vindr_detection_v1_folds.csv`.
    """
    def __init__(self, df, data_dir, img_dir, label_col, transform=None, mean=0.3089279, std=0.25053555):
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

        if self.transform:
            img = self.transform(image=img)["image"]

        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)
        img = (img - self.mean) / self.std

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        if self.label_col == "Malignancy":
            birads_str = str(row.get("breast_birads", ""))
            if "4" in birads_str or "5" in birads_str:
                label = 1
            else:
                label = 0
        else:
            label = int(row[self.label_col])
            
        return {"image": img, "label": label}


def get_transforms(img_size, is_train=True):
    transforms = []
    default_size = (912, 1520)
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, logger):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        texts = batch["text"]  # list of strings (length B)
        B = images.shape[0]

        # Forward pass: Contrastive Mode (is_eval=False)
        image_proj, text_proj, logit_scale = model(images, texts, is_eval=False)

        # InfoNCE Loss
        logits_per_image = logit_scale * image_proj @ text_proj.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(B, device=device, dtype=torch.long)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2.0

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
def evaluate(model, loader, device, class_templates, n_cls):
    """Zero-Shot evaluation on classification task."""
    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        # Forward pass: Evaluation Mode (is_eval=True)
        # text_batch is the fixed class_templates list (length C)
        logits = model(images, class_templates, is_eval=True)  # [B, C]
        
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)  # [N, n_cls]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    probs = F.softmax(all_logits, dim=1)  # [N, n_cls]

    if n_cls == 2:
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
    parser = argparse.ArgumentParser(description="Mammo-CoCoOp Contrastive Training")

    parser.add_argument("--clip_chk_pt_path", required=True, type=str)
    
    parser.add_argument("--label", default="Mass", type=str,
                        choices=["Mass", "Suspicious_Calcification", "Malignancy", "density"],
                        help="Target label for zero-shot evaluation")
    parser.add_argument("--data-dir", default="/data/nas07_new/PersonalData/robit/Mammo-CLIP/data", type=str)
    parser.add_argument("--img-dir", default="vindr/images_png", type=str)
    
    # We now use two CSVs: one for prompts (train), one for labels (test)
    parser.add_argument("--prompts_csv", default="vindr/clip_vindr_final_prompts.csv", type=str)
    parser.add_argument("--labels_csv", default="vindr/vindr_detection_v1_folds.csv", type=str)
    
    parser.add_argument("--data_frac", default=1.0, type=float)

    parser.add_argument("--n_ctx", default=4, type=int)
    parser.add_argument("--ctx_init", default="", type=str)
    parser.add_argument("--meta_net_reduction", default=16, type=int)

    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.002, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--img-size", nargs=2, default=[1520, 912], type=int)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555, type=float)

    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--num_workers", default=4, type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_path = Path(args.output_path)
    chk_pt_path = output_path / "checkpoints"
    tb_logs_path = output_path / "tb_logs"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)

    pickle.dump(vars(args), open(output_path / "train_config.pkl", "wb"))
    with open(output_path / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    logger = SummaryWriter(tb_logs_path)

    # Class templates for zero-shot eval
    class_templates = LABEL_CLASSNAMES[args.label]
    n_cls = len(class_templates)
    print(f"Eval Label: {args.label} | Classes: {n_cls}")

    data_dir = Path(args.data_dir)
    
    # 1. Load Train Dataset (Prompts CSV)
    prompts_df = pd.read_csv(data_dir / args.prompts_csv).fillna("")
    train_df = prompts_df[prompts_df["split"] == "training"].copy()
    
    if args.data_frac < 1.0:
        train_df = train_df.sample(frac=args.data_frac, random_state=args.seed).reset_index(drop=True)
        
    train_dataset = VinDrContrastiveTrainDataset(
        df=train_df, data_dir=data_dir, img_dir=args.img_dir,
        transform=get_transforms(args.img_size, is_train=True),
        mean=args.mean, std=args.std,
    )

    # 2. Load Test Dataset (Labels CSV)
    labels_df = pd.read_csv(data_dir / args.labels_csv).fillna(0)
    test_df = labels_df[labels_df["split"] == "test"].copy()
    
    test_dataset = VinDrZeroShotTestDataset(
        df=test_df, data_dir=data_dir, img_dir=args.img_dir,
        label_col=args.label,
        transform=get_transforms(args.img_size, is_train=False),
        mean=args.mean, std=args.std,
    )

    print(f"Train samples (images): {len(train_dataset)} | Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Build model
    model = MammoCoCoOp(
        clip_ckpt_path=args.clip_chk_pt_path,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        meta_net_reduction=args.meta_net_reduction,
    )
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    prompt_learner = model.module.prompt_learner if isinstance(model, torch.nn.DataParallel) else model.prompt_learner
    optimizer = torch.optim.SGD(
        prompt_learner.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_metric = 0.0
    best_epoch = -1
    patience_counter = 0
    metric_name = "accuracy" if args.label == "density" else "auc_roc"

    print(f"\n{'='*60}")
    print(f"Starting Contrastive CoCoOp training for {args.label}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train (InfoNCE)
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, logger)

        # Evaluate (Zero-Shot Classification)
        eval_results = evaluate(model, test_loader, device, class_templates, n_cls)
        current_metric = eval_results[metric_name]

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Contrastive Loss: {avg_loss:.4f} | "
            f"ZS AUC: {eval_results['auc_roc']:.4f} | "
            f"ZS Acc: {eval_results['accuracy']:.4f} | "
            f"LR: {lr:.6f} | "
            f"Time: {elapsed:.0f}s"
        )

        logger.add_scalar(f"val/{metric_name}", current_metric, epoch + 1)
        logger.add_scalar("val/auc_roc", eval_results["auc_roc"], epoch + 1)
        logger.add_scalar("val/accuracy", eval_results["accuracy"], epoch + 1)

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0

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
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*60}")
    print(f"Training complete. Best {metric_name}: {best_metric:.4f} at epoch {best_epoch}")

    best_ckpt = torch.load(chk_pt_path / "cocoop_best.pth", map_location="cpu")
    prompt_learner.load_state_dict(best_ckpt["prompt_learner"])

    final_results = evaluate(model, test_loader, device, class_templates, n_cls)
    print(f"Final AUC-ROC: {final_results['auc_roc']:.4f}")
    
    results_df = pd.DataFrame({
        "prediction": final_results["predictions"],
        "label": final_results["labels"],
    })
    results_df.to_csv(output_path / f"cocoop_{args.label}_predictions.csv", index=False)

    summary = {
        "label": args.label,
        "best_epoch": best_epoch,
        "best_metric_value": best_metric,
        "final_auc_roc": final_results["auc_roc"],
        "clip_checkpoint": args.clip_chk_pt_path,
        "epochs_run": best_epoch,
    }
    with open(output_path / f"cocoop_{args.label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.close()


if __name__ == "__main__":
    main()
