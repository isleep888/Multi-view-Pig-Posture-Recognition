#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pig posture recognition training script for RTX 4090.

Baseline from the public high-score notebook:
- train2 / test CSV
- manual label correction with changes.csv
- abnormal sample removal
- GroupKFold by image_id
- EfficientNetV2-S ImageNet-21k pretrained
- WeightedRandomSampler
- AdamW + AMP + Cosine LR + gradient clipping
- 5-fold softmax ensemble

Added improvements:
- context crop around bbox
- soft isolation: suppress other pigs inside the crop
- no rotation / no horizontal flip, preserving left/right posture semantics
- pad-ratio TTA at inference
- optional pig-level/multi-view fusion if a grouping column exists
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import random
import re
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold


# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Basic utils
# ----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_bbox_string(bbox) -> Tuple[float, float, float, float]:
    if isinstance(bbox, (list, tuple, np.ndarray)):
        vals = list(bbox)
    else:
        s = str(bbox).strip()
        try:
            vals = ast.literal_eval(s)
        except Exception:
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            s = s.replace(",", " ")
            vals = [float(x) for x in s.split() if x]
    if len(vals) != 4:
        raise ValueError(f"Bad bbox: {bbox}")
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def find_image_path(img_dir: Path, image_id: str) -> Path:
    image_id = str(image_id)
    candidates = [
        img_dir / image_id,
        img_dir / f"{image_id}.jpg",
        img_dir / f"{image_id}.jpeg",
        img_dir / f"{image_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find image {image_id} under {img_dir}")


def get_domain_from_image_id(image_id: str) -> str:
    name = Path(str(image_id)).stem
    parts = name.split("_")
    if len(parts) >= 4 and parts[0] in {"train", "test"}:
        return "_".join(parts[1:4])
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return name


def safe_filename(x: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(x))


def maybe_download_changes_csv(changes_csv: Optional[Path], changes_url: str, out_dir: Path) -> Optional[Path]:
    if changes_csv is not None and changes_csv.exists():
        return changes_csv

    if not changes_url:
        return None

    ensure_dir(out_dir)
    dst = out_dir / "changes.csv"
    if dst.exists():
        return dst

    print(f"Downloading changes.csv from: {changes_url}")
    try:
        urllib.request.urlretrieve(changes_url, dst)
        print(f"Saved changes.csv to {dst}")
        return dst
    except Exception as e:
        print(f"[WARN] Failed to download changes.csv: {repr(e)}")
        print("[WARN] Continue without label correction.")
        return None


# ----------------------------
# Data preparation
# ----------------------------

def prepare_df(csv_path: Path, is_train: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"bbox", "image_id", "row_id"}
    if is_train:
        required.add("class_id")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    parsed = df["bbox"].apply(parse_bbox_string)
    df["x"] = parsed.apply(lambda z: z[0])
    df["y"] = parsed.apply(lambda z: z[1])
    df["w"] = parsed.apply(lambda z: z[2])
    df["h"] = parsed.apply(lambda z: z[3])

    df["row_id"] = df["row_id"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    df["domain"] = df["image_id"].apply(get_domain_from_image_id)

    if is_train:
        df["class_id"] = df["class_id"].astype(int)

    return df


def apply_label_corrections(df: pd.DataFrame, changes_csv: Optional[Path]) -> pd.DataFrame:
    if changes_csv is None or not changes_csv.exists():
        print("[WARN] No changes.csv found. Training without manual label corrections.")
        return df

    changes = pd.read_csv(changes_csv)
    if "corrected_class_id" not in changes.columns:
        print("[WARN] changes.csv has no corrected_class_id column. Skip corrections.")
        return df
    if "row_id" not in changes.columns:
        print("[WARN] changes.csv has no row_id column. Skip corrections.")
        return df

    changes = changes.copy()
    changes["class_id"] = changes["corrected_class_id"].astype(int)

    before = df["class_id"].copy()

    df2 = df.set_index("row_id")
    ch = changes.set_index("row_id")

    common = df2.index.intersection(ch.index)
    df2.loc[common, "class_id"] = ch.loc[common, "class_id"].astype(int)

    df2 = df2.reset_index()

    n_changed = int((before.values != df2["class_id"].values).sum())
    print(f"Applied label corrections: {len(common)} matched rows, {n_changed} class_id changed.")
    return df2


def remove_known_bad_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Public notebook cleanup:
    Drop class-4 pigs near bottom border in pen1_orb_cam2_20250108_085.
    This removes likely truncated / mislabeled samples.
    """
    if "height" not in df.columns:
        print("[WARN] Column 'height' not in train csv. Skip known bad sample removal.")
        return df

    mask = (df["image_id"].str.contains("pen1_orb_cam2_20250108_085", regex=False)) & (df["class_id"] == 4)
    cand = df[mask].copy()

    if len(cand) == 0:
        print("Known bad sample removal: no candidates found.")
        return df

    def is_near_bottom(row) -> bool:
        _, y, _, h = parse_bbox_string(row["bbox"])
        return float(row["height"]) - (y + h) < 5

    drop_idx = cand[cand.apply(is_near_bottom, axis=1)].index
    print(f"Known bad sample removal: dropping {len(drop_idx)} rows.")
    return df.drop(drop_idx).reset_index(drop=True)


def print_data_diagnosis(df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("Dataset diagnosis")
    print("=" * 80)
    print("Train rows:", len(df), "images:", df["image_id"].nunique())
    print("Test rows:", len(test_df), "images:", test_df["image_id"].nunique())

    print("\nTrain class distribution:")
    print(df["class_id"].value_counts().sort_index())

    print("\nTrain domain distribution:")
    print(df["domain"].value_counts())

    print("\nTest domain distribution:")
    print(test_df["domain"].value_counts())


# ----------------------------
# Crop and soft isolation
# ----------------------------

def clip_box(box: np.ndarray, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array(
        [
            max(0, min(W, x1)),
            max(0, min(H, y1)),
            max(0, min(W, x2)),
            max(0, min(H, y2)),
        ],
        dtype=np.float32,
    )


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def intersect_box(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> np.ndarray:
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def suppress_region_gray(crop: np.ndarray, x1: float, y1: float, x2: float, y2: float, strength: float = 0.30) -> np.ndarray:
    H, W = crop.shape[:2]
    x1 = int(max(0, min(W, round(x1))))
    y1 = int(max(0, min(H, round(y1))))
    x2 = int(max(0, min(W, round(x2))))
    y2 = int(max(0, min(H, round(y2))))

    if x2 <= x1 or y2 <= y1:
        return crop

    region = crop[y1:y2, x1:x2].copy()
    if region.size == 0:
        return crop

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    blended = (1.0 - strength) * region.astype(np.float32) + strength * gray_rgb.astype(np.float32)
    crop[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return crop


def context_soft_isolation_crop(
    img_rgb: np.ndarray,
    target_row: pd.Series,
    image_rows: Sequence[dict],
    pad_ratio: float,
    img_size: int,
    soft_isolation: bool = True,
    suppress_strength: float = 0.30,
    min_inter_ratio: float = 0.003,
) -> np.ndarray:
    H, W = img_rgb.shape[:2]

    x, y, w, h = float(target_row["x"]), float(target_row["y"]), float(target_row["w"]), float(target_row["h"])
    target_box = clip_box(xywh_to_xyxy(x, y, w, h), W, H)

    if box_area(target_box) <= 1:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    x1, y1, x2, y2 = target_box
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio

    crop_box = clip_box(
        np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=np.float32),
        W,
        H,
    )

    cx1, cy1, cx2, cy2 = crop_box.astype(int)
    if cx2 <= cx1 or cy2 <= cy1:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    crop = img_rgb[cy1:cy2, cx1:cx2].copy()
    crop_area = max(1.0, box_area(crop_box))

    if soft_isolation:
        target_id = str(target_row["row_id"])

        for other in image_rows:
            if str(other["row_id"]) == target_id:
                continue

            ox, oy, ow, oh = float(other["x"]), float(other["y"]), float(other["w"]), float(other["h"])
            other_box = clip_box(xywh_to_xyxy(ox, oy, ow, oh), W, H)
            inter = intersect_box(crop_box, other_box)
            if inter is None:
                continue

            inter_area = box_area(inter)
            if inter_area / crop_area < min_inter_ratio:
                continue

            crop = suppress_region_gray(
                crop,
                inter[0] - cx1,
                inter[1] - cy1,
                inter[2] - cx1,
                inter[3] - cy1,
                strength=suppress_strength,
            )

    # Letterbox square without stretching pig shape
    h0, w0 = crop.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    scale = min(img_size / w0, img_size / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    mean_color = [int(v) for v in crop.reshape(-1, 3).mean(axis=0)]
    canvas[:, :] = mean_color

    left = (img_size - nw) // 2
    top = (img_size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


# ----------------------------
# Dataset
# ----------------------------

class PigPostureDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        transform,
        img_size: int,
        train_mode: bool = True,
        train_pad_ratios: Sequence[float] = (0.0,),
        fixed_pad_ratio: float = 0.0,
        soft_isolation: bool = True,
        cache_images: bool = True,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.img_size = img_size
        self.train_mode = train_mode
        self.train_pad_ratios = list(train_pad_ratios)
        self.fixed_pad_ratio = fixed_pad_ratio
        self.soft_isolation = soft_isolation
        self.cache_images = cache_images
        self.is_test = is_test

        self.rows_by_image: Dict[str, List[dict]] = {
            image_id: g.to_dict("records")
            for image_id, g in self.df.groupby("image_id")
        }

        self.image_cache: Dict[str, np.ndarray] = {}
        if self.cache_images:
            unique_images = self.df["image_id"].unique().tolist()
            print(f"Preloading {len(unique_images)} images into RAM from {self.image_dir}")
            for image_id in tqdm(unique_images, desc="Preload images"):
                p = find_image_path(self.image_dir, image_id)
                img = Image.open(p).convert("RGB")
                self.image_cache[image_id] = np.array(img)

    def __len__(self) -> int:
        return len(self.df)

    def _read_image(self, image_id: str) -> np.ndarray:
        if image_id in self.image_cache:
            return self.image_cache[image_id]
        p = find_image_path(self.image_dir, image_id)
        return np.array(Image.open(p).convert("RGB"))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        img = self._read_image(image_id)

        if self.train_mode:
            pad_ratio = random.choice(self.train_pad_ratios)
        else:
            pad_ratio = self.fixed_pad_ratio

        crop = context_soft_isolation_crop(
            img_rgb=img,
            target_row=row,
            image_rows=self.rows_by_image[image_id],
            pad_ratio=pad_ratio,
            img_size=self.img_size,
            soft_isolation=self.soft_isolation,
        )

        pil = Image.fromarray(crop)
        x = self.transform(pil)

        if self.is_test:
            return x, str(row["row_id"])

        y = int(row["class_id"])
        return x, y, str(row["row_id"]), image_id


def make_transforms(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            # No rotation and no horizontal flip.
            # This preserves left/right posture semantics.
            transforms.RandomResizedCrop(img_size, scale=(0.88, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.25, contrast=0.22, saturation=0.25, hue=0.02)],
                p=0.75,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.4, 2.5), value="random"),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, val_tf


def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    labels = df["class_id"].astype(int).values
    counts = np.bincount(labels, minlength=5).astype(np.float32)
    weights_per_class = counts.sum() / np.maximum(counts, 1)
    sample_weights = weights_per_class[labels]

    print("Sampler class counts:", counts.tolist())
    print("Sampler class weights:", weights_per_class.tolist())

    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ----------------------------
# Model
# ----------------------------

def create_model(model_name: str, num_classes: int, pretrained: bool, drop_rate: float, drop_path_rate: float):
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    if pretrained:
        from safetensors.torch import load_file
        ckpt_path = "/data/yuehan/Swim_pose/resnet_pig_package/model.safetensors"
        state = load_file(ckpt_path)

        # 去掉分类头，因为你的任务是 5 类，官方权重是 ImageNet 1000 类
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}

        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded local pretrained weights:", ckpt_path)
        print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))

    return model


# ----------------------------
# Train / eval
# ----------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> dict:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels, _, _ in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        total_loss += float(loss.item())
        n_batches += 1

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(5)))

    return {
        "val_loss": float(total_loss / max(1, n_batches)),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "cm": cm.tolist(),
    }


def train_one_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args,
    paths: dict,
    device: torch.device,
) -> dict:
    print("\n" + "=" * 80)
    print(f"Fold {fold + 1}/{args.n_splits}")
    print("=" * 80)
    print("Train rows:", len(train_df), "Val rows:", len(val_df))
    print("Val class distribution:")
    print(val_df["class_id"].value_counts().sort_index())

    train_tf, val_tf = make_transforms(args.img_size)

    train_ds = PigPostureDataset(
        df=train_df,
        image_dir=paths["train_img_dir"],
        transform=train_tf,
        img_size=args.img_size,
        train_mode=True,
        train_pad_ratios=args.train_pad_ratios,
        fixed_pad_ratio=args.val_pad_ratio,
        soft_isolation=not args.no_soft_isolation,
        cache_images=not args.no_cache_images,
        is_test=False,
    )

    val_ds = PigPostureDataset(
        df=val_df,
        image_dir=paths["train_img_dir"],
        transform=val_tf,
        img_size=args.img_size,
        train_mode=False,
        train_pad_ratios=args.train_pad_ratios,
        fixed_pad_ratio=args.val_pad_ratio,
        soft_isolation=not args.no_soft_isolation,
        cache_images=not args.no_cache_images,
        is_test=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=make_sampler(train_df),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    model = create_model(
        model_name=args.model,
        num_classes=5,
        pretrained=True,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    best_f1 = -1.0
    bad_epochs = 0
    best_path = paths["model_dir"] / f"{args.model.replace('/', '_')}_fold{fold + 1}_best.pth"
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Fold {fold + 1} epoch {epoch}/{args.epochs}")

        for images, labels, _, _ in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=args.use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=running_loss / max(1, n_batches), lr=optimizer.param_groups[0]["lr"])

        scheduler.step()

        train_loss = running_loss / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, device, args.use_amp)
        val_f1 = val_metrics["macro_f1"]

        elapsed_min = (time.time() - t0) / 60.0

        row = {
            "fold": fold + 1,
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_min": elapsed_min,
        }
        history.append(row)

        print(
            f"[fold {fold + 1}] epoch={epoch} "
            f"loss={train_loss:.5f} val_loss={val_metrics['val_loss']:.5f} "
            f"acc={val_metrics['acc']:.5f} macro_f1={val_f1:.5f} "
            f"time={elapsed_min:.2f} min"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "fold": fold + 1,
                    "epoch": epoch,
                    "best_f1": float(best_f1),
                    "img_size": args.img_size,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_path,
            )
            print("Saved best:", best_path)
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stop fold {fold + 1} at epoch {epoch}. Best f1={best_f1:.5f}")
            break

    hist_df = pd.DataFrame(history)
    hist_path = paths["model_dir"] / f"{args.model.replace('/', '_')}_fold{fold + 1}_history.csv"
    hist_df.to_csv(hist_path, index=False)

    # free memory
    del model, train_loader, val_loader, train_ds, val_ds
    torch.cuda.empty_cache()

    return {
        "fold": fold + 1,
        "best_path": str(best_path),
        "best_f1": float(best_f1),
        "history_path": str(hist_path),
    }


# ----------------------------
# Inference
# ----------------------------

def load_fold_models(ckpt_paths: Sequence[str], device: torch.device) -> List[nn.Module]:
    models = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = ckpt["model_name"]
        saved_args = ckpt.get("args", {})
        drop_rate = float(saved_args.get("drop_rate", 0.40))
        drop_path_rate = float(saved_args.get("drop_path_rate", 0.30))

        model = create_model(
            model_name=model_name,
            num_classes=5,
            pretrained=False,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.to(device)
        model.eval()
        models.append(model)

        print(f"Loaded model: {ckpt_path}")

    return models


def choose_fusion_key(test_df: pd.DataFrame, requested_key: str) -> Optional[str]:
    if requested_key and requested_key in test_df.columns:
        print(f"Using requested fusion key: {requested_key}")
        return requested_key

    for c in ["pig_id", "animal_id", "track_id", "instance_id", "object_id"]:
        if c in test_df.columns:
            print(f"Auto-detected fusion key: {c}")
            return c

    print("No pig-level fusion key found. Predict each row independently.")
    return None


@torch.no_grad()
def predict_submission(
    ckpt_paths: Sequence[str],
    df_test: pd.DataFrame,
    args,
    paths: dict,
    device: torch.device,
) -> Tuple[Path, Path]:
    models = load_fold_models(ckpt_paths, device)
    _, val_tf = make_transforms(args.img_size)

    fusion_key = choose_fusion_key(df_test, args.fusion_key)

    image_rows_by_image = {
        image_id: g.to_dict("records")
        for image_id, g in df_test.groupby("image_id")
    }

    image_cache: Dict[str, np.ndarray] = {}

    def get_image(image_id: str) -> np.ndarray:
        if image_id not in image_cache:
            img_path = find_image_path(paths["test_img_dir"], image_id)
            image_cache[image_id] = np.array(Image.open(img_path).convert("RGB"))
        return image_cache[image_id]

    preds: Dict[str, int] = {}
    prob_rows = []

    # If no fusion key, group by row_id so each row has its own TTA ensemble.
    group_col = fusion_key if fusion_key is not None else "row_id"

    for key, g_key in tqdm(df_test.groupby(group_col), total=df_test[group_col].nunique(), desc="Predict"):
        tensors = []
        row_ids = []
        image_ids = []
        domains = []

        for _, row in g_key.iterrows():
            image_id = str(row["image_id"])
            img = get_image(image_id)
            image_rows = image_rows_by_image[image_id]

            for pad_ratio in args.tta_pad_ratios:
                crop = context_soft_isolation_crop(
                    img_rgb=img,
                    target_row=row,
                    image_rows=image_rows,
                    pad_ratio=pad_ratio,
                    img_size=args.img_size,
                    soft_isolation=not args.no_soft_isolation,
                )
                x = val_tf(Image.fromarray(crop))
                tensors.append(x)
                row_ids.append(str(row["row_id"]))
                image_ids.append(image_id)
                domains.append(row["domain"])

        if not tensors:
            continue

        X = torch.stack(tensors, dim=0)
        probs_accum = []

        for start in range(0, X.size(0), args.infer_bs):
            xb = X[start:start + args.infer_bs].to(device, non_blocking=True)
            batch_prob = None

            with torch.amp.autocast("cuda", enabled=args.use_amp):
                for model in models:
                    logits = model(xb)
                    probs = torch.softmax(logits, dim=1)
                    if batch_prob is None:
                        batch_prob = probs
                    else:
                        batch_prob += probs

                batch_prob = batch_prob / len(models)

            probs_accum.append(batch_prob.detach().cpu().numpy().astype(np.float64))

        probs_all = np.concatenate(probs_accum, axis=0)

        # Average over TTA and optionally over views in same fusion group.
        fused = probs_all.mean(axis=0)
        pred = int(np.argmax(fused))

        unique_rids = list(dict.fromkeys(row_ids))
        for rid in unique_rids:
            preds[rid] = pred
            first_idx = row_ids.index(rid)
            prob_rows.append(
                {
                    "row_id": rid,
                    "fusion_key": key,
                    "image_id": image_ids[first_idx],
                    "domain": domains[first_idx],
                    "pred_class": pred,
                    **{f"prob_{i}": float(fused[i]) for i in range(5)},
                }
            )

    prob_df = pd.DataFrame(prob_rows)
    prob_path = paths["output_dir"] / "debug_probs.csv"
    prob_df.to_csv(prob_path, index=False)

    sample_path = paths["sample_submission"]
    if sample_path.exists():
        sub = pd.read_csv(sample_path)
        id_col = sub.columns[0]
        pred_col = sub.columns[1]
        sub[id_col] = sub[id_col].astype(str)

        missing = set(sub[id_col]) - set(preds.keys())
        if missing:
            raise RuntimeError(f"Missing predictions for row_ids, examples: {list(missing)[:5]}")

        sub[pred_col] = sub[id_col].map(preds).astype(int)
    else:
        sub = df_test[["row_id"]].copy()
        sub["class_id"] = sub["row_id"].astype(str).map(preds).astype(int)

    sub_path = paths["output_dir"] / "submission.csv"
    sub.to_csv(sub_path, index=False)

    print("\nSubmission saved:", sub_path)
    print("Debug probabilities saved:", prob_path)
    print("\nSubmission class distribution:")
    print(sub.iloc[:, 1].value_counts().sort_index())

    del models
    torch.cuda.empty_cache()

    return sub_path, prob_path


def final_submission_check(sub_path: Path, test_csv: Path) -> None:
    test = pd.read_csv(test_csv)
    sub = pd.read_csv(sub_path)

    print("\nFinal submission check")
    print("test shape:", test.shape)
    print("submission shape:", sub.shape)
    print("columns:", list(sub.columns))
    print("missing:")
    print(sub.isna().sum())
    print("duplicated row_id:", sub.iloc[:, 0].duplicated().sum())

    same_len = len(test) == len(sub)
    same_set = set(test["row_id"].astype(str)) == set(sub.iloc[:, 0].astype(str))
    same_order = test["row_id"].astype(str).equals(sub.iloc[:, 0].astype(str))

    print("same length:", same_len)
    print("same row_id set:", same_set)
    print("same row_id order:", same_order)

    if not same_len:
        raise RuntimeError("Submission length mismatch.")
    if not same_set:
        raise RuntimeError("Submission row_id set mismatch.")


# ----------------------------
# Main
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("./pig_effv2_outputs"))
    parser.add_argument("--train-id", type=int, default=2)

    parser.add_argument("--model", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="0-indexed folds to run. Default: all folds.")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--drop-rate", type=float, default=0.40)
    parser.add_argument("--drop-path-rate", type=float, default=0.30)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--train-pad-ratios", type=float, nargs="+", default=[0.06, 0.12, 0.20, 0.30])
    parser.add_argument("--val-pad-ratio", type=float, default=0.12)
    parser.add_argument("--tta-pad-ratios", type=float, nargs="+", default=[0.06, 0.12, 0.20, 0.30, 0.40])

    parser.add_argument("--changes-csv", type=Path, default=None)
    parser.add_argument(
        "--changes-url",
        type=str,
        default="https://drive.google.com/uc?export=download&id=1LxfkP-tpyXdLGt2DEJwrZJSSggVwYjv8",
    )

    parser.add_argument("--output-name", type=str, default="effv2_softiso_5fold_4090")
    parser.add_argument("--infer-bs", type=int, default=256)
    parser.add_argument("--fusion-key", type=str, default="")
    parser.add_argument("--no-soft-isolation", action="store_true")
    parser.add_argument("--no-cache-images", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Only run inference using --ckpts.")
    parser.add_argument("--ckpts", type=str, nargs="*", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.use_amp = torch.cuda.is_available() and (not args.no_amp)
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    paths = {
        "train_csv": args.data_dir / f"train{args.train_id}.csv",
        "train_img_dir": args.data_dir / f"train{args.train_id}_images",
        "test_csv": args.data_dir / "test.csv",
        "test_img_dir": args.data_dir / "test_images",
        "sample_submission": args.data_dir / "sample_submission.csv",
        "output_dir": args.out_dir / f"output_{args.output_name}",
        "model_dir": args.out_dir / f"models_{args.output_name}",
    }

    for k in ["train_csv", "train_img_dir", "test_csv", "test_img_dir"]:
        print(f"{k}: {paths[k]} exists={paths[k].exists()}")
        if not paths[k].exists():
            raise FileNotFoundError(paths[k])

    reset_dir(paths["output_dir"])
    ensure_dir(paths["model_dir"])

    df = prepare_df(paths["train_csv"], is_train=True)
    test_df = prepare_df(paths["test_csv"], is_train=False)

    changes_csv = maybe_download_changes_csv(args.changes_csv, args.changes_url, paths["output_dir"])
    df = apply_label_corrections(df, changes_csv)
    df = remove_known_bad_samples(df)

    print_data_diagnosis(df, test_df)

    ckpt_paths: List[str] = []

    if not args.skip_train:
        groups = df["image_id"].astype(str).values
        gkf = GroupKFold(n_splits=args.n_splits)

        all_results = []
        folds_to_run = set(args.folds) if args.folds is not None else set(range(args.n_splits))

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, df["class_id"], groups=groups)):
            if fold not in folds_to_run:
                print(f"Skip fold {fold + 1}")
                continue

            train_df = df.iloc[tr_idx].reset_index(drop=True)
            val_df = df.iloc[va_idx].reset_index(drop=True)

            result = train_one_fold(fold, train_df, val_df, args, paths, device)
            all_results.append(result)
            ckpt_paths.append(result["best_path"])

        result_df = pd.DataFrame(all_results)
        result_path = paths["output_dir"] / "fold_results.csv"
        result_df.to_csv(result_path, index=False)

        print("\nFold results:")
        print(result_df)

    else:
        if not args.ckpts:
            raise ValueError("--skip-train requires --ckpts")
        ckpt_paths = args.ckpts

    sub_path, prob_path = predict_submission(ckpt_paths, test_df, args, paths, device)
    final_submission_check(sub_path, paths["test_csv"])

    run_info = {
        "args": vars(args),
        "ckpt_paths": ckpt_paths,
        "submission": str(sub_path),
        "debug_probs": str(prob_path),
    }
    with open(paths["output_dir"] / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("\nAll done.")
    print("Submit:", sub_path)


if __name__ == "__main__":
    main()


   
