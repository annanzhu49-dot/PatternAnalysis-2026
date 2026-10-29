"""Train and evaluate the standard 3D U-Net baseline on HipMRI."""

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset, get_data_pairs, split_by_patient
from modules import UNet3D


NUM_CLASSES = 6
TARGET_SHAPE = (64, 128, 128)


def calculate_metrics(logits, targets):
    """Return per-class Dice and IoU, excluding empty GT/pred classes."""
    predictions = logits.argmax(dim=1)
    dice_scores = []
    iou_scores = []

    for class_id in range(NUM_CLASSES):
        pred = predictions == class_id
        target = targets == class_id

        intersection = (pred & target).sum().item()
        pred_count = pred.sum().item()
        target_count = target.sum().item()
        union = pred_count + target_count - intersection

        if pred_count + target_count == 0:
            dice_scores.append(float("nan"))
        else:
            dice_scores.append(
                2.0 * intersection / (pred_count + target_count)
            )

        if union == 0:
            iou_scores.append(float("nan"))
        else:
            iou_scores.append(intersection / union)

    return dice_scores, iou_scores


def run_epoch(model, loader, criterion, optimizer, device, max_batches):
    """Run one training or evaluation epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    count = 0
    dice_total = [0.0] * NUM_CLASSES
    iou_total = [0.0] * NUM_CLASSES
    metric_counts = [0] * NUM_CLASSES

    for batch_index, (images, masks) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        dice, iou = calculate_metrics(outputs.detach(), masks)

        for class_id in range(NUM_CLASSES):
            if dice[class_id] == dice[class_id]:
                dice_total[class_id] += dice[class_id]
                iou_total[class_id] += iou[class_id]
                metric_counts[class_id] += 1

        total_loss += loss.item()
        count += 1

        print(
            f"{'Train' if training else 'Val'} "
            f"batch {count}: loss={loss.item():.4f}",
            flush=True,
        )

    if count == 0:
        raise RuntimeError("No batches processed.")

    mean_dice = [
        dice_total[i] / metric_counts[i] if metric_counts[i] else float("nan")
        for i in range(NUM_CLASSES)
    ]
    mean_iou = [
        iou_total[i] / metric_counts[i] if metric_counts[i] else float("nan")
        for i in range(NUM_CLASSES)
    ]

    return total_loss / count, mean_dice, mean_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--output-dir", default="results/baseline")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: submit through Slurm.")

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = get_data_pairs()
    train_pairs, val_pairs, test_pairs = split_by_patient(pairs, seed=42)

    print(
        f"Data split: {len(train_pairs)} train, "
        f"{len(val_pairs)} val, {len(test_pairs)} test",
        flush=True,
    )

    train_loader = DataLoader(
        HipMRIDataset(train_pairs, target_shape=TARGET_SHAPE),
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        HipMRIDataset(val_pairs, target_shape=TARGET_SHAPE),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = UNet3D(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_channels=8,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    history_path = output_dir / "history.csv"

    with history_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["epoch", "train_loss", "val_loss"]
            + [f"val_dice_{i}" for i in range(NUM_CLASSES)]
            + [f"val_iou_{i}" for i in range(NUM_CLASSES)]
        )

        for epoch in range(1, args.epochs + 1):
            print(f"\n===== Epoch {epoch}/{args.epochs} =====", flush=True)

            train_loss, _, _ = run_epoch(
                model, train_loader, criterion, optimizer,
                device, args.max_train_batches,
            )

            val_loss, val_dice, val_iou = run_epoch(
                model, val_loader, criterion, None,
                device, args.max_val_batches,
            )

            writer.writerow(
                [epoch, train_loss, val_loss] + val_dice + val_iou
            )
            file.flush()

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}",
                flush=True,
            )
            print("Validation Dice:", val_dice, flush=True)
            print("Validation IoU:", val_iou, flush=True)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    output_dir / "best_model.pt",
                )

    print("Peak allocated GPU memory (GB):",
          round(torch.cuda.max_memory_allocated() / 1024**3, 3))
    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
