import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from backend.app.config import BASE_DIR
from backend.app.ml.traffic_sign_cnn import ARCHITECTURE_VERSION, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, TrafficSignCNN
from backend.app.services.model_service import MODEL_PATH, TRAFFIC_SIGN_CLASSES


def evaluate(model, loader, device, class_names: list[str]) -> dict:
    model.eval()
    correct = 0
    total = 0
    class_totals = torch.zeros(len(class_names), dtype=torch.long)
    class_correct = torch.zeros(len(class_names), dtype=torch.long)
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
            for class_id in labels.unique():
                mask = labels == class_id
                class_totals[class_id] += int(mask.sum().item())
                class_correct[class_id] += int((predictions[mask] == labels[mask]).sum().item())
    return {
        "accuracy": correct / total if total else 0.0,
        "samples": total,
        "split": "stratified_holdout",
        "class_accuracy": {
            class_names[index]: float(class_correct[index] / class_totals[index])
            for index in range(len(class_names))
            if class_totals[index] > 0
        },
    }


def stratified_split(targets: list[int], validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[target].append(index)
    rng = np.random.default_rng(seed)
    training_indices = []
    validation_indices = []
    for indices in by_class.values():
        shuffled = rng.permutation(indices)
        validation_count = max(1, round(len(indices) * validation_fraction))
        validation_indices.extend(shuffled[:validation_count].tolist())
        training_indices.extend(shuffled[validation_count:].tolist())
    return training_indices, validation_indices


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    training_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

    local_training_dir = args.data_dir / "Train"
    if local_training_dir.is_dir():
        training_source = datasets.ImageFolder(root=str(local_training_dir), transform=training_transform)
        validation_source = datasets.ImageFolder(root=str(local_training_dir), transform=evaluation_transform)
        if training_source.class_to_idx != validation_source.class_to_idx:
            raise ValueError("Training and validation class mappings do not match.")
        expected_class_ids = [f"{class_id:05d}" for class_id in range(len(TRAFFIC_SIGN_CLASSES))]
        if training_source.classes != expected_class_ids:
            raise ValueError("The local Train folder must contain all GTSRB class folders 00000 through 00042.")
        if len(training_source) != len(validation_source):
            raise ValueError("The local training dataset changed while it was being inspected.")
        class_names = TRAFFIC_SIGN_CLASSES
        training_indices, validation_indices = stratified_split(
            training_source.targets,
            args.validation_fraction,
            args.seed,
        )
        training_data = Subset(training_source, training_indices)
        validation_data = Subset(validation_source, validation_indices)
        source_split = "stratified_holdout_from_local_train"
    else:
        training_data = datasets.GTSRB(root=str(args.data_dir), split="train", download=True, transform=training_transform)
        validation_data = datasets.GTSRB(root=str(args.data_dir), split="test", download=True, transform=evaluation_transform)
        class_names = TRAFFIC_SIGN_CLASSES
        source_split = "official_test"

    generator = torch.Generator().manual_seed(args.seed)
    if args.max_train_samples:
        indices = torch.randperm(len(training_data), generator=generator)[:args.max_train_samples].tolist()
        training_data = torch.utils.data.Subset(training_data, indices)
    if args.max_test_samples:
        indices = torch.randperm(len(validation_data), generator=generator)[:args.max_test_samples].tolist()
        validation_data = torch.utils.data.Subset(validation_data, indices)

    sampled_targets = [training_data.dataset.targets[index] for index in training_data.indices] if isinstance(training_data, Subset) and isinstance(training_data.dataset, datasets.ImageFolder) else None
    if sampled_targets is not None:
        class_counts = np.bincount(sampled_targets, minlength=len(TRAFFIC_SIGN_CLASSES))
        sample_weights = [1.0 / np.sqrt(class_counts[target]) for target in sampled_targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)
        training_loader = DataLoader(training_data, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    else:
        training_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = TrafficSignCNN(num_classes=len(TRAFFIC_SIGN_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        samples = 0
        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * len(labels)
            samples += len(labels)
        scheduler.step()
        validation = evaluate(model, validation_loader, device, class_names)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({
            "epoch": epoch + 1,
            "epochs": args.epochs,
            "training_loss": loss_total / max(samples, 1),
            "validation_accuracy": validation["accuracy"],
            "best_validation_accuracy": best_accuracy,
        }), flush=True)

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    evaluation = evaluate(model, validation_loader, device, class_names)
    evaluation["split"] = source_split
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save({
        "architecture": ARCHITECTURE_VERSION,
        "training_complete": True,
        "model_state_dict": model.cpu().state_dict(),
        "class_names": TRAFFIC_SIGN_CLASSES,
        "image_size": IMAGE_SIZE,
        "normalize_mean": NORMALIZE_MEAN,
        "normalize_std": NORMALIZE_STD,
        "evaluation": evaluation,
    }, temporary_path)
    temporary_path.replace(args.output)
    result = {
        "checkpoint": str(args.output),
        "architecture": ARCHITECTURE_VERSION,
        "evaluation": evaluation,
        "device": str(device),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a traffic-sign CNN and evaluate it on a held-out GTSRB split.")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data" / "gtsrb")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.0008)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional subset limit for smoke tests; 0 uses the full training split.")
    parser.add_argument("--max-test-samples", type=int, default=0, help="Optional subset limit for smoke tests; 0 uses the full test split.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    if args.max_train_samples < 0 or args.max_test_samples < 0:
        parser.error("sample limits must be zero (unlimited) or positive.")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1.")
    return args


if __name__ == "__main__":
    train(parse_args())
