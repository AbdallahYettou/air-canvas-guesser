"""Run on Kaggle GPU. Saves the best model and a resume checkpoint each epoch."""

import argparse
import json
import random
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.model import create_model
from core.preprocess import IMAGE_SIZE, PREPROCESS_VERSION, prepare_sketch, rasterize_strokes


def load_category(label, count, seed):
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{quote(label, safe='')}.ndjson"
    # Read a bounded prefix, then shuffle it. We do not claim a random global sample.
    images, ids, seen = [], [], set()
    drawing_rng = np.random.default_rng(seed)
    with urlopen(url, timeout=90) as response:
        for line in response:
            record = json.loads(line)
            key = str(record["key_id"])
            if not record.get("recognized") or key in seen:
                continue
            # Vary stroke thickness while keeping the app brush fixed.
            image = prepare_sketch(rasterize_strokes(record["drawing"],
                                   thickness=int(drawing_rng.choice([3, 5, 7]))))
            if image is None:
                continue
            seen.add(key)
            images.append(image)
            ids.append(key)
            if len(images) >= count:
                break
    if len(images) < count:
        raise RuntimeError(f"Only found {len(images)} usable drawings for {label}; wanted {count}")
    order = np.random.default_rng(seed).permutation(len(images))
    return np.stack(images)[order], np.asarray(ids)[order]


class SketchDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images, self.labels, self.augment = images, labels, augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.augment:
            matrix = cv2.getRotationMatrix2D((31.5, 31.5), random.uniform(-12, 12), random.uniform(0.9, 1.08))
            matrix[:, 2] += np.random.uniform(-3, 3, 2)
            image = cv2.warpAffine(image, matrix, (IMAGE_SIZE, IMAGE_SIZE))
            if random.random() < 0.2:
                image = cv2.dilate(image, np.ones((2, 2), np.uint8))
        return torch.from_numpy(image.copy()).unsqueeze(0).float() / 255, int(self.labels[index])


def evaluate(model, loader, device, classes):
    model.eval()
    correct, top3, count = 0, 0, 0
    confusion = np.zeros((classes, classes), dtype=np.int64)
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = logits.topk(min(3, classes), dim=1).indices.cpu()
            correct += (predictions[:, 0] == labels).sum().item()
            top3 += (predictions == labels[:, None]).any(dim=1).sum().item()
            count += len(labels)
            np.add.at(confusion, (labels.numpy(), predictions[:, 0].numpy()), 1)
    return {"top1": correct / count, "top3": top3 / count, "samples": count,
            "confusion_matrix": confusion.tolist()}


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train_candidate(architecture, classes, loaders, args, resume=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = create_model(architecture, len(classes)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    metadata = {"classes": classes, "image_size": IMAGE_SIZE, "preprocess_version": PREPROCESS_VERSION,
                "seed": args.seed, "samples_per_class": args.samples, "architecture": architecture,
                "input_format": "grayscale-mask-or-BGRA-alpha",
                "training_brush_widths": [3, 5, 7]}
    folder = args.output / architecture
    folder.mkdir(parents=True, exist_ok=True)
    best, best_state, start_epoch, history = -1.0, None, 0, []
    if resume is not None:
        for key in ["architecture", "classes", "image_size", "preprocess_version", "seed", "samples_per_class"]:
            if resume[key] != metadata[key]:
                raise ValueError(f"Recovery checkpoint does not match {key}")
        if resume["total_epochs"] != args.epochs:
            raise ValueError("Keep --epochs unchanged when recovering a run")
        model.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        best, best_state = resume["best_accuracy"], resume["best_state_dict"]
        start_epoch, history = resume["epoch"], resume["history"]
        random.setstate(resume["python_rng"])
        np.random.set_state(("MT19937", np.array(resume["numpy_rng"], dtype=np.uint32), resume["numpy_pos"], 0, 0.0))
        torch.set_rng_state(resume["torch_rng"])
        torch.cuda.set_rng_state_all(resume["cuda_rng"])
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.monotonic()
        model.train()
        total_loss = 0.0
        for images, labels in loaders["train"]:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_fn(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
        scheduler.step()
        validation = evaluate(model, loaders["validation"], "cuda", len(classes))
        row = {"architecture": architecture, "epoch": epoch + 1,
               "loss": total_loss / len(loaders["train"].dataset),
               "validation_top1": validation["top1"], "validation_top3": validation["top3"],
               "seconds": time.monotonic() - epoch_start}
        history.append(row)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if validation["top1"] > best:
            best, best_state = validation["top1"], state
            atomic_save({**metadata, "state_dict": state}, folder / "best.pth")
        numpy_rng = np.random.get_state()
        recovery = {**metadata, "state_dict": state, "best_state_dict": best_state,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "total_epochs": args.epochs,
                    "best_accuracy": best, "history": history, "python_rng": random.getstate(),
                    "numpy_rng": numpy_rng[1].tolist(), "numpy_pos": numpy_rng[2],
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        atomic_save(recovery, folder / "last_checkpoint.pth")
        (folder / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        print(f"{architecture}: approx. {(args.epochs - epoch - 1) * row['seconds'] / 60:.1f} minutes left", flush=True)
    model.load_state_dict(best_state)
    atomic_save({**metadata, "state_dict": best_state}, folder / "best.pth")
    result = {"architecture": architecture, "validation_top1": best,
              "parameters": sum(p.numel() for p in model.parameters()), "history": history}
    (folder / "summary.json").write_text(json.dumps(result, indent=2))
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return result


def fit_temperature(model, loader):
    logits, labels = [], []
    model.eval()
    with torch.inference_mode():
        for images, target in loader:
            logits.append(model(images.cuda()).cpu())
            labels.append(target)
    logits, labels = torch.cat(logits), torch.cat(labels)
    candidates = torch.linspace(0.5, 3.0, 26)
    losses = [torch.nn.functional.cross_entropy(logits / value, labels).item() for value in candidates]
    best = int(np.argmin(losses))
    return float(candidates[best]), {"validation_nll_before": torch.nn.functional.cross_entropy(logits, labels).item(),
                                    "validation_nll_after": losses[best]}


def main():
    parser = argparse.ArgumentParser(description="Compare two sketch networks on the same 100-class split on Kaggle.")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--architectures", nargs="+", default=["SketchCNN-v1", "SketchResNet-v1"],
                        choices=["SketchCNN-v1", "SketchResNet-v1"])
    parser.add_argument("--resume", type=Path, help="Recover a single candidate; select it with --architectures")
    args = parser.parse_args()
    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise SystemExit("Training is configured for a Kaggle GPU session. Enable GPU and Internet there.")
    if args.samples < 20 or args.epochs < 1:
        raise SystemExit("Use at least 20 samples per class and one epoch.")
    if args.resume and len(args.architectures) != 1:
        raise SystemExit("Resume one model at a time with --architectures ARCHITECTURE")
    cv2.setNumThreads(0)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    classes = json.loads(Path(__file__).with_name("classes.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    parts = {name: ([], []) for name in ["train", "validation", "test"]}
    split_ids = {name: {} for name in parts}
    for index, label in enumerate(classes):
        for attempt in range(3):
            try:
                images, ids = load_category(label, args.samples, args.seed + index)
                break
            except (OSError, ValueError) as exc:
                if attempt == 2:
                    raise
                print(f"Retrying {label}: {exc}", flush=True)
                time.sleep(2 * (attempt + 1))
        first, second = int(len(images) * 0.8), int(len(images) * 0.9)
        for name, interval in [("train", slice(0, first)), ("validation", slice(first, second)), ("test", slice(second, None))]:
            subset = images[interval]
            parts[name][0].append(subset)
            parts[name][1].extend([index] * len(subset))
            split_ids[name][label] = ids[interval].tolist()
        print(f"Prepared {label}: {len(images)} sketches ({index + 1}/{len(classes)})", flush=True)
    (args.output / "split_ids.json").write_text(json.dumps(split_ids))
    loaders = {}
    for name, (images, labels) in parts.items():
        dataset = SketchDataset(np.concatenate(images), np.asarray(labels), augment=name == "train")
        loaders[name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train", num_workers=0, pin_memory=True)
    del parts
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    comparisons = []
    for architecture in args.architectures:
        print(f"Training {architecture} on {len(classes)} classes", flush=True)
        comparisons.append(train_candidate(architecture, classes, loaders, args, resume))
        (args.output / "comparison.json").write_text(json.dumps(comparisons, indent=2))
    winner = max(comparisons, key=lambda result: result["validation_top1"])
    # Choose using validation only; evaluate held-out data after making that choice.
    for result in comparisons:
        checkpoint = torch.load(args.output / result["architecture"] / "best.pth", map_location="cpu", weights_only=True)
        model = create_model(result["architecture"], len(classes)).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        result["test"] = evaluate(model, loaders["test"], "cuda", len(classes))
        if result is winner:
            temperature, calibration = fit_temperature(model, loaders["validation"])
            atomic_save({**checkpoint, "temperature": temperature}, args.output / "quickdraw_cnn.pth")
            selected_checkpoint = checkpoint
        del model
        torch.cuda.empty_cache()
    report = {key: value for key, value in selected_checkpoint.items() if key != "state_dict"}
    report.update({"test": winner["test"], "best_validation_top1": winner["validation_top1"],
                   "temperature": temperature, "calibration": calibration, "parameters": winner["parameters"],
                   "comparison": comparisons, "seconds_total": time.monotonic() - started,
                   "device": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__),
                   "numpy_version": np.__version__, "dataset": "Google Quick, Draw! simplified vectors (CC BY 4.0)",
                   "sampling": "first recognized unique sketches per class; seeded 80/10/10 split"})
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    (args.output / "comparison.json").write_text(json.dumps(comparisons, indent=2))
    print(f"Selected {winner['architecture']}; held-out top-1 {winner['test']['top1']:.1%}, top-3 {winner['test']['top3']:.1%}", flush=True)


if __name__ == "__main__":
    main()
