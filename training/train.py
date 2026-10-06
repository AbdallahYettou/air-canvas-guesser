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

from core.model import SketchCNN
from core.preprocess import IMAGE_SIZE, PREPROCESS_VERSION, prepare_sketch, rasterize_strokes


def load_category(label, count, seed):
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{quote(label, safe='')}.ndjson"
    # Read a bounded prefix, then shuffle it. We do not claim a random global sample.
    images, ids, seen = [], [], set()
    with urlopen(url, timeout=90) as response:
        for line in response:
            record = json.loads(line)
            key = str(record["key_id"])
            if not record.get("recognized") or key in seen:
                continue
            image = prepare_sketch(rasterize_strokes(record["drawing"]))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise SystemExit("Training is configured for a Kaggle GPU session. Enable GPU and Internet there.")
    if args.samples < 20 or args.epochs < 1:
        raise SystemExit("Use at least 20 samples per class and one epoch.")
    cv2.setNumThreads(0)
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    classes = json.loads(Path(__file__).with_name("classes.json").read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    parts = {name: ([], []) for name in ["train", "validation", "test"]}
    split_ids = {name: {} for name in parts}
    for index, label in enumerate(classes):
        images, ids = load_category(label, args.samples, args.seed + index)
        first, second = int(len(images) * 0.8), int(len(images) * 0.9)
        for name, interval in [("train", slice(0, first)), ("validation", slice(first, second)), ("test", slice(second, None))]:
            subset = images[interval]
            parts[name][0].append(subset)
            parts[name][1].extend([index] * len(subset))
            split_ids[name][label] = ids[interval].tolist()
        print(f"Prepared {label}: {len(images)} sketches ({index + 1}/{len(classes)})", flush=True)
    (output / "split_ids.json").write_text(json.dumps(split_ids))
    loaders = {}
    for name, (images, labels) in parts.items():
        dataset = SketchDataset(np.concatenate(images), np.asarray(labels), augment=name == "train")
        loaders[name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train", num_workers=0, pin_memory=True)
    del parts
    device = torch.device("cuda")
    model = SketchCNN(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    best, start_epoch, history = -1.0, 0, []
    metadata = {"classes": classes, "image_size": IMAGE_SIZE, "preprocess_version": PREPROCESS_VERSION,
                "seed": args.seed, "samples_per_class": args.samples, "architecture": "SketchCNN-v1"}
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key in ["classes", "image_size", "preprocess_version", "seed", "samples_per_class"]:
            if saved[key] != metadata[key]:
                raise ValueError(f"Resume checkpoint does not match {key}")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best, history = saved["epoch"], saved["best_accuracy"], saved["history"]
        if start_epoch >= args.epochs:
            raise ValueError("--epochs must exceed the resumed epoch")
        atomic_save({**metadata, "state_dict": saved["best_state_dict"]}, output / "quickdraw_cnn.pth")
        random.setstate(saved["python_rng"])
        np.random.set_state(("MT19937", np.array(saved["numpy_rng"], dtype=np.uint32), saved["numpy_pos"], 0, 0.0))
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    best_state = None
    if args.resume:
        best_state = saved["best_state_dict"]
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.monotonic()
        model.train()
        total_loss = 0.0
        for images, labels in loaders["train"]:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        validation = evaluate(model, loaders["validation"], device, len(classes))
        row = {"epoch": epoch + 1, "loss": total_loss / len(loaders["train"].dataset),
               "validation_top1": validation["top1"], "validation_top3": validation["top3"],
               "seconds": time.monotonic() - epoch_start}
        history.append(row)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if validation["top1"] > best:
            best, best_state = validation["top1"], state
            atomic_save({**metadata, "state_dict": state}, output / "quickdraw_cnn.pth")
        numpy_rng = np.random.get_state()
        atomic_save({**metadata, "state_dict": state, "best_state_dict": best_state,
                     "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                     "best_accuracy": best, "history": history, "python_rng": random.getstate(),
                     "numpy_rng": numpy_rng[1].tolist(), "numpy_pos": numpy_rng[2],
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, output / "last_checkpoint.pth")
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        print(f"Approx. training time left: {(args.epochs - epoch - 1) * row['seconds'] / 60:.1f} minutes", flush=True)
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], device, len(classes))
    report = {**metadata, "best_validation_top1": best, "test": test_metrics,
              "seconds_total": time.monotonic() - started, "device": torch.cuda.get_device_name(0),
              "parameters": sum(p.numel() for p in model.parameters()),
              "torch_version": str(torch.__version__), "numpy_version": np.__version__,
              "sampling": "first recognized unique sketches per class; seeded 80/10/10 split",
              "dataset": "Google Quick, Draw! simplified vectors (CC BY 4.0)"}
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"Done. Held-out top-1: {test_metrics['top1']:.1%}; top-3: {test_metrics['top3']:.1%}", flush=True)


if __name__ == "__main__":
    main()
