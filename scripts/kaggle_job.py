"""Prepare, submit, check, and fetch the private Kaggle training job."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".kaggle-run"
SLUG = "air-canvas-100-v3"


def prepare(username):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Use your Kaggle username, not an email or profile URL")
    BUILD.mkdir(exist_ok=True)
    # Kaggle uploads the code_file, not the rest of our Python package. Bundle
    # the shared sources directly so inference and training cannot drift apart.
    sources = []
    for relative in ["core/preprocess.py", "core/model.py", "training/train.py"]:
        source = (ROOT / relative).read_text()
        source = "\n".join(line for line in source.splitlines()
                           if not line.startswith("from core."))
        sources.append(f"# ---- {relative} ----\n{source}")
    classes = json.loads((ROOT / "training/classes.json").read_text())
    code = "\n\n".join(sources)
    code = code.replace('json.loads(Path(__file__).with_name("classes.json").read_text())', repr(classes))
    compile(code, "train.py", "exec")
    (BUILD / "train.py").write_text(code + "\n")
    metadata = {
        "id": f"{username}/{SLUG}", "title": "Air Canvas 100 Classes v3",
        "code_file": "train.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }
    (BUILD / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata["id"]


def cli(*args):
    executable = Path(sys.executable).with_name("kaggle")
    subprocess.run([str(executable), *args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "push", "status", "fetch"])
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    if args.action in ["prepare", "push"]:
        kernel_id = prepare(args.username)
    else:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
            parser.error("Invalid Kaggle username")
        kernel_id = f"{args.username}/{SLUG}"
    if args.action == "push":
        cli("kernels", "push", "-p", str(BUILD), "--accelerator", "NvidiaTeslaT4", "--timeout", "7200")
    elif args.action == "status":
        cli("kernels", "status", kernel_id)
    elif args.action == "fetch":
        output = ROOT / "runs" / SLUG
        cli("kernels", "output", kernel_id, "-p", str(output))
        checkpoint = output / "quickdraw_cnn.pth"
        if not checkpoint.is_file():
            raise SystemExit("No model in the output yet. Check the job status and logs.")
        # Load and validate before replacing an existing working model.
        sys.path.insert(0, str(ROOT))
        from core.guesser import SketchGuesser
        SketchGuesser(checkpoint)
        destination = ROOT / "models" / "quickdraw_cnn.pth"
        temporary = destination.with_suffix(".tmp")
        shutil.copy2(checkpoint, temporary)
        temporary.replace(destination)
        print("Model installed. Start the game with: python app.py")
    print(f"https://www.kaggle.com/code/{kernel_id}")


if __name__ == "__main__":
    main()
