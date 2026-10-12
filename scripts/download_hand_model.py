"""Download Google's pretrained hand model once; gameplay is offline."""

from pathlib import Path
import hashlib
from urllib.request import urlopen

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DESTINATION = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def main():
    if DESTINATION.is_file() and hashlib.sha256(DESTINATION.read_bytes()).hexdigest() == SHA256:
        print(f"Hand model already exists: {DESTINATION}")
        return
    DESTINATION.parent.mkdir(exist_ok=True)
    temporary = DESTINATION.with_suffix(".part")
    try:
        with urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("The hand model download failed its checksum check")
        temporary.replace(DESTINATION)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded {DESTINATION}")
    print(f"SHA256: {hashlib.sha256(DESTINATION.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
