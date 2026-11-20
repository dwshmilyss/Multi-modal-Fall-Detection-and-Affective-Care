"""Download optional HSEmotion ONNX weights for the benchmark."""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path


MODEL_METADATA = {
    "enet_b0_8_best_vgaf": dict(
        url=(
        "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/"
        "models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx"
        ),
        size_bytes=16_039_595,
    ),
    "enet_b2_8": dict(
        url=(
        "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/"
        "models/affectnet_emotions/onnx/enet_b2_8.onnx"
        ),
        size_bytes=30_779_724,
    ),
}
MODEL_NAMES = tuple(sorted(MODEL_METADATA))


def download_models(
    models: list[str],
    target_dir: Path,
    retry_attempts: int = 3,
    retry_delay_seconds: float = 2.0,
) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name in models:
        metadata = MODEL_METADATA[name]
        target = target_dir / f"{name}.onnx"
        is_complete = target.is_file() and target.stat().st_size == metadata["size_bytes"]
        if not is_complete:
            if target.exists():
                print(f"Removing incomplete model: {target}")
                target.unlink()
            partial = target.with_suffix(target.suffix + ".part")
            print(f"Downloading {name} ...")
            last_error: Exception | None = None
            for attempt in range(1, retry_attempts + 1):
                try:
                    urllib.request.urlretrieve(metadata["url"], partial)
                    if partial.stat().st_size != metadata["size_bytes"]:
                        raise RuntimeError(
                            f"Downloaded {name} has {partial.stat().st_size} bytes; "
                            f"expected {metadata['size_bytes']}"
                        )
                    partial.replace(target)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if partial.exists():
                        partial.unlink()
                    if attempt < retry_attempts:
                        print(f"Download attempt {attempt} failed for {name}: {exc}; retrying...")
                        time.sleep(retry_delay_seconds)
            if last_error is not None:
                if partial.exists():
                    partial.unlink()
                raise RuntimeError(f"Could not download {name} after {retry_attempts} attempts") from last_error
        downloaded.append(target)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_NAMES),
        choices=MODEL_METADATA,
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "affectnet_emotions" / "onnx",
    )
    args = parser.parse_args()
    for path in download_models(args.models, args.target_dir):
        print(f"Ready: {path}")


if __name__ == "__main__":
    main()
