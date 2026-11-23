# -*- coding: utf-8 -*-
"""Download the official URFD cam0 videos and annotation CSVs."""

import argparse
import json
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deidao.fall_metrics import build_urfd_manifest


URFD_BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
ANNOTATION_FILES = (
    "urfall-cam0-falls.csv",
    "urfall-cam0-adls.csv",
)


@dataclass(frozen=True)
class DownloadItem:
    """One official remote file and its absolute local destination."""

    remote_name: str
    local_path: Path


def urfd_file_url(remote_name: str) -> str:
    """Return the official download URL for a URFD data file."""

    return f"{URFD_BASE_URL}{remote_name}"


def build_download_plan(output_dir: str | Path) -> list[DownloadItem]:
    """Build the two-CSV and 70-cam0-video download plan."""

    output_path = Path(output_dir).resolve()
    items = [
        DownloadItem(remote_name, output_path / remote_name)
        for remote_name in ANNOTATION_FILES
    ]
    items.extend(
        DownloadItem(
            f"fall-{number:02d}-cam0.mp4",
            output_path / "fall" / f"fall-{number:02d}-cam0.mp4",
        )
        for number in range(1, 31)
    )
    items.extend(
        DownloadItem(
            f"adl-{number:02d}-cam0.mp4",
            output_path / "adl" / f"adl-{number:02d}-cam0.mp4",
        )
        for number in range(1, 41)
    )
    return items


def _is_complete_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _part_path(local_path: Path) -> Path:
    return local_path.with_suffix(f"{local_path.suffix}.part")


def _download_file(url: str, part_path: Path) -> None:
    """Copy a complete HTTP response to a staging file."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "deidao-urfd-downloader/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if getattr(response, "status", 200) != 200:
            raise URLError(f"unexpected HTTP status {response.status}")
        with part_path.open("wb") as part_file:
            shutil.copyfileobj(response, part_file, length=1024 * 1024)


def _write_manifest(dataset_dir: Path) -> dict:
    manifest = build_urfd_manifest(
        dataset_dir,
        dataset_dir / "urfall-cam0-falls.csv",
        dataset_dir / "urfall-cam0-adls.csv",
        fps=30.0,
    )
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def download_urfd(
    output_dir: str | Path,
    retries: int = 3,
    retry_delay_seconds: float = 2.0,
    create_manifest: bool = True,
) -> dict:
    """Download each planned file independently, recording files that remain unavailable."""

    output_path = Path(output_dir).resolve()
    attempt_count = max(1, retries)
    delay_seconds = max(0.0, retry_delay_seconds)
    downloaded = 0
    skipped = 0
    failed = []

    for item in build_download_plan(output_path):
        if _is_complete_file(item.local_path):
            skipped += 1
            continue

        item.local_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = _part_path(item.local_path)
        for attempt in range(attempt_count):
            try:
                _download_file(urfd_file_url(item.remote_name), part_path)
            except (OSError, URLError) as error:
                if attempt + 1 < attempt_count:
                    time.sleep(delay_seconds)
                else:
                    try:
                        part_path.unlink(missing_ok=True)
                    except OSError as cleanup_error:
                        print(
                            f"Could not remove {part_path}: {cleanup_error}",
                            file=sys.stderr,
                        )
                    failed.append(item.remote_name)
                    print(
                        f"Failed after {attempt_count} attempts: "
                        f"{item.remote_name}: {error}",
                        file=sys.stderr,
                    )
            else:
                part_path.replace(item.local_path)
                downloaded += 1
                break

    manifest = None
    if create_manifest and not failed:
        manifest = _write_manifest(output_path)

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "manifest": manifest,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="test_videos/evaluation/urfd",
        help="dataset output directory",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument(
        "--no-manifest",
        action="store_false",
        dest="create_manifest",
        help="do not write manifest.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = download_urfd(
        args.output,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
        create_manifest=args.create_manifest,
    )
    print(
        f"Downloaded {result['downloaded']}; skipped {result['skipped']}; "
        f"failed {len(result['failed'])}."
    )
    if result["failed"]:
        print("Failed files:", ", ".join(result["failed"]), file=sys.stderr)
        return 1
    if args.create_manifest:
        print(f"Manifest written to {Path(args.output).resolve() / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
