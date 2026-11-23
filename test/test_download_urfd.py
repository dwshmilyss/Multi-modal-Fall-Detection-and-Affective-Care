import json
import os
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from urllib.error import URLError

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deidao.fall_metrics import build_urfd_manifest
from tools.download_urfd import (
    URFD_BASE_URL,
    DownloadItem,
    build_download_plan,
    download_urfd,
    parse_args,
    urfd_file_url,
)


EXPECTED_CSVS = {
    "urfall-cam0-falls.csv",
    "urfall-cam0-adls.csv",
}
EXPECTED_VIDEOS = {
    *(f"fall-{number:02d}-cam0.mp4" for number in range(1, 31)),
    *(f"adl-{number:02d}-cam0.mp4" for number in range(1, 41)),
}


def make_fake_downloader(monkeypatch, failures=None, errors=None):
    failures = dict(failures or {})
    errors = dict(errors or {})
    calls = []

    def fake_download_file(url, part_path):
        remote_name = url.removeprefix(URFD_BASE_URL)
        calls.append(remote_name)
        part_path.parent.mkdir(parents=True, exist_ok=True)
        remaining_failures = failures.get(remote_name, 0)
        if remaining_failures:
            failures[remote_name] = remaining_failures - 1
            part_path.write_bytes(b"incomplete")
            error = errors.get(remote_name, URLError("temporary network failure"))
            raise error
        part_path.write_bytes(remote_name.encode("utf-8"))

    monkeypatch.setattr("tools.download_urfd._download_file", fake_download_file)
    return calls


def test_urfd_file_url_uses_official_data_base():
    assert urfd_file_url("fall-01-cam0.mp4") == (
        "https://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam0.mp4"
    )


def test_build_download_plan_contains_official_videos_and_annotations(tmp_path):
    plan = build_download_plan(tmp_path)
    items = {item.remote_name: item for item in plan}

    assert len(plan) == 72
    assert set(items) == EXPECTED_CSVS | EXPECTED_VIDEOS
    assert len(EXPECTED_VIDEOS) == 70
    assert all(items[remote_name].local_path.is_absolute() for remote_name in items)
    assert all(
        items[remote_name].local_path == tmp_path / remote_name
        for remote_name in EXPECTED_CSVS
    )
    assert all(
        items[remote_name].local_path == tmp_path / "fall" / remote_name
        for remote_name in EXPECTED_VIDEOS
        if remote_name.startswith("fall-")
    )
    assert all(
        items[remote_name].local_path == tmp_path / "adl" / remote_name
        for remote_name in EXPECTED_VIDEOS
        if remote_name.startswith("adl-")
    )

    item = items["fall-01-cam0.mp4"]
    assert tuple(field.name for field in fields(DownloadItem)) == (
        "remote_name",
        "local_path",
    )
    with pytest.raises(FrozenInstanceError):
        item.remote_name = "changed"


def test_download_skips_nonempty_files_and_replaces_empty_files(tmp_path, monkeypatch):
    calls = make_fake_downloader(monkeypatch)
    skipped = tmp_path / "fall" / "fall-01-cam0.mp4"
    skipped.parent.mkdir(parents=True)
    skipped.write_bytes(b"already downloaded")
    empty = tmp_path / "adl" / "adl-01-cam0.mp4"
    empty.parent.mkdir(parents=True)
    empty.write_bytes(b"")

    result = download_urfd(
        tmp_path,
        retries=3,
        retry_delay_seconds=0.0,
        create_manifest=False,
    )

    assert result == {
        "downloaded": 71,
        "skipped": 1,
        "failed": [],
        "manifest": None,
    }
    assert len(calls) == 71
    assert calls.count("fall-01-cam0.mp4") == 0
    assert calls.count("adl-01-cam0.mp4") == 1
    assert empty.read_bytes() == b"adl-01-cam0.mp4"
    assert not list(tmp_path.rglob("*.part"))


def test_download_retries_transient_errors(tmp_path, monkeypatch):
    calls = make_fake_downloader(
        monkeypatch,
        failures={"fall-02-cam0.mp4": 1, "adl-02-cam0.mp4": 1},
        errors={"adl-02-cam0.mp4": OSError("temporary I/O failure")},
    )
    sleeps = []
    monkeypatch.setattr(
        "tools.download_urfd.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    result = download_urfd(tmp_path, retries=3, retry_delay_seconds=1.5)

    assert result["failed"] == []
    assert result["downloaded"] == 72
    assert calls.count("fall-02-cam0.mp4") == 2
    assert calls.count("adl-02-cam0.mp4") == 2
    assert sleeps == [1.5, 1.5]
    assert not list(tmp_path.rglob("*.part"))


def test_download_records_final_failure_and_continues(tmp_path, monkeypatch):
    calls = make_fake_downloader(monkeypatch)
    sleeps = []
    monkeypatch.setattr("tools.download_urfd.time.sleep", lambda seconds: sleeps.append(seconds))

    def fail_all_attempts(url, part_path):
        remote_name = url.removeprefix(URFD_BASE_URL)
        calls.append(remote_name)
        if remote_name == "adl-03-cam0.mp4":
            part_path.parent.mkdir(parents=True, exist_ok=True)
            part_path.write_bytes(b"incomplete")
            raise URLError("offline")
        part_path.write_bytes(remote_name.encode("utf-8"))

    monkeypatch.setattr("tools.download_urfd._download_file", fail_all_attempts)

    result = download_urfd(tmp_path, retries=3, retry_delay_seconds=0.0)

    assert result["downloaded"] == 71
    assert result["skipped"] == 0
    assert result["failed"] == ["adl-03-cam0.mp4"]
    assert result["manifest"] is None
    assert calls.count("adl-03-cam0.mp4") == 3
    assert len(calls) == 74
    assert sleeps == [0.0, 0.0]
    failed_part = tmp_path / "adl" / "adl-03-cam0.mp4.part"
    assert not failed_part.exists()
    assert not list(tmp_path.rglob("*.part"))
    assert (tmp_path / "fall" / "fall-01-cam0.mp4").read_bytes().startswith(b"fall-")


def test_download_writes_manifest_after_all_items_succeed(tmp_path, monkeypatch):
    make_fake_downloader(monkeypatch)

    result = download_urfd(tmp_path, retries=3, retry_delay_seconds=0.0)

    manifest_path = tmp_path / "manifest.json"
    expected_manifest = build_urfd_manifest(
        tmp_path,
        tmp_path / "urfall-cam0-falls.csv",
        tmp_path / "urfall-cam0-adls.csv",
        fps=30.0,
    )
    assert result == {
        "downloaded": 72,
        "skipped": 0,
        "failed": [],
        "manifest": expected_manifest,
    }
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == expected_manifest
    assert len(expected_manifest["entries"]) == 70


def test_cli_defaults_and_options_are_official_dataset_settings():
    defaults = parse_args([])
    enabled = parse_args(
        [
            "--output",
            "custom/urfd",
            "--retries",
            "5",
            "--retry-delay-seconds",
            "0.25",
        ]
    )
    disabled = parse_args(["--no-manifest"])

    assert defaults.output == "test_videos/evaluation/urfd"
    assert defaults.retries == 3
    assert defaults.retry_delay_seconds == 2.0
    assert defaults.create_manifest is True
    assert enabled.output == "custom/urfd"
    assert enabled.retries == 5
    assert enabled.retry_delay_seconds == 0.25
    assert enabled.create_manifest is True
    assert disabled.create_manifest is False
