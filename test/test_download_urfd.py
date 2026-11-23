import json
import os
import ssl
import sys
import urllib.request
from dataclasses import FrozenInstanceError, fields
from http.client import HTTPException
from io import BytesIO
from pathlib import Path
from urllib.error import URLError

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deidao.fall_metrics import build_urfd_manifest
import tools.download_urfd as download_urfd_module
from tools.download_urfd import (
    URFD_BASE_URL,
    DownloadItem,
    _download_file,
    _write_manifest,
    build_download_plan,
    download_urfd,
    main,
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

    def fake_download_file(url, part_path, verify_ssl=True):
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


class FakeHTTPResponse:
    def __init__(self, body, content_length=None):
        self.status = 200
        self.headers = {} if content_length is None else {
            "Content-Length": content_length,
        }
        self.body = BytesIO(body)

    def read(self, size=-1):
        return self.body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.body.close()


def redirect_request(newurl):
    request = urllib.request.Request(urfd_file_url("fall-01-cam0.mp4"))
    handler = download_urfd_module.URFDRedirectHandler()
    return handler.redirect_request(
        request,
        None,
        301,
        "Moved Permanently",
        {},
        newurl,
    )


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


def test_download_retries_http_exception_and_continues(tmp_path, monkeypatch):
    calls = make_fake_downloader(
        monkeypatch,
        failures={"fall-01-cam0.mp4": 1},
        errors={"fall-01-cam0.mp4": HTTPException("IncompleteRead")},
    )
    sleeps = []
    monkeypatch.setattr(
        "tools.download_urfd.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    result = download_urfd(
        tmp_path,
        retries=3,
        retry_delay_seconds=0.0,
        create_manifest=False,
    )

    assert result["downloaded"] == 72
    assert result["failed"] == []
    assert calls.count("fall-01-cam0.mp4") == 2
    assert len(calls) == 73
    assert sleeps == [0.0]
    assert (
        tmp_path / "adl" / "adl-40-cam0.mp4"
    ).read_bytes() == b"adl-40-cam0.mp4"
    assert not list(tmp_path.rglob("*.part"))


def test_download_records_rename_failure_and_continues(tmp_path, monkeypatch):
    calls = make_fake_downloader(monkeypatch)
    original_replace = Path.replace

    def fail_one_rename(source, destination):
        if destination.name == "fall-01-cam0.mp4":
            raise PermissionError("destination is locked")
        return original_replace(source, destination)

    monkeypatch.setattr("tools.download_urfd.Path.replace", fail_one_rename)

    result = download_urfd(
        tmp_path,
        retries=1,
        retry_delay_seconds=0.0,
        create_manifest=False,
    )

    assert result["downloaded"] == 71
    assert result["failed"] == ["fall-01-cam0.mp4"]
    assert len(calls) == 72
    assert calls[-1] == "adl-40-cam0.mp4"
    assert (tmp_path / "fall" / "fall-02-cam0.mp4").is_file()
    assert not list(tmp_path.rglob("*.part"))


def test_download_uses_unique_staging_path_for_each_attempt(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "fall" / "fall-01-cam0.mp4"
    for item in build_download_plan(tmp_path):
        if item.remote_name == "fall-01-cam0.mp4":
            continue
        item.local_path.parent.mkdir(parents=True, exist_ok=True)
        item.local_path.write_bytes(b"already downloaded")

    staging_paths = []

    def fake_download_file(url, part_path, verify_ssl=True):
        remote_name = url.removeprefix(URFD_BASE_URL)
        assert remote_name == "fall-01-cam0.mp4"
        staging_paths.append(part_path)
        if len(staging_paths) == 1:
            part_path.write_bytes(b"incomplete")
            raise URLError("temporary failure")
        part_path.write_bytes(b"complete")

    monkeypatch.setattr("tools.download_urfd._download_file", fake_download_file)
    monkeypatch.setattr("tools.download_urfd.time.sleep", lambda seconds: None)

    result = download_urfd(
        tmp_path,
        retries=2,
        retry_delay_seconds=0.0,
        create_manifest=False,
    )

    assert result["downloaded"] == 1
    assert result["failed"] == []
    assert len(staging_paths) == 2
    assert len({str(path) for path in staging_paths}) == 2
    assert all(path.parent == target.parent for path in staging_paths)
    assert all(path.name.startswith(".fall-01-cam0.mp4.") for path in staging_paths)
    assert all(path.suffix == ".part" for path in staging_paths)
    assert target.read_bytes() == b"complete"
    assert not list(tmp_path.rglob("*.part"))


def test_download_records_final_failure_and_continues(tmp_path, monkeypatch):
    calls = make_fake_downloader(monkeypatch)
    sleeps = []
    monkeypatch.setattr("tools.download_urfd.time.sleep", lambda seconds: sleeps.append(seconds))

    def fail_all_attempts(url, part_path, verify_ssl=True):
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


def test_download_file_rejects_empty_and_length_mismatched_responses(
    tmp_path,
    monkeypatch,
):
    stage_path = tmp_path / ".response.part"

    def fake_open(self, request, timeout):
        if request.full_url == urfd_file_url("fall-01-cam0.mp4"):
            return FakeHTTPResponse(b"", content_length=None)
        return FakeHTTPResponse(b"abc", content_length="4")

    monkeypatch.setattr(
        "tools.download_urfd.urllib.request.OpenerDirector.open",
        fake_open,
    )

    with pytest.raises(URLError, match="empty download"):
        _download_file(urfd_file_url("fall-01-cam0.mp4"), stage_path)
    with pytest.raises(URLError, match="Content-Length mismatch"):
        _download_file(urfd_file_url("fall-02-cam0.mp4"), stage_path)


def test_download_file_uses_certificate_verification_by_default(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_open(self, request, timeout):
        calls.append((request, timeout))
        return FakeHTTPResponse(b"body", content_length="4")

    monkeypatch.setattr(
        "tools.download_urfd.urllib.request.OpenerDirector.open",
        fake_open,
    )

    _download_file(urfd_file_url("fall-01-cam0.mp4"), tmp_path / ".body.part")

    request, timeout = calls[0]
    assert request.full_url == urfd_file_url("fall-01-cam0.mp4")
    assert timeout == 60


def test_download_file_insecure_uses_explicit_unverified_tls_context(
    tmp_path,
    monkeypatch,
):
    contexts = []

    class FakeOpener:
        def open(self, request, timeout):
            return FakeHTTPResponse(b"body", content_length="4")

    def fake_build_opener(*handlers):
        contexts.extend(
            handler._context
            for handler in handlers
            if isinstance(handler, urllib.request.HTTPSHandler)
        )
        return FakeOpener()

    monkeypatch.setattr(
        "tools.download_urfd.urllib.request.build_opener",
        fake_build_opener,
    )

    _download_file(
        urfd_file_url("fall-01-cam0.mp4"),
        tmp_path / ".body.part",
        verify_ssl=False,
    )

    context = contexts[0]
    assert context is not None
    assert context.check_hostname is False
    assert context.verify_mode == ssl.CERT_NONE


def test_download_file_builds_opener_with_redirect_handler(tmp_path, monkeypatch):
    handler_types = []

    class FakeOpener:
        def open(self, request, timeout):
            return FakeHTTPResponse(b"body", content_length="4")

    def fake_build_opener(*handlers):
        handler_types.extend(type(handler) for handler in handlers)
        return FakeOpener()

    monkeypatch.setattr(
        "tools.download_urfd.urllib.request.build_opener",
        fake_build_opener,
    )
    monkeypatch.setattr(
        "tools.download_urfd.urllib.request.urlopen",
        lambda *args, **kwargs: pytest.fail("urlopen must not bypass the URFD opener"),
    )

    _download_file(
        urfd_file_url("fall-01-cam0.mp4"),
        tmp_path / ".redirect.part",
    )

    assert download_urfd_module.URFDRedirectHandler in handler_types


def test_redirect_handler_rejects_http_downgrade():
    with pytest.raises(URLError, match="non-HTTPS URFD redirect"):
        redirect_request("http://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam0.mp4")


def test_redirect_handler_rejects_cross_host_redirect():
    with pytest.raises(URLError, match="cross-host URFD redirect"):
        redirect_request("https://example.com/fall-01-cam0.mp4")


def test_redirect_handler_allows_same_host_https_redirect():
    redirected = redirect_request(
        "https://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam0.mp4?token=1"
    )

    assert isinstance(redirected, urllib.request.Request)
    assert redirected.full_url == (
        "https://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam0.mp4?token=1"
    )


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


def test_manifest_write_is_atomic_and_cleans_staging(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("existing manifest", encoding="utf-8")
    monkeypatch.setattr(
        "tools.download_urfd.build_urfd_manifest",
        lambda *args, **kwargs: {"entries": []},
    )

    def fail_after_partial_write(self, data, encoding=None, errors=None, newline=None):
        with self.open("w", encoding=encoding, errors=errors, newline=newline) as file:
            file.write(data[:20])
            raise OSError("disk full")

    monkeypatch.setattr("tools.download_urfd.Path.write_text", fail_after_partial_write)

    with pytest.raises(OSError, match="disk full"):
        _write_manifest(tmp_path)

    assert manifest_path.read_text(encoding="utf-8") == "existing manifest"
    assert not list(tmp_path.rglob("*.part"))


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
    assert defaults.verify_ssl is True
    assert enabled.output == "custom/urfd"
    assert enabled.retries == 5
    assert enabled.retry_delay_seconds == 0.25
    assert enabled.create_manifest is True
    assert disabled.create_manifest is False
    assert parse_args(["--insecure"]).verify_ssl is False


def test_download_urfd_propagates_ssl_verification_choice(tmp_path, monkeypatch):
    calls = []

    for item in build_download_plan(tmp_path):
        if item.remote_name == "fall-01-cam0.mp4":
            continue
        item.local_path.parent.mkdir(parents=True, exist_ok=True)
        item.local_path.write_bytes(b"already downloaded")

    def spy_download_file(url, part_path, verify_ssl=True):
        calls.append(verify_ssl)
        part_path.write_bytes(b"downloaded")

    monkeypatch.setattr("tools.download_urfd._download_file", spy_download_file)

    download_urfd(
        tmp_path,
        retries=1,
        retry_delay_seconds=0.0,
        create_manifest=False,
        verify_ssl=False,
    )

    assert calls == [False]


def test_download_urfd_warns_once_for_direct_insecure_api_use(capsys, tmp_path):
    for item in build_download_plan(tmp_path):
        item.local_path.parent.mkdir(parents=True, exist_ok=True)
        item.local_path.write_bytes(b"already downloaded")

    result = download_urfd(
        tmp_path,
        retries=1,
        retry_delay_seconds=0.0,
        create_manifest=False,
        verify_ssl=False,
    )

    captured = capsys.readouterr()
    assert result["failed"] == []
    assert captured.err.count(
        "Warning: certificate verification is disabled; "
        "the connection is vulnerable to tampering."
    ) == 1


def test_cli_warns_when_insecure_selected(tmp_path, capsys):
    for item in build_download_plan(tmp_path):
        item.local_path.parent.mkdir(parents=True, exist_ok=True)
        item.local_path.write_bytes(b"already downloaded")

    exit_code = main(
        ["--output", str(tmp_path), "--insecure", "--no-manifest"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err.count(
        "Warning: certificate verification is disabled; "
        "the connection is vulnerable to tampering."
    ) == 1


def test_cli_records_retried_redirect_rejection_without_traceback(
    tmp_path,
    monkeypatch,
    capsys,
):
    for item in build_download_plan(tmp_path):
        if item.remote_name == "fall-01-cam0.mp4":
            continue
        item.local_path.parent.mkdir(parents=True, exist_ok=True)
        item.local_path.write_bytes(b"already downloaded")

    def reject_redirect(url, part_path, verify_ssl=True):
        redirect_request("http://fenix.ur.edu.pl/fall-01-cam0.mp4")

    monkeypatch.setattr("tools.download_urfd._download_file", reject_redirect)
    monkeypatch.setattr(
        "tools.download_urfd.time.sleep",
        lambda seconds: None,
    )

    exit_code = main(
        [
            "--output",
            str(tmp_path),
            "--retries",
            "2",
            "--no-manifest",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Failed after 2 attempts: fall-01-cam0.mp4" in captured.err
    assert "non-HTTPS URFD redirect" in captured.err
    assert "Traceback" not in captured.err


def test_readme_documents_official_adl_group_counts():
    readme = Path(__file__).resolve().parent.parent / "README.md"
    content = readme.read_text(encoding="utf-8")

    assert "| basic_adl | 24 |" in content
    assert "| low_posture_adl | 16 |" in content


def test_cli_failure_returns_error_and_prints_summary(monkeypatch, capsys):
    def fail_download(
        output_dir,
        retries,
        retry_delay_seconds,
        create_manifest,
        verify_ssl,
    ):
        return {
            "downloaded": 2,
            "skipped": 3,
            "failed": ["fall-01-cam0.mp4"],
            "manifest": None,
        }

    monkeypatch.setattr("tools.download_urfd.download_urfd", fail_download)

    exit_code = main(["--output", "custom/urfd", "--no-manifest"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Downloaded 2; skipped 3; failed 1." in captured.out
    assert "Failed files: fall-01-cam0.mp4" in captured.err
