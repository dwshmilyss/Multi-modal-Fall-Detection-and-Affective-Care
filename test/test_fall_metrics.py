import json
from dataclasses import FrozenInstanceError, fields

import pytest

from deidao.fall_metrics import (
    FallEvaluationReport,
    URFDAnnotation,
    build_urfd_manifest,
    load_manifest,
    parse_urfd_annotation_csv,
    render_markdown_report,
    summarize_results,
    VideoEvaluationResult,
)


def write_csv(path, rows):
    path.write_text(
        "\n".join(f"{sequence},{frame},{label},1,1,1,1,1,1,1,1" for sequence, frame, label in rows),
        encoding="utf-8",
    )


def create_video(dataset_dir, relative_path):
    path = dataset_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")
    return path


def test_parse_urfd_annotation_csv_groups_frames_by_sequence(tmp_path):
    falls_csv = tmp_path / "urfall-cam0-falls.csv"
    write_csv(
        falls_csv,
        [
            ("fall-01", 9, -1),
            ("fall-01", 10, 0),
            ("fall-01", 19, 0),
            ("fall-01", 20, 1),
            ("fall-02", 12, -1),
            ("fall-02", 15, 0),
            ("fall-02", 24, 1),
        ],
    )
    # The official files contain extra posture columns; malformed lines must
    # not prevent the valid rows in the same file from being parsed.
    falls_csv.write_text(
        falls_csv.read_text(encoding="utf-8") + "\nnot-a-valid-row\n",
        encoding="utf-8",
    )

    annotations = parse_urfd_annotation_csv(falls_csv)

    assert annotations == {
        "fall-01": URFDAnnotation(
            sequence="fall-01",
            first_falling_frame=10,
            first_lying_frame=20,
            has_low_posture=True,
        ),
        "fall-02": URFDAnnotation(
            sequence="fall-02",
            first_falling_frame=15,
            first_lying_frame=24,
            has_low_posture=True,
        ),
    }


def test_adl_annotation_with_low_posture_stays_negative_but_is_grouped(tmp_path):
    falls_csv = tmp_path / "urfall-cam0-falls.csv"
    adls_csv = tmp_path / "urfall-cam0-adls.csv"
    write_csv(falls_csv, [("fall-01", 10, 0), ("fall-01", 20, 1)])
    write_csv(
        adls_csv,
        [
            ("adl-01", 8, -1),
            ("adl-01", 18, 0),
            ("adl-01", 30, 1),
            ("adl-02", 5, -1),
        ],
    )
    create_video(tmp_path, "fall/fall-01-cam0.mp4")
    create_video(tmp_path, "adl/adl-01-cam0.mp4")
    create_video(tmp_path, "adl/adl-02-cam0.mp4")

    annotations = parse_urfd_annotation_csv(adls_csv)

    assert annotations["adl-01"].has_low_posture is True
    assert annotations["adl-02"].has_low_posture is False

    manifest = build_urfd_manifest(tmp_path, falls_csv, adls_csv, fps=30.0)

    assert manifest["entries"][0] == {
        "path": "fall/fall-01-cam0.mp4",
        "label": "fall",
        "should_alarm": True,
        "action_start_seconds": 10 / 30.0,
        "lying_start_seconds": 20 / 30.0,
        "max_alarm_delay_seconds": 2.0,
        "activity_group": "fall",
    }
    assert manifest["entries"][1] == {
        "path": "adl/adl-01-cam0.mp4",
        "label": "adl",
        "should_alarm": False,
        "action_start_seconds": 18 / 30.0,
        "lying_start_seconds": 30 / 30.0,
        "max_alarm_delay_seconds": 2.0,
        "activity_group": "low_posture_adl",
    }
    assert manifest["entries"][2]["activity_group"] == "basic_adl"


def test_manifest_includes_official_urfd_metadata(tmp_path):
    falls_csv = tmp_path / "urfall-cam0-falls.csv"
    adls_csv = tmp_path / "urfall-cam0-adls.csv"
    write_csv(falls_csv, [("fall-01", 10, 0), ("fall-01", 20, 1)])
    write_csv(adls_csv, [("adl-01", 5, -1)])
    create_video(tmp_path, "fall/fall-01-cam0.mp4")
    create_video(tmp_path, "adl/adl-01-cam0.mp4")

    manifest = build_urfd_manifest(tmp_path, falls_csv, adls_csv, fps=30.0)

    assert manifest == {
        "dataset": "URFD",
        "camera": "cam0",
        "fps": 30.0,
        "source_url": "https://fenix.ur.edu.pl/~mkepski/ds/uf.html",
        "entries": [
            {
                "path": "fall/fall-01-cam0.mp4",
                "label": "fall",
                "should_alarm": True,
                "action_start_seconds": 10 / 30.0,
                "lying_start_seconds": 20 / 30.0,
                "max_alarm_delay_seconds": 2.0,
                "activity_group": "fall",
            },
            {
                "path": "adl/adl-01-cam0.mp4",
                "label": "adl",
                "should_alarm": False,
                "action_start_seconds": None,
                "lying_start_seconds": None,
                "max_alarm_delay_seconds": 2.0,
                "activity_group": "basic_adl",
            },
        ],
    }


def test_load_manifest_rejects_missing_video(tmp_path):
    falls_csv = tmp_path / "urfall-cam0-falls.csv"
    adls_csv = tmp_path / "urfall-cam0-adls.csv"
    write_csv(falls_csv, [("fall-01", 10, 0), ("fall-01", 20, 1)])
    write_csv(adls_csv, [])
    create_video(tmp_path, "fall/fall-01-cam0.mp4")

    manifest = build_urfd_manifest(tmp_path, falls_csv, adls_csv, fps=30.0)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "fall" / "fall-01-cam0.mp4").unlink()

    with pytest.raises(FileNotFoundError):
        load_manifest(manifest_path)


def make_result(
    path,
    label,
    should_alarm,
    alarm_times,
    action_start_seconds,
    activity_group,
    **overrides,
):
    values = {
        "path": path,
        "label": label,
        "should_alarm": should_alarm,
        "alarm_times": tuple(alarm_times),
        "action_start_seconds": action_start_seconds,
        "max_alarm_delay_seconds": 2.0,
        "activity_group": activity_group,
        "frame_count": 300,
        "fps": 30.0,
        "detected_person_frame_count": 280,
        "processing_fps": 48.0,
        "features": ("rapid_hip_descent", "rapid_torso_rotation"),
        "read_error": None,
    }
    values.update(overrides)
    return VideoEvaluationResult(**values)


def test_result_and_report_models_are_frozen_with_declared_fields():
    assert tuple(field.name for field in fields(VideoEvaluationResult)) == (
        "path",
        "label",
        "should_alarm",
        "alarm_times",
        "action_start_seconds",
        "max_alarm_delay_seconds",
        "activity_group",
        "frame_count",
        "fps",
        "detected_person_frame_count",
        "processing_fps",
        "features",
        "read_error",
    )
    assert tuple(field.name for field in fields(FallEvaluationReport)) == (
        "dataset",
        "camera",
        "evaluated_count",
        "read_errors",
        "overall_metrics",
        "group_metrics",
    )

    result = make_result("fall/fall-01-cam0.mp4", "fall", True, [10.8], 10.0, "fall")
    with pytest.raises(FrozenInstanceError):
        result.path = "changed.mp4"


def test_summarize_results_classifies_timely_late_missed_and_negative_videos():
    results = [
        make_result("fall/fall-01-cam0.mp4", "fall", True, [10.8], 10.0, "fall"),
        make_result("fall/fall-02-cam0.mp4", "fall", True, [14.0], 10.0, "fall"),
        make_result("fall/fall-03-cam0.mp4", "fall", True, [], 10.0, "fall"),
        make_result("adl/adl-01-cam0.mp4", "adl", False, [5.0], None, "basic_adl"),
        make_result("adl/adl-02-cam0.mp4", "adl", False, [], None, "basic_adl"),
        make_result("adl/adl-03-cam0.mp4", "adl", False, [], None, "low_posture_adl"),
        make_result("broken.mp4", "fall", True, [], None, "fall", read_error="file opened as video failed"),
    ]

    report = summarize_results(results)

    assert isinstance(report, FallEvaluationReport)
    assert report.dataset == "URFD"
    assert report.camera == "cam0"
    assert report.evaluated_count == 6
    assert report.read_errors == ("broken.mp4: file opened as video failed",)

    assert report.overall_metrics["tp"] == 1
    assert report.overall_metrics["late_detected"] == 1
    assert report.overall_metrics["fn"] == 2
    assert report.overall_metrics["fp"] == 1
    assert report.overall_metrics["tn"] == 2
    assert report.overall_metrics["recall"] == pytest.approx(1 / 3)
    assert report.overall_metrics["precision"] == pytest.approx(0.5)
    assert report.overall_metrics["average_alarm_latency_seconds"] == pytest.approx(0.8)
    assert report.overall_metrics["missed_alarm_rate"] == pytest.approx(2 / 3)
    assert report.overall_metrics["fp_rate"] == pytest.approx(1 / 3)

    assert report.group_metrics["fall"]["evaluated_count"] == 3
    assert report.group_metrics["fall"]["recall"] == pytest.approx(1 / 3)
    assert report.group_metrics["fall"]["average_alarm_latency_seconds"] == pytest.approx(0.8)
    assert report.group_metrics["all_adl"]["fp"] == 1
    assert report.group_metrics["all_adl"]["tn"] == 2
    assert report.group_metrics["all_adl"]["fp_rate"] == pytest.approx(1 / 3)
    assert report.group_metrics["basic_adl"]["fp_rate"] == pytest.approx(0.5)
    assert report.group_metrics["low_posture_adl"]["fp_rate"] == pytest.approx(0.0)
    assert report.group_metrics["basic_adl"]["recall"] is None
    assert report.group_metrics["low_posture_adl"]["precision"] is None


def test_render_markdown_report_explains_counts_and_metric_limits():
    results = [
        make_result("fall/fall-01-cam0.mp4", "fall", True, [10.8], 10.0, "fall"),
        make_result("fall/fall-02-cam0.mp4", "fall", True, [14.0], 10.0, "fall"),
        make_result("fall/fall-03-cam0.mp4", "fall", True, [], 10.0, "fall"),
        make_result("adl/adl-01-cam0.mp4", "adl", False, [5.0], None, "basic_adl"),
        make_result("adl/adl-02-cam0.mp4", "adl", False, [], None, "basic_adl"),
        make_result("adl/adl-03-cam0.mp4", "adl", False, [], None, "low_posture_adl"),
    ]
    report = summarize_results(results)

    markdown = render_markdown_report(report)

    assert "# URFD 跌倒检测评测报告" in markdown
    assert "UR Fall Detection Dataset" in markdown
    assert "跌倒样本：3" in markdown
    assert "TP：1" in markdown
    assert "late_detected：1" in markdown
    assert "FN：2" in markdown
    assert "全量 ADL" in markdown
    assert "基础 ADL" in markdown
    assert "低姿态 ADL" in markdown
    assert "late_detected 不计入召回率" in markdown
