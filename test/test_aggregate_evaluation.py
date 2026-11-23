import importlib.util
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from deidao.fall_metrics import VideoEvaluationResult, summarize_results

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_cli_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "tools" / "aggregate_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("aggregate_evaluation_cli", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_result(
    path,
    *,
    should_alarm,
    activity_group,
    alarm_times=(),
    action_start_seconds=None,
    read_error=None,
):
    return VideoEvaluationResult(
        path=path,
        label="fall" if should_alarm else "adl",
        should_alarm=should_alarm,
        alarm_times=tuple(alarm_times),
        action_start_seconds=action_start_seconds,
        max_alarm_delay_seconds=2.0,
        activity_group=activity_group,
        frame_count=300,
        fps=30.0,
        detected_person_frame_count=280,
        processing_fps=48.0,
        features=("rapid_hip_descent",) if alarm_times else (),
        read_error=read_error,
    )


def build_fall_report():
    results = []
    for number in range(1, 31):
        path = f"fall/fall-{number:02d}-cam0.mp4"
        if number <= 28:
            alarm_times = (10.9,)
        elif number == 29:
            alarm_times = (15.0,)
        else:
            alarm_times = ()
        results.append(
            make_result(
                path,
                should_alarm=True,
                activity_group="fall",
                alarm_times=alarm_times,
                action_start_seconds=10.0,
            )
        )

    for number in range(1, 31):
        results.append(
            make_result(
                f"adl/adl-{number:02d}-cam0.mp4",
                should_alarm=False,
                activity_group="basic_adl",
                alarm_times=(5.0,) if number == 1 else (),
            )
        )
    for number in range(1, 11):
        results.append(
            make_result(
                f"adl/low-posture-adl-{number:02d}-cam0.mp4",
                should_alarm=False,
                activity_group="low_posture_adl",
                alarm_times=(5.0,) if number in (1, 2) else (),
            )
        )

    report = asdict(summarize_results(results))
    report["dataset"] = "URFD"
    report["camera"] = "cam0"
    report["results"] = [asdict(result) for result in results]
    return report


def make_fall_report():
    return build_fall_report()


def make_emotion_report():
    return {
        "dataset": {
            "total_samples": 700,
            "evaluated_frames": 680,
            "no_face_frames": 20,
        },
        "backends": {
            "vit": {
                "accuracy": 0.72,
                "macro_f1": 0.68,
                "average_latency_ms": 8.4,
                "evaluated": 680,
            },
            "hse_b0": {
                "accuracy": 0.64,
                "macro_f1": 0.59,
                "average_latency_ms": 3.2,
                "evaluated": 680,
            },
        },
    }


def write_reports(tmp_path):
    fall_report_path = tmp_path / "fall.json"
    emotion_report_path = tmp_path / "emotion.json"
    fall_report_path.write_text(
        json.dumps(make_fall_report(), ensure_ascii=False),
        encoding="utf-8",
    )
    emotion_report_path.write_text(
        json.dumps(make_emotion_report(), ensure_ascii=False),
        encoding="utf-8",
    )
    return fall_report_path, emotion_report_path


def test_load_json_reads_utf8_json(tmp_path):
    from tools.aggregate_evaluation import load_json

    path = tmp_path / "report.json"
    path.write_text('{"dataset": "URFD", "label": "中文"}', encoding="utf-8")

    assert load_json(path) == {"dataset": "URFD", "label": "中文"}


def test_render_aggregate_markdown_contains_required_evidence(tmp_path):
    from tools.aggregate_evaluation import render_aggregate_markdown

    fall_report_path, emotion_report_path = write_reports(tmp_path)

    markdown = render_aggregate_markdown(fall_report_path, emotion_report_path)

    assert "# 算法评测证据汇总" in markdown
    assert "https://fenix.ur.edu.pl/~mkepski/ds/uf.html" in markdown
    assert "URFD 选择/尝试评测视频：70" in markdown
    assert "URFD 有效评测视频：70" in markdown
    assert "fall 有效评测样本：30" in markdown
    assert "all_adl 有效评测样本：40" in markdown
    assert "basic_adl 有效评测样本：30" in markdown
    assert "low_posture_adl 有效评测样本：10" in markdown
    assert "| 分组 | 有效评测数 |" in markdown
    assert "跌倒检测汇总：28/30" in markdown
    assert "全量 ADL 误报率：7.50%" in markdown
    assert "平均及时报警延迟：0.900 s" in markdown
    assert "| vit | 72.00% | 0.680 | 8.4 | 680 |" in markdown
    assert "| hse_b0 | 64.00% | 0.590 | 3.2 | 680 |" in markdown
    assert "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International" in markdown
    assert "Bogdan Kwolek and Michal Kepski" in markdown
    assert "Human fall detection on embedded platform using depth maps and wireless accelerometer" in markdown
    assert "Computer Methods and Programs in Biomedicine" in markdown
    assert "Volume 117, Issue 3, December 2014, pages 489-501" in markdown
    assert "不构成医疗诊断或医疗建议" in markdown
    assert "本地生成，不提交到仓库" in markdown


def test_render_aggregate_markdown_reports_missing_emotion_without_inventing_metrics(
    tmp_path,
):
    from tools.aggregate_evaluation import render_aggregate_markdown

    fall_report_path, _ = write_reports(tmp_path)

    markdown = render_aggregate_markdown(fall_report_path, None)

    assert "情绪基准报告未提供" in markdown
    assert "72.00%" not in markdown
    assert "64.00%" not in markdown
    assert "8.4" not in markdown
    assert "3.2" not in markdown


def test_render_aggregate_markdown_distinguishes_selected_and_valid_counts(tmp_path):
    from tools.aggregate_evaluation import render_aggregate_markdown

    usable_result = make_result(
        "fall/fall-01-cam0.mp4",
        should_alarm=True,
        activity_group="fall",
        alarm_times=(11.0,),
        action_start_seconds=None,
    )
    failed_result = make_result(
        "fall/fall-02-cam0.mp4",
        should_alarm=True,
        activity_group="fall",
        read_error="decoder failed",
    )
    report = asdict(summarize_results([usable_result, failed_result]))
    report["dataset"] = "URFD"
    report["camera"] = "cam0"
    report["results"] = [asdict(usable_result), asdict(failed_result)]
    fall_report_path = tmp_path / "fall-with-read-error.json"
    fall_report_path.write_text(json.dumps(report), encoding="utf-8")

    markdown = render_aggregate_markdown(fall_report_path)

    assert "URFD 选择/尝试评测视频：2" in markdown
    assert "URFD 有效评测视频：1" in markdown
    assert "fall 有效评测样本：1" in markdown
    assert "| 分组 | 有效评测数 |" in markdown
    assert "| fall | 1 | 1 |" in markdown
    assert "读取失败：1" in markdown


def test_render_aggregate_markdown_prefers_explicit_selected_count(tmp_path):
    from tools.aggregate_evaluation import render_aggregate_markdown

    usable_result = make_result(
        "fall/fall-01-cam0.mp4",
        should_alarm=True,
        activity_group="fall",
        alarm_times=(11.0,),
        action_start_seconds=None,
    )
    failed_result = make_result(
        "fall/fall-02-cam0.mp4",
        should_alarm=True,
        activity_group="fall",
        read_error="decoder failed",
    )
    report = asdict(summarize_results([usable_result, failed_result]))
    report["dataset"] = "URFD"
    report["camera"] = "cam0"
    report["selected_count"] = 5
    report["results"] = [asdict(usable_result), asdict(failed_result)]
    fall_report_path = tmp_path / "fall-with-explicit-selected-count.json"
    fall_report_path.write_text(json.dumps(report), encoding="utf-8")

    markdown = render_aggregate_markdown(fall_report_path)

    assert "URFD 选择/尝试评测视频：5" in markdown
    assert "URFD 有效评测视频：1" in markdown
    assert "读取失败：1" in markdown


def test_render_aggregate_markdown_requires_a_supplied_emotion_report(tmp_path):
    from tools.aggregate_evaluation import render_aggregate_markdown

    fall_report_path, _ = write_reports(tmp_path)
    missing_emotion_path = tmp_path / "missing-emotion-report.json"

    with pytest.raises(
        FileNotFoundError,
        match=f"Emotion report not found: {re.escape(str(missing_emotion_path))}",
    ):
        render_aggregate_markdown(fall_report_path, missing_emotion_path)


def test_render_aggregate_markdown_requires_a_fall_report(tmp_path):
    from tools.aggregate_evaluation import render_aggregate_markdown

    missing_path = tmp_path / "missing-fall-report.json"

    with pytest.raises(
        FileNotFoundError,
        match=f"Fall report not found: {re.escape(str(missing_path))}",
    ):
        render_aggregate_markdown(missing_path)


def test_cli_parse_args_exposes_required_and_default_paths():
    aggregator = load_cli_module()

    args = aggregator.parse_args(
        [
            "--fall-report",
            "fall.json",
            "--emotion-report",
            "emotion.json",
            "--output",
            "custom/evidence.md",
        ]
    )

    assert args.fall_report == "fall.json"
    assert args.emotion_report == "emotion.json"
    assert args.output == "custom/evidence.md"


def test_cli_defaults_to_the_repository_evidence_report():
    aggregator = load_cli_module()

    args = aggregator.parse_args(["--fall-report", "fall.json"])

    assert args.emotion_report is None
    assert args.output == "output/evaluation/algorithm_evidence.md"


def test_cli_writes_aggregate_report(tmp_path):
    aggregator = load_cli_module()
    fall_report_path, emotion_report_path = write_reports(tmp_path)
    output_path = tmp_path / "aggregate" / "algorithm_evidence.md"

    exit_code = aggregator.main(
        [
            "--fall-report",
            str(fall_report_path),
            "--emotion-report",
            str(emotion_report_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    markdown = output_path.read_text(encoding="utf-8")
    assert markdown.startswith("# 算法评测证据汇总")
    assert "跌倒检测汇总：28/30" in markdown
    assert list(output_path.parent.iterdir()) == [output_path]


def test_cli_atomic_write_preserves_existing_output_and_cleans_staging(
    tmp_path,
    monkeypatch,
):
    aggregator = load_cli_module()
    fall_report_path, _ = write_reports(tmp_path)
    output_path = tmp_path / "aggregate" / "algorithm_evidence.md"
    output_path.parent.mkdir()
    output_path.write_text("existing aggregate evidence", encoding="utf-8")
    original_write_text = Path.write_text

    def fail_after_partial_write(self, data, encoding=None, errors=None, newline=None):
        if self == output_path or (
            self.parent == output_path.parent and self.suffix == ".part"
        ):
            original_write_text(self, data[:20], encoding=encoding)
            raise OSError("disk full")
        return original_write_text(self, data, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", fail_after_partial_write)

    with pytest.raises(OSError, match="disk full"):
        aggregator.main(
            [
                "--fall-report",
                str(fall_report_path),
                "--output",
                str(output_path),
            ]
        )

    assert output_path.read_text(encoding="utf-8") == "existing aggregate evidence"
    assert not list(output_path.parent.glob("*.part"))
