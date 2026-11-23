import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_cli_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "tools" / "aggregate_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("aggregate_evaluation_cli", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fall_report():
    empty_fall_metrics = {
        "evaluated_count": 0,
        "tp": 0,
        "late_detected": 0,
        "missing_ground_truth": 0,
        "fn": 0,
        "fp": 0,
        "tn": 0,
        "recall": None,
        "precision": None,
        "missed_alarm_rate": None,
        "fp_rate": None,
        "average_alarm_latency_seconds": None,
    }

    def metrics(**overrides):
        return {**empty_fall_metrics, **overrides}

    return {
        "dataset": "URFD",
        "camera": "cam0",
        "evaluated_count": 70,
        "read_errors": [],
        "overall_metrics": metrics(
            evaluated_count=70,
            tp=28,
            late_detected=1,
            missing_ground_truth=0,
            fn=2,
            fp=3,
            tn=37,
            recall=28 / 30,
            precision=28 / 31,
            missed_alarm_rate=2 / 30,
            fp_rate=3 / 40,
            average_alarm_latency_seconds=0.9,
        ),
        "group_metrics": {
            "fall": metrics(
                evaluated_count=30,
                tp=28,
                late_detected=1,
                missing_ground_truth=0,
                fn=2,
                recall=28 / 30,
                precision=1.0,
                missed_alarm_rate=2 / 30,
                average_alarm_latency_seconds=0.9,
            ),
            "all_adl": metrics(
                evaluated_count=40,
                fp=3,
                tn=37,
                precision=0.0,
                fp_rate=0.075,
            ),
            "basic_adl": metrics(
                evaluated_count=30,
                fp=1,
                tn=29,
                precision=0.0,
                fp_rate=1 / 30,
            ),
            "low_posture_adl": metrics(
                evaluated_count=10,
                fp=2,
                tn=8,
                precision=0.0,
                fp_rate=0.2,
            ),
        },
        "results": [],
    }


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
            "onnx": {
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
    assert "URFD 样本总数：70" in markdown
    assert "跌倒样本：30" in markdown
    assert "全量 ADL 样本：40" in markdown
    assert "基础 ADL 样本：30" in markdown
    assert "低姿态 ADL 样本：10" in markdown
    assert "跌倒检测汇总：28/30" in markdown
    assert "全量 ADL 误报率：7.50%" in markdown
    assert "平均及时报警延迟：0.900 s" in markdown
    assert "| vit | 72.00% | 0.680 | 8.4 | 680 |" in markdown
    assert "| onnx | 64.00% | 0.590 | 3.2 | 680 |" in markdown
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
