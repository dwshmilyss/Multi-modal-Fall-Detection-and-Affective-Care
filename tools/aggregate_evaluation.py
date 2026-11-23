"""Aggregate URFD fall and emotion evaluation reports into Markdown evidence."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


URFD_SOURCE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"
URFD_LICENSE = (
    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International"
)
URFD_CITATION = (
    "Bogdan Kwolek and Michal Kepski, "
    "“Human fall detection on embedded platform using depth maps and wireless "
    "accelerometer,” Computer Methods and Programs in Biomedicine, "
    "Volume 117, Issue 3, December 2014, pages 489-501."
)


def load_json(path: str | Path) -> dict:
    """Read a UTF-8 JSON object report."""

    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    if not isinstance(report, dict):
        raise ValueError(f"JSON report must contain an object: {report_path}")
    return report


def _format_metric(value, *, percent: bool = False, seconds: bool = False) -> str:
    if value is None:
        return "--"
    if percent:
        return f"{value:.2%}"
    if seconds:
        return f"{value:.3f} s"
    return str(value)


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".part",
        dir=path.parent,
    )
    os.close(descriptor)
    staging_path = Path(staging_name)
    try:
        staging_path.write_text(text, encoding="utf-8")
        staging_path.replace(path)
    finally:
        staging_path.unlink(missing_ok=True)


def _fall_metrics_table(report: dict) -> list[str]:
    lines = [
        "| 分组 | 有效评测数 | missing_ground_truth | TP | late_detected | FN | FP | TN | "
        "召回率 | 精确率 | 漏报率 | 误报率 | 平均及时延迟 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for group_name in ("fall", "all_adl", "basic_adl", "low_posture_adl"):
        metrics = report["group_metrics"][group_name]
        lines.append(
            f"| {group_name} | {metrics['evaluated_count']} | "
            f"{metrics['missing_ground_truth']} | {metrics['tp']} | "
            f"{metrics['late_detected']} | {metrics['fn']} | {metrics['fp']} | "
            f"{metrics['tn']} | "
            f"{_format_metric(metrics.get('recall'), percent=True)} | "
            f"{_format_metric(metrics.get('precision'), percent=True)} | "
            f"{_format_metric(metrics.get('missed_alarm_rate'), percent=True)} | "
            f"{_format_metric(metrics.get('fp_rate'), percent=True)} | "
            f"{_format_metric(metrics.get('average_alarm_latency_seconds'), seconds=True)} |"
        )
    return lines


def _emotion_metrics_section(emotion_report: dict | None) -> list[str]:
    if emotion_report is None:
        return [
            "## 情绪模型证据",
            "",
            "情绪基准报告未提供；因此准确率、宏 F1、平均延迟和评测人脸数均明确省略，不做推测。",
        ]

    dataset = emotion_report["dataset"]
    lines = [
        "## 情绪模型证据",
        "",
        f"- 样本总数：{dataset['total_samples']}",
        f"- 已评测人脸帧：{dataset['evaluated_frames']}",
        f"- 未检测到人脸帧：{dataset['no_face_frames']}",
        "",
        "| Backend | Accuracy | Macro F1 | Avg latency (ms) | Evaluated |",
        "|---|---:|---:|---:|---:|",
    ]
    for backend_name, metrics in emotion_report["backends"].items():
        lines.append(
            f"| {backend_name} | {metrics['accuracy']:.2%} | "
            f"{metrics['macro_f1']:.3f} | "
            f"{metrics['average_latency_ms']:.1f} | "
            f"{metrics['evaluated']} |"
        )
    return lines


def render_aggregate_markdown(
    fall_report_path: str | Path,
    emotion_report_path: str | Path | None = None,
) -> str:
    """Render aggregate evidence from local evaluation-report JSON files."""

    fall_path = Path(fall_report_path)
    if not fall_path.is_file():
        raise FileNotFoundError(f"Fall report not found: {fall_path}")

    fall_report = load_json(fall_path)
    emotion_report = None
    if emotion_report_path is not None:
        emotion_path = Path(emotion_report_path)
        if not emotion_path.is_file():
            raise FileNotFoundError(f"Emotion report not found: {emotion_path}")
        emotion_report = load_json(emotion_path)

    overall = fall_report["overall_metrics"]
    groups = fall_report["group_metrics"]
    results = fall_report.get("results", [])
    selected_count = fall_report.get("selected_count", len(results))
    fall_metrics = groups["fall"]
    timely_fall_count = fall_metrics["tp"] + fall_metrics["fn"]
    adl_metrics = groups["all_adl"]
    lines = [
        "# 算法评测证据汇总",
        "",
        "## URFD 数据来源与样本构成",
        "",
        f"- 数据集：[UR Fall Detection Dataset]({URFD_SOURCE_URL})",
        f"- 相机：{fall_report['camera']}；原始 URFD 视频为 depth 左半 / RGB 右半，评测使用右半 RGB。",
        f"- URFD 选择/尝试评测视频：{selected_count}",
        f"- URFD 有效评测视频：{fall_report['evaluated_count']}",
        f"- fall 有效评测样本：{fall_metrics['evaluated_count']}",
        f"- all_adl 有效评测样本：{adl_metrics['evaluated_count']}",
        f"- basic_adl 有效评测样本：{groups['basic_adl']['evaluated_count']}",
        f"- low_posture_adl 有效评测样本：{groups['low_posture_adl']['evaluated_count']}",
        f"- 读取失败：{len(fall_report['read_errors'])}",
        "",
        "## 跌倒检测证据",
        "",
        f"跌倒检测汇总：{fall_metrics['tp']}/{timely_fall_count} 及时检测。",
        f"全量 ADL 误报率：{_format_metric(adl_metrics.get('fp_rate'), percent=True)}。",
        "平均及时报警延迟："
        f"{_format_metric(overall.get('average_alarm_latency_seconds'), seconds=True)}。",
        "",
        *_fall_metrics_table(fall_report),
        "",
        "指标说明：及时检测要求报警时间位于动作开始后允许延迟窗口内；late_detected 只作延迟诊断，不计入召回率。",
        *_emotion_metrics_section(emotion_report),
        "",
        "## 数据许可与引用",
        "",
        f"- URFD 数据源许可：{URFD_LICENSE}。",
        "- 引用：" + URFD_CITATION,
        "",
        "## 使用边界与产物管理",
        "",
        "- 本评测只用于算法研究，不构成医疗诊断或医疗建议，也不应作为唯一安全决策依据。",
        "- 所有数据集、原始媒体与本报告均只在本地生成，不提交到仓库。",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fall-report", required=True)
    parser.add_argument("--emotion-report")
    parser.add_argument(
        "--output",
        default="output/evaluation/algorithm_evidence.md",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    markdown = render_aggregate_markdown(args.fall_report, args.emotion_report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(output_path, markdown)
    print(f"Aggregate evidence written to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
