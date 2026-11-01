import argparse
import csv
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


METRICS = {
    "psnr_metric.csv": ("PSNR (dB)", "Higher: protected edit is closer to clean edit"),
    "ssim_metric.csv": ("SSIM", "Higher: protected edit is more structurally similar"),
    "lpips_metric.csv": ("LPIPS", "Lower: protected edit is more perceptually similar"),
    "clip_i_metric.csv": ("CLIP-I", "Higher: edited image retains more source-image semantics"),
    "clip_s_metric.csv": ("CLIP-S", "Higher: edit direction better matches the prompt"),
    "facial_metric.csv": ("Face similarity", "Lower: less identity similarity to the source"),
}

REPORTED_RESULTS = [
    ["No Defense", "0.118 ± 0.037", "N/A", "N/A", "N/A", "0.808 ± 0.074", "0.833 ± 0.111"],
    ["PhotoGuard Encoder attack", "0.108 ± 0.030", "15.44 ± 2.01", "0.612 ± 0.056", "0.403 ± 0.071", "0.670 ± 0.118", "0.590 ± 0.264"],
    ["EditShield", "0.110 ± 0.026", "17.74 ± 2.20", "0.593 ± 0.072", "0.382 ± 0.071", "0.677 ± 0.096", "0.641 ± 0.231"],
    ["Untargeted Encoder attack", "0.116 ± 0.023", "16.74 ± 2.27", "0.589 ± 0.084", "0.371 ± 0.094", "0.653 ± 0.090", "0.563 ± 0.236"],
    ["CW L2 attack", "0.115 ± 0.031", "19.64 ± 2.46", "0.701 ± 0.060", "0.247 ± 0.062", "0.733 ± 0.089", "0.725 ± 0.173"],
    ["VAE attack", "0.114 ± 0.034", "19.40 ± 1.70", "0.715 ± 0.039", "0.251 ± 0.060", "0.786 ± 0.061", "0.846 ± 0.097"],
    ["FACELOCK (ours)", "0.114 ± 0.024", "17.11 ± 2.36", "0.589 ± 0.079", "0.436 ± 0.065", "0.648 ± 0.089", "0.315 ± 0.109"],
]


def _method_label(value):
    value = value.strip()
    if not value or value.lower() == "clean":
        return "clean"
    path = Path(value.rstrip("/"))
    if path.name.startswith("budget_") and path.parent.name:
        return f"{path.parent.name}/{path.name}"
    return path.name or value


def _read_metrics(metrics_dir):
    results = {}
    found_files = []
    for filename, (label, _) in METRICS.items():
        csv_path = metrics_dir / filename
        if not csv_path.is_file():
            continue
        found_files.append(filename)
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                method = _method_label(row.get("method", ""))
                if not method:
                    continue
                raw_score = row.get("mean", "")
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = None
                results.setdefault(method, {})[label] = score
    return results, found_files


def _format_score(score):
    return "Not measured" if score is None else f"{score:.4f}"


def _add_report_page(pdf, metrics_dir, output_path, results, found_files):
    figure = plt.figure(figsize=(11.7, 8.3), facecolor="#f7f8f6")
    figure.text(0.07, 0.91, "MULTI-PERSON FACELOCK", fontsize=10, color="#177e73", weight="bold")
    figure.text(0.07, 0.85, "Evaluation results", fontsize=25, color="#172b2a", weight="bold")
    figure.text(0.07, 0.81, f"Generated {date.today().isoformat()}", fontsize=10, color="#556563")

    if found_files:
        status = f"Measured data loaded from {len(found_files)} metric CSV file(s)."
        detail = "Values below are arithmetic means exported by the existing evaluation scripts."
    else:
        status = "Quantitative evaluation is pending."
        detail = (
            "No metric CSV files were found. No numerical results are claimed. "
            "Run the evaluation scripts on matched clean and protected edit outputs, then regenerate this report."
        )
    figure.text(0.07, 0.745, status, fontsize=13, color="#172b2a", weight="bold")
    figure.text(0.07, 0.705, detail, fontsize=10, color="#556563", wrap=True)

    columns = ["Method"] + [details[0] for details in METRICS.values()]
    methods = sorted(results, key=lambda name: (name != "clean", name.lower()))
    if not methods:
        methods = ["No results available"]
    rows = []
    for method in methods:
        scores = results.get(method, {})
        rows.append([method] + [_format_score(scores.get(metric)) for metric in columns[1:]])

    axis = figure.add_axes([0.055, 0.38, 0.89, 0.27])
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d5ddda")
        if row == 0:
            cell.set_facecolor("#177e73")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#edf2ef")
            cell.get_text().set_color("#172b2a")

    figure.text(0.07, 0.32, "How to read these metrics", fontsize=12, color="#172b2a", weight="bold")
    metric_lines = [f"{name}: {meaning}." for name, meaning in METRICS.values()]
    figure.text(0.07, 0.285, "\n".join(metric_lines), fontsize=8.5, color="#344744", linespacing=1.6, va="top")

    caveat = (
        "Interpretation: PSNR, SSIM, and LPIPS compare protected-image edits against clean-image edits; "
        "CLIP-I, CLIP-S, and face similarity evaluate edited outputs. These are complementary measures, "
        "not a single privacy score. The current facial_metric.csv is image-level and is not a worst-face "
        "multi-person metric. Reported means are only as representative as the prompts, seeds, and images tested."
    )
    figure.text(0.07, 0.08, caveat, fontsize=8, color="#556563", wrap=True, va="bottom")
    figure.text(0.07, 0.035, f"Metrics directory: {metrics_dir}\nPDF: {output_path}", fontsize=7, color="#71807d")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _add_reported_comparison_page(pdf):
    figure = plt.figure(figsize=(11.7, 8.3), facecolor="#f7f8f6")
    figure.text(0.07, 0.91, "MULTI-PERSON FACELOCK", fontsize=10, color="#177e73", weight="bold")
    figure.text(0.07, 0.85, "Reported method comparison", fontsize=23, color="#172b2a", weight="bold")
    figure.text(
        0.07,
        0.80,
        "Mean ± standard deviation values supplied for this project",
        fontsize=10,
        color="#556563",
    )

    columns = [
        "Method",
        "CLIP-S (lower)",
        "PSNR (lower)",
        "SSIM (lower)",
        "LPIPS (higher)",
        "CLIP-I (lower)",
        "FR (lower)",
    ]
    axis = figure.add_axes([0.04, 0.40, 0.92, 0.34])
    axis.axis("off")
    table = axis.table(
        cellText=REPORTED_RESULTS,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.23, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.75)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d5ddda")
        if row == 0:
            cell.set_facecolor("#177e73")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#edf2ef")
            cell.get_text().set_color("#172b2a")
            if row == len(REPORTED_RESULTS):
                cell.get_text().set_weight("bold")

    figure.text(0.07, 0.31, "Provenance and interpretation", fontsize=12, color="#172b2a", weight="bold")
    note = (
        "These published single-face comparison values are transcribed from Table 1 of the supplied CVPR 2025 paper; "
        "they were not recalculated in this workspace. The first scored baseline is kept under the combined table label "
        "'PhotoGuard Encoder attack' as printed in the paper. These are not Multi-face results. "
        "Directions reproduce the requested arrows: lower for CLIP-S, PSNR, SSIM, CLIP-I, and FR; higher for LPIPS. "
        "CSV-derived local metrics are reported separately on the first page."
    )
    figure.text(0.07, 0.265, note, fontsize=9, color="#344744", wrap=True, va="top")
    figure.text(
        0.07,
        0.13,
        "The arrow directions describe the requested comparison table; metric desirability can depend on whether the goal is edit fidelity, prompt adherence, or identity obfuscation. Read them alongside the metric definitions and evaluation protocol.",
        fontsize=8.5,
        color="#556563",
        wrap=True,
        va="top",
    )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _add_multiface_method_page(pdf, found_files):
    figure = plt.figure(figsize=(11.7, 8.3), facecolor="#f7f8f6")
    figure.text(0.07, 0.91, "MULTI-PERSON FACELOCK", fontsize=10, color="#177e73", weight="bold")
    figure.text(0.07, 0.85, "Optimization and run settings", fontsize=23, color="#172b2a", weight="bold")
    figure.text(0.07, 0.80, "Implementation summary from methods.py", fontsize=10, color="#556563")

    settings = [
        ["Perturbation budget (epsilon)", "0.03", "L-infinity projection bound"],
        ["Initial step size", "0.01", "Sign-gradient step; decays toward 1%"],
        ["Joint iterations", "100", "Full-image worst-face refinement"],
        ["Warm start per face", "25 by default", "Configurable FaceLock crop optimization steps"],
    ]
    axis = figure.add_axes([0.07, 0.57, 0.86, 0.17])
    axis.axis("off")
    table = axis.table(
        cellText=settings,
        colLabels=["Parameter", "Default", "Role"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.31, 0.13, 0.56],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d5ddda")
        if row == 0:
            cell.set_facecolor("#177e73")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#edf2ef")
            cell.get_text().set_color("#172b2a")

    figure.text(0.07, 0.51, "Joint loss", fontsize=12, color="#172b2a", weight="bold")
    figure.text(
        0.07,
        0.47,
        "L = max_i(s_i) + 0.2 L_latent + L_LPIPS + 0.02 L_TV + 0.1 L_boundary",
        fontsize=11,
        color="#177e73",
        family="monospace",
    )
    figure.text(
        0.07,
        0.42,
        "The max identity term targets the least-protected face. Latent MSE, LPIPS, total variation, and mask-edge penalties preserve image quality and smoothness.",
        fontsize=9,
        color="#344744",
        wrap=True,
    )

    figure.text(0.07, 0.35, "Function flow", fontsize=12, color="#172b2a", weight="bold")
    flow = (
        "detect_face_boxes -> crop each detected face -> facelock warm start per crop -> soft-mask blend -> "
        "VAE encode/decode -> compute_score for each face -> torch.autograd.grad -> sign step, epsilon clip, and face-mask projection."
    )
    figure.text(0.07, 0.31, flow, fontsize=9, color="#344744", wrap=True, va="top")

    if found_files:
        result_note = "Metric CSVs were present for this report; see the evaluation-results page for their means."
    else:
        result_note = (
            "No metric CSVs were present when this report was generated. These settings are configuration, not measured results; "
            "Multi-face outcome scores remain pending. The current facial evaluator does not report worst-face similarity."
        )
    figure.text(0.07, 0.18, "Measurement status", fontsize=12, color="#172b2a", weight="bold")
    figure.text(0.07, 0.14, result_note, fontsize=9, color="#556563", wrap=True, va="top")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _add_extended_metrics_page(pdf, metrics_dir):
    files = ["per_face_privacy.csv", "robustness.csv", "editing_robustness.csv", "reidentification.csv", "efficiency.csv"]
    summaries = []
    for filename in files:
        path = metrics_dir / filename
        if not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        numeric = {}
        for key in rows[0]:
            values = []
            for row in rows:
                try:
                    value = float(row[key])
                    if value == value and abs(value) != float("inf"):
                        values.append(value)
                except (TypeError, ValueError):
                    pass
            if values:
                numeric[key] = sum(values) / len(values)
        summaries.append((filename, len(rows), numeric))

    figure = plt.figure(figsize=(11.7, 8.3), facecolor="#f7f8f6")
    figure.text(0.07, 0.91, "MULTI-PERSON FACELOCK", fontsize=10, color="#177e73", weight="bold")
    figure.text(0.07, 0.85, "Per-face, robustness, and efficiency", fontsize=22, color="#172b2a", weight="bold")
    figure.text(0.07, 0.80, f"Extended evaluation CSVs in {metrics_dir}", fontsize=9, color="#556563")
    if not summaries:
        figure.text(0.07, 0.70, "No extended evaluation CSVs found. Run evaluation/eval_multiface.py first.", fontsize=12, color="#344744")
    else:
        y = 0.73
        for filename, count, numeric in summaries:
            figure.text(0.07, y, f"{filename} ({count} rows)", fontsize=12, color="#177e73", weight="bold")
            y -= 0.035
            lines = [f"{key}: {value:.4f}" for key, value in numeric.items()]
            figure.text(0.09, y, "   ".join(lines) if lines else "No numeric columns", fontsize=9, color="#344744", wrap=True)
            y -= 0.085
            if y < 0.13:
                break
        figure.text(0.07, 0.08,
                    "CSV means are unweighted across rows. Use per-face rows for identity-level analysis. Verification attack success depends on the calibrated threshold; re-identification is only valid when an identity manifest was supplied.",
                    fontsize=8, color="#556563", wrap=True, va="bottom")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Combine FaceLock metric CSVs into a PDF evaluation report.")
    parser.add_argument(
        "--metrics_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing psnr_metric.csv, ssim_metric.csv, and other metric CSVs",
    )
    parser.add_argument(
        "--output_pdf",
        type=Path,
        default=Path(__file__).resolve().parent / "multiface_evaluation_report.pdf",
        help="where to write the PDF report",
    )
    args = parser.parse_args()

    metrics_dir = args.metrics_dir.expanduser().resolve()
    output_path = args.output_pdf.expanduser().resolve()
    results, found_files = _read_metrics(metrics_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        _add_report_page(pdf, metrics_dir, output_path, results, found_files)
        _add_reported_comparison_page(pdf)
        _add_multiface_method_page(pdf, found_files)
        _add_extended_metrics_page(pdf, metrics_dir)

    print(f"PDF report written to: {output_path}")
    if found_files:
        print("Metric CSVs included: " + ", ".join(found_files))
    else:
        print("No metric CSVs found; report marks quantitative evaluation as pending.")


if __name__ == "__main__":
    main()
