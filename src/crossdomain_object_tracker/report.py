"""Report generation for cross-domain evaluation.

Produces HTML summary reports from evaluation results. Includes
statistical summaries, domain gap analysis, and embedded visualizations.

Usage:
    from crossdomain_object_tracker.report import generate_report
    generate_report(results, output_dir="outputs/report/")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from crossdomain_object_tracker.evaluate import compare_domains, compute_domain_gap
from crossdomain_object_tracker.visualize import (
    plot_class_distribution,
    plot_confidence_distribution,
    plot_detection_counts,
)

logger = logging.getLogger(__name__)

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cross-Domain Object Detection Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background: #f8f9fa;
        color: #333;
    }}
    h1 {{
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }}
    h2 {{
        color: #34495e;
        margin-top: 30px;
    }}
    h3 {{
        color: #7f8c8d;
    }}
    .summary-card {{
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
        background: white;
    }}
    th, td {{
        border: 1px solid #ddd;
        padding: 10px 14px;
        text-align: left;
    }}
    th {{
        background-color: #3498db;
        color: white;
        font-weight: 600;
    }}
    tr:nth-child(even) {{
        background-color: #f2f2f2;
    }}
    tr:hover {{
        background-color: #e8f4f8;
    }}
    .plot-container {{
        text-align: center;
        margin: 20px 0;
    }}
    .plot-container img {{
        max-width: 100%;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }}
    .metric-box {{
        background: white;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .metric-value {{
        font-size: 2em;
        font-weight: bold;
        color: #3498db;
    }}
    .metric-label {{
        color: #7f8c8d;
        font-size: 0.9em;
    }}
    .gap-table td:first-child {{
        font-weight: bold;
    }}
    .footer {{
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
        color: #95a5a6;
        font-size: 0.85em;
    }}
</style>
</head>
<body>
<h1>Cross-Domain Object Detection Report</h1>
<p>Generated: {timestamp}</p>

<div class="metric-grid">
{metric_boxes}
</div>

<h2>Comparison Summary</h2>
<div class="summary-card">
{comparison_table}
</div>

<h2>Visualizations</h2>

<h3>Detection Statistics</h3>
<div class="plot-container">
<img src="detection_counts.png" alt="Detection Counts">
</div>

<h3>Class Distribution</h3>
<div class="plot-container">
<img src="class_distribution.png" alt="Class Distribution">
</div>

<h3>Confidence Distribution</h3>
<div class="plot-container">
<img src="confidence_distribution.png" alt="Confidence Distribution">
</div>

<h2>Domain Gap Analysis</h2>
<div class="summary-card">
<p>Overall domain gap score: <strong>{overall_gap_score}</strong>
(0 = identical domains, 1 = maximally different)</p>

<h3>Pairwise Class Overlap (Jaccard Similarity)</h3>
{class_overlap_table}

<h3>Pairwise Confidence Difference</h3>
{confidence_diff_table}

<h3>Pairwise Detection Rate Difference</h3>
{detection_rate_table}
</div>

<h2>Per-Dataset Details</h2>
{per_dataset_details}

<div class="footer">
<p>Cross-Domain Object Detection Tracker v0.1.0</p>
</div>
</body>
</html>
"""


def _pairwise_dict_to_html(d: dict[str, float]) -> str:
    """Convert a pairwise metric dict to an HTML table."""
    if not d:
        return "<p>No data available.</p>"

    rows = ""
    for pair, value in d.items():
        a, b = pair.split("_vs_")
        rows += f"<tr><td>{a}</td><td>{b}</td><td>{value}</td></tr>\n"

    return f'<table class="gap-table">\n<tr><th>Dataset A</th><th>Dataset B</th><th>Value</th></tr>\n{rows}</table>'


def _make_metric_box(value: str, label: str) -> str:
    return (
        f'<div class="metric-box"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>'
    )


def generate_report(
    results: dict[str, dict[str, Any]],
    output_dir: str | Path = "outputs/report",
    format: str = "html",
) -> Path:
    """Generate a comprehensive cross-domain comparison report.

    Args:
        results: Dictionary mapping dataset names to evaluation results.
        output_dir: Directory to save the report and associated files.
        format: Output format ('html' supported).

    Returns:
        Path to the generated report file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison DataFrame
    comparison_df = compare_domains(results)
    gap = compute_domain_gap(results)

    # Generate plots
    plot_class_distribution(results, output_path=output_dir / "class_distribution.png")
    plot_confidence_distribution(results, output_path=output_dir / "confidence_distribution.png")
    plot_detection_counts(results, output_path=output_dir / "detection_counts.png")

    # Build metric boxes
    total_images = sum(r.get("num_images", 0) for r in results.values())
    total_dets = sum(r.get("total_detections", 0) for r in results.values())
    num_domains = len(results)

    metric_boxes = "\n".join(
        [
            _make_metric_box(str(num_domains), "Domains Evaluated"),
            _make_metric_box(str(total_images), "Total Images"),
            _make_metric_box(str(total_dets), "Total Detections"),
            _make_metric_box(str(gap["overall_gap_score"]), "Domain Gap Score"),
        ]
    )

    # Build per-dataset details
    per_dataset_html = ""
    for name, res in results.items():
        top_classes = ", ".join(list(res.get("class_distribution", {}).keys())[:10])
        per_dataset_html += f"""
        <div class="summary-card">
        <h3>{name}</h3>
        <ul>
            <li>Images processed: {res.get("num_images", 0)}</li>
            <li>Total detections: {res.get("total_detections", 0)}</li>
            <li>Avg detections/image: {res.get("avg_detections_per_image", 0):.2f}</li>
            <li>Avg confidence: {res.get("avg_confidence", 0):.4f}</li>
            <li>Avg inference time: {res.get("avg_inference_time_ms", 0):.1f} ms</li>
            <li>Top classes: {top_classes or "N/A"}</li>
        </ul>
        </div>
        """

    # Build HTML
    html = _HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metric_boxes=metric_boxes,
        comparison_table=comparison_df.to_html(classes="comparison"),
        overall_gap_score=gap["overall_gap_score"],
        class_overlap_table=_pairwise_dict_to_html(gap["pairwise_class_overlap"]),
        confidence_diff_table=_pairwise_dict_to_html(gap["pairwise_confidence_diff"]),
        detection_rate_table=_pairwise_dict_to_html(gap["pairwise_detection_rate_diff"]),
        per_dataset_details=per_dataset_html,
    )

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    # Also save results JSON alongside the report
    results_json_path = output_dir / "results.json"
    serializable = {}
    for name, res in results.items():
        s = dict(res)
        s.pop("per_image_results", None)
        serializable[name] = s
    results_json_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")

    logger.info("Report generated: %s", report_path)
    return report_path
