"""Reusable Markdown-compatible table rendering for dataset-metric layouts."""

from __future__ import annotations

from html import escape
from typing import Mapping, Sequence


def render_dataset_metric_table(
    *,
    row_header: str,
    rows: Sequence[tuple[str, str]],
    datasets: Sequence[tuple[str, str]],
    metrics: Sequence[tuple[str, str]],
    values: Mapping[tuple[str, str, str], str],
) -> list[str]:
    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        f'      <th rowspan="2">{escape(row_header)}</th>',
    ]
    for _, dataset_label in datasets:
        lines.append(
            f'      <th colspan="{len(metrics)}">{escape(dataset_label)}</th>'
        )
    lines.extend(["    </tr>", "    <tr>"])
    for _ in datasets:
        for _, metric_label in metrics:
            lines.append(f"      <th>{escape(metric_label)}</th>")
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])

    for row_key, row_label in rows:
        lines.extend(["    <tr>", f"      <td>{escape(row_label)}</td>"])
        for dataset_key, _ in datasets:
            for metric_key, _ in metrics:
                value_key = (row_key, dataset_key, metric_key)
                if value_key not in values:
                    raise KeyError(f"Missing Markdown table value: {value_key}")
                lines.append(
                    f'      <td style="text-align: right;">{values[value_key]}</td>'
                )
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return lines
