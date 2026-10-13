"""Compact rebuttal Markdown tables for the readout-sensitivity analysis.

Tables are generated at dataset level so incompatible semantic targets are never
pooled, and no global cross-module score is produced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_ci(ci: Sequence[float] | None) -> str:
    if not ci:
        return "n/a"
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def _readout_label(readout: str, logistic_c: float | None) -> str:
    if readout == "nearest_centroid":
        return "Nearest centroid"
    return f"Logistic C={logistic_c:g}"


def _ordering_bs_direction_rows(
    module_summary: Mapping[str, Any],
    paired: Mapping[str, Any],
    module: str,
) -> List[str]:
    """One row per (dataset, readout): ordering, SA-REL adjusted CI, direction."""
    rows: List[str] = []
    for entry in module_summary["entries"]:
        dataset = entry["dataset"]
        feature_type = entry["feature_type"]
        readout = entry["readout"]
        logistic_c = entry["logistic_c"]
        ordering = entry["ordering"]["ordering_string"]
        sa_rel = _lookup_paired(paired, module, dataset, feature_type, readout, logistic_c, "SA-REL")
        adjusted = _fmt_ci(sa_rel["ci_bonferroni"]) if sa_rel else "n/a"
        retained = "yes" if (sa_rel and sa_rel["excludes_zero_bonferroni"]) else "no"
        iso_rel_sa = entry["ordering"]["iso_rel_sa_means"]
        triple = "/".join(_fmt(iso_rel_sa.get(k)) for k in ("ISO", "REL", "SA"))
        rows.append(
            f"| {dataset} ({feature_type}) | {_readout_label(readout, logistic_c)} | {triple} | {adjusted} | {retained} |"
        )
    return rows


def _lookup_paired(paired, module, dataset, feature_type, readout, logistic_c, comparison):
    for row in paired["rows"]:
        if (
            row["module"] == module
            and row["dataset"] == dataset
            and row["feature_type"] == feature_type
            and row["readout"] == readout
            and row["logistic_c"] == logistic_c
            and row["comparison"] == comparison
        ):
            return row
    return None


def render_markdown(
    semantic_summary: Mapping[str, Any],
    anatomical_summary: Mapping[str, Any],
    paired: Mapping[str, Any],
    agreement: Mapping[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Readout-Sensitivity Compact Tables")
    lines.append("")
    lines.append(
        "Dataset-level tables. Semantic targets are annotation-derived and are not "
        "pooled across datasets. No global cross-module score is produced. "
        "`SA-REL [adjusted 95% CI]` uses a Bonferroni-adjusted interval across "
        "{REL-ISO, SA-ISO, SA-REL}. An interval containing zero is not evidence of equivalence."
    )
    lines.append("")

    lines.append("## Semantic transfer")
    lines.append("")
    lines.append("| Dataset | Readout | ISO/REL/SA | SA-REL [adjusted 95% CI] | Direction retained |")
    lines.append("|---|---|---:|---:|---:|")
    lines.extend(_ordering_bs_direction_rows(semantic_summary, paired, "semantic_transfer"))
    lines.append("")

    lines.append("## Low-shot transfer")
    lines.append("")
    lines.append("| Dataset | Readout | ISO/REL/SA BA | SA-REL [adjusted 95% CI] | Direction retained |")
    lines.append("|---|---|---:|---:|---:|")
    lines.extend(_ordering_bs_direction_rows(anatomical_summary, paired, "low_shot_transfer"))
    lines.append("")

    lines.append("## Cross-module readout summary")
    lines.append("")
    lines.append("| Module | Logistic C=1 ordering | Nearest-centroid ordering | Matching dataset-feature surfaces |")
    lines.append("|---|---|---|---:|")
    for module_key, module_label in (
        ("semantic_transfer", "Semantic transfer"),
        ("low_shot_transfer", "Low-shot transfer"),
    ):
        block = agreement["ordering_agreement"][module_key]
        lines.append(
            f"| {module_label} | {block['logistic_c1_orderings']} | "
            f"{block['nearest_centroid_orderings']} | "
            f"{block['n_matching_surfaces']}/{block['n_surfaces']} |"
        )
    lines.append("")
    return "\n".join(lines)
