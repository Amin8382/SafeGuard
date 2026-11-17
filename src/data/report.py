from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Generate the SafeGuard data-quality report.")
    parser.add_argument("--audit", default="reports/source_audit.json")
    parser.add_argument("--harmonization", default="data/interim/harmonization_report.json")
    parser.add_argument("--build", default="data/processed/build_report.json")
    parser.add_argument("--near-duplicates", default="reports/near_duplicates.json")
    parser.add_argument("--output", default="reports/DATA_QUALITY_REPORT.md")
    args = parser.parse_args()

    audit = load_json(Path(args.audit), [])
    harmonization = load_json(Path(args.harmonization), [])
    build = load_json(Path(args.build), {})
    near = load_json(Path(args.near_duplicates), {})

    lines = [
        "# SafeGuard Data Quality Report",
        "",
        "## Canonical intersection ontology",
        "",
        "`person`, `helmet`, `vest`, `no_helmet`",
        "",
        "## Source audit",
        "",
        "| Source | Images | Boxes | Classes | Missing labels | Invalid boxes | Lock OK |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in audit:
        lines.append(
            f"| {item.get('source')} | {item.get('images_total', 0)} | {item.get('boxes_total', 0)} | "
            f"{item.get('class_count', 0)} | {item.get('missing_label_files', 0)} | "
            f"{item.get('invalid_boxes', 0)} | {item.get('lock_validation', {}).get('all_ok', True)} |"
        )

    lines += ["", "## Harmonization", ""]
    for item in harmonization:
        lines += [
            f"### {item.get('source')}",
            f"- Images seen: {item.get('images_seen', 0)}",
            f"- Images kept: {item.get('images_kept', 0)}",
            f"- Intersection boxes kept: {item.get('boxes_kept', 0)}",
            f"- Source-only boxes dropped: {item.get('boxes_dropped', 0)}",
            f"- Images dropped with no intersection annotations: {item.get('images_dropped_no_intersection_boxes', 0)}",
            "",
        ]

    merged = build.get("merged", {})
    lines += [
        "## Final merged dataset",
        "",
        f"- Images: {merged.get('images', 0)}",
        f"- Train / val / test: {merged.get('splits', {})}",
        f"- Source contribution: {merged.get('sources', {})}",
        f"- Class counts: {merged.get('class_counts', {})}",
        f"- Exact duplicates removed: {build.get('duplicates_removed', 0)}",
        "",
        "## Near-duplicate review",
        "",
        f"- Candidate pairs reported: {len(near.get('pairs', [])) if isinstance(near, dict) else 0}",
        "- Near duplicates are review-only and are never removed automatically.",
        "",
        "## Leakage policy",
        "",
        "Exact deduplication is performed before a deterministic content-hash train/validation/test split. "
        "This prevents an exact image duplicate from crossing evaluation boundaries.",
        "",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
