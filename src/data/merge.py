from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from .common import IMAGE_EXTENSIONS, load_yaml


@dataclass(frozen=True)
class Record:
    source: str
    original_split: str
    image_path: Path
    label_path: Path
    sha256: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stable_bucket(content_hash: str, namespace: str) -> int:
    digest = hashlib.sha256(f'{namespace}:{content_hash}'.encode('utf-8')).hexdigest()
    return int(digest[:8], 16) % 100


def choose_split(content_hash: str, split_cfg: dict) -> str:
    train_pct = int(split_cfg.get('train_pct', 80))
    val_pct = int(split_cfg.get('val_pct', 10))
    namespace = str(split_cfg.get('seed_namespace', 'safeguard-v1'))
    bucket = stable_bucket(content_hash, namespace)
    if bucket < train_pct:
        return 'train'
    if bucket < train_pct + val_pct:
        return 'val'
    return 'test'


def parse_label_classes(path: Path) -> set[int]:
    classes: set[int] = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            classes.add(int(float(line.split()[0])))
    return classes


def collect_records(input_root: Path) -> list[Record]:
    records: list[Record] = []
    for source_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for split in ['train', 'val', 'test']:
            image_dir = source_dir / 'images' / split
            label_dir = source_dir / 'labels' / split
            if not image_dir.exists():
                continue
            for image_path in sorted(image_dir.rglob('*')):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label_path = label_dir / f'{image_path.stem}.txt'
                if not label_path.exists():
                    continue
                records.append(
                    Record(
                        source=source_dir.name,
                        original_split=split,
                        image_path=image_path,
                        label_path=label_path,
                        sha256=sha256(image_path),
                    )
                )
    return records


def deduplicate(records: list[Record]) -> tuple[list[Record], list[dict]]:
    # Deterministic choice: source name/path order. The split is recreated later,
    # so retaining a source's original train/val/test location does not create leakage.
    seen: dict[str, Record] = {}
    duplicates: list[dict] = []
    for record in sorted(records, key=lambda r: (r.source, str(r.image_path))):
        kept = seen.get(record.sha256)
        if kept is not None:
            duplicates.append({
                'sha256': record.sha256,
                'duplicate_source': record.source,
                'duplicate_path': str(record.image_path),
                'kept_source': kept.source,
                'kept_path': str(kept.image_path),
            })
            continue
        seen[record.sha256] = record
    return list(seen.values()), duplicates


def ensure_minimum_split_coverage(assignments: dict[str, list[Record]]) -> None:
    """For tiny smoke-test datasets, make val/test non-empty when possible.

    Real datasets are large enough for the hash policy naturally. This fallback is
    deterministic and only moves records from train when a target split is empty.
    """
    total = sum(len(v) for v in assignments.values())
    if total < 3:
        return
    for target in ['val', 'test']:
        if not assignments[target] and len(assignments['train']) > 1:
            assignments['train'].sort(key=lambda r: r.sha256)
            assignments[target].append(assignments['train'].pop())


def write_dataset(
    records: list[Record],
    output_root: Path,
    canonical: dict[int, str],
    split_cfg: dict,
    variant_name: str,
) -> dict:
    assignments: dict[str, list[Record]] = {'train': [], 'val': [], 'test': []}
    for record in records:
        assignments[choose_split(record.sha256, split_cfg)].append(record)
    ensure_minimum_split_coverage(assignments)

    stats = {
        'variant': variant_name,
        'images': 0,
        'splits': {},
        'sources': Counter(),
        'class_counts': Counter(),
        'multi_label_images': 0,
    }
    provenance_rows = []

    for split in ['train', 'val', 'test']:
        out_images = output_root / 'images' / split
        out_labels = output_root / 'labels' / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for record in sorted(assignments[split], key=lambda r: (r.source, r.sha256)):
            new_stem = f'{record.source}_{record.sha256[:12]}_{record.image_path.stem}'
            dest_image = out_images / f'{new_stem}{record.image_path.suffix.lower()}'
            dest_label = out_labels / f'{new_stem}.txt'
            shutil.copy2(record.image_path, dest_image)
            shutil.copy2(record.label_path, dest_label)

            stats['images'] += 1
            stats['sources'][record.source] += 1
            class_ids = parse_label_classes(record.label_path)
            if len(class_ids) > 1:
                stats['multi_label_images'] += 1
            for line in record.label_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    class_id = int(float(line.split()[0]))
                    stats['class_counts'][canonical[class_id]] += 1

            provenance_rows.append({
                'final_split': split,
                'source': record.source,
                'original_split': record.original_split,
                'sha256': record.sha256,
                'original_image': str(record.image_path),
                'output_image': str(dest_image),
            })

        stats['splits'][split] = len(assignments[split])

    dataset_yaml = {
        'path': str(output_root.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': canonical,
    }
    (output_root / 'safeguard.yaml').write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding='utf-8'
    )

    with (output_root / 'provenance.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(provenance_rows[0].keys()) if provenance_rows else [
            'final_split', 'source', 'original_split', 'sha256', 'original_image', 'output_image'
        ])
        writer.writeheader()
        writer.writerows(provenance_rows)

    stats['sources'] = dict(stats['sources'])
    stats['class_counts'] = dict(stats['class_counts'])
    (output_root / 'dataset_manifest.json').write_text(
        json.dumps(stats, indent=2), encoding='utf-8'
    )
    return stats


def write_dataset_card(output_root: Path, merged_stats: dict, duplicates: list[dict], canonical: dict[int, str]) -> None:
    class_lines = '\n'.join(f'- `{idx}`: `{name}`' for idx, name in canonical.items())
    source_lines = '\n'.join(f'- {source}: {count} images' for source, count in merged_stats['sources'].items())
    class_count_lines = '\n'.join(f'- {name}: {count} boxes' for name, count in merged_stats['class_counts'].items())
    text = f"""# SafeGuard Unified Dataset Card

## Purpose
SafeGuard's training data is built from the semantic intersection of the Roboflow Construction Site Safety and Ultralytics Construction-PPE datasets. Source-only classes are deliberately excluded to avoid contradictory supervision.

## Canonical ontology
{class_lines}

## Final split policy
A deterministic content-hash split is applied *after* harmonization and exact deduplication. This prevents exact duplicate images from crossing train/validation/test boundaries and gives every rebuild the same split.

- Train: {merged_stats['splits'].get('train', 0)} images
- Validation: {merged_stats['splits'].get('val', 0)} images
- Test: {merged_stats['splits'].get('test', 0)} images

## Source contribution
{source_lines or '- No images'}

## Class distribution
{class_count_lines or '- No boxes'}

## Deduplication
- Exact duplicates removed across sources: {len(duplicates)}

## Reproducibility
Every final image is recorded in `provenance.csv` with its source dataset, original split, SHA-256 content hash, and final split. Source audits are stored under `reports/`.
"""
    (output_root / 'DATASET_CARD.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--lock', default='configs/dataset_lock.yaml')
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    cfg = load_yaml(Path(args.config))
    lock = load_yaml(Path(args.lock))
    canonical = {int(k): v for k, v in cfg['canonical_classes'].items()}
    split_cfg = lock.get('split_policy', {})

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = collect_records(input_root)
    unique_records, duplicates = deduplicate(records)

    # Main merged dataset.
    merged_stats = write_dataset(unique_records, output_root, canonical, split_cfg, 'merged')

    # Source-only variants use the same canonical ontology and split policy so the
    # model comparison (Roboflow-only vs Ultralytics-only vs merged) is fair.
    variants_root = output_root / 'variants'
    variant_stats = {}
    for source in sorted({r.source for r in unique_records}):
        source_records = [r for r in unique_records if r.source == source]
        variant_stats[source] = write_dataset(
            source_records,
            variants_root / source,
            canonical,
            split_cfg,
            source,
        )

    (output_root / 'duplicates.json').write_text(json.dumps(duplicates, indent=2), encoding='utf-8')
    write_dataset_card(output_root, merged_stats, duplicates, canonical)

    summary = {
        'merged': merged_stats,
        'variants': variant_stats,
        'duplicates_removed': len(duplicates),
    }
    (output_root / 'build_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
