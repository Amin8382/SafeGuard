from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from PIL import Image

from .common import find_dataset_yaml, iter_images, label_path_for_image, load_yaml, resolve_dataset, split_image_dirs


def audit_dataset(source: str, root: Path) -> dict:
    base, meta, names = resolve_dataset(root)
    splits = split_image_dirs(base, meta)

    report = {
        'source': source,
        'root': str(root.resolve()),
        'dataset_yaml': str(find_dataset_yaml(root).resolve()),
        'classes': names,
        'class_count': len(names),
        'splits': {},
        'images_total': 0,
        'labels_total': 0,
        'boxes_total': 0,
        'missing_label_files': 0,
        'invalid_label_lines': 0,
        'invalid_boxes': 0,
        'class_counts': Counter(),
        'image_widths': [],
        'image_heights': [],
        'box_area_ratio': [],
    }

    for split, image_dir in splits.items():
        split_stats = defaultdict(int)
        for image_path in iter_images(image_dir):
            split_stats['images'] += 1
            report['images_total'] += 1
            try:
                with Image.open(image_path) as im:
                    w, h = im.size
                    report['image_widths'].append(w)
                    report['image_heights'].append(h)
            except Exception:
                w = h = None

            label_path = label_path_for_image(image_path, image_dir)
            if not label_path.exists():
                report['missing_label_files'] += 1
                split_stats['missing_labels'] += 1
                continue
            report['labels_total'] += 1

            for raw in label_path.read_text(encoding='utf-8', errors='ignore').splitlines():
                if not raw.strip():
                    continue
                parts = raw.split()
                if len(parts) != 5:
                    report['invalid_label_lines'] += 1
                    split_stats['invalid_lines'] += 1
                    continue
                try:
                    class_id = int(float(parts[0]))
                    x, y, bw, bh = map(float, parts[1:])
                except ValueError:
                    report['invalid_label_lines'] += 1
                    split_stats['invalid_lines'] += 1
                    continue
                if class_id < 0 or class_id >= len(names):
                    report['invalid_label_lines'] += 1
                    split_stats['invalid_lines'] += 1
                    continue
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    report['invalid_boxes'] += 1
                    split_stats['invalid_boxes'] += 1
                    continue
                report['boxes_total'] += 1
                split_stats['boxes'] += 1
                report['class_counts'][names[class_id]] += 1
                report['box_area_ratio'].append(bw * bh)

        report['splits'][split] = dict(split_stats)

    def summary(values):
        if not values:
            return None
        return {
            'min': min(values),
            'median': median(values),
            'max': max(values),
        }

    report['class_counts'] = dict(report['class_counts'])
    report['image_width_summary'] = summary(report.pop('image_widths'))
    report['image_height_summary'] = summary(report.pop('image_heights'))
    areas = report.pop('box_area_ratio')
    report['box_area_ratio_summary'] = summary(areas)
    report['small_box_fraction_lt_1pct'] = (
        sum(1 for x in areas if x < 0.01) / len(areas) if areas else None
    )
    return report


def compare_with_lock(report: dict, source_cfg: dict) -> dict:
    checks = []
    rel = source_cfg.get('selected_release', {})

    expected_images = rel.get('expected_images', source_cfg.get('expected_project_images'))
    if expected_images is not None:
        checks.append({
            'name': 'image_count',
            'expected': expected_images,
            'actual': report['images_total'],
            'ok': expected_images == report['images_total'],
        })

    expected_classes = rel.get('expected_classes')
    if expected_classes is not None:
        checks.append({
            'name': 'class_count',
            'expected': expected_classes,
            'actual': report['class_count'],
            'ok': expected_classes == report['class_count'],
        })

    if 'expected_class_names' in rel:
        expected_names = set(rel.get('expected_class_names', []))
        actual_names = set(report['classes'])
        checks.append({
            'name': 'class_names',
            'expected': sorted(expected_names),
            'actual': sorted(actual_names),
            'ok': expected_names == actual_names,
        })

    for split_name, expected_count in rel.get('expected_splits', {}).items():
        actual_count = report.get('splits', {}).get(split_name, {}).get('images', 0)
        checks.append({
            'name': f'split_{split_name}_images',
            'expected': expected_count,
            'actual': actual_count,
            'ok': expected_count == actual_count,
        })

    return {'checks': checks, 'all_ok': all(c['ok'] for c in checks) if checks else True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/datasets.yaml')
    parser.add_argument('--lock', default='configs/dataset_lock.yaml')
    parser.add_argument('--output', default='reports/source_audit.json')
    parser.add_argument('--strict', action='store_true', help='Exit non-zero when a source differs from the locked release.')
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    lock = load_yaml(Path(args.lock))
    reports = []
    for source, source_cfg in cfg['sources'].items():
        report = audit_dataset(source, Path(source_cfg['root']))
        report['lock_validation'] = compare_with_lock(report, lock.get('sources', {}).get(source, {}))
        reports.append(report)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))
    if args.strict:
        lock_bad = any(not r.get('lock_validation', {}).get('all_ok', True) for r in reports)
        integrity_bad = any(
            r.get('invalid_label_lines', 0) > 0 or r.get('invalid_boxes', 0) > 0
            for r in reports
        )
        if lock_bad or integrity_bad:
            raise SystemExit(
                'Dataset audit gate failed. Review reports/source_audit.json before continuing.'
            )


if __name__ == '__main__':
    main()
