from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
import yaml

from .common import iter_images, label_path_for_image, load_yaml, resolve_dataset, split_image_dirs


def harmonize_source(source_name: str, source_cfg: dict, canonical: dict, output_root: Path) -> dict:
    root = Path(source_cfg['root']).resolve()
    base, meta, source_names = resolve_dataset(root)
    splits = split_image_dirs(base, meta)

    canonical_name_to_id = {name: int(idx) for idx, name in canonical.items()}
    source_name_to_id = {name: idx for idx, name in enumerate(source_names)}

    class_map = source_cfg['class_map']
    unknown = [name for name in class_map if name not in source_name_to_id]
    if unknown:
        raise ValueError(f'{source_name}: configured labels not found in dataset YAML: {unknown}')

    source_id_to_canonical_id = {
        source_name_to_id[src]: canonical_name_to_id[dst]
        for src, dst in class_map.items()
    }

    stats = {
        'source': source_name,
        'source_root': str(root),
        'classes_original': source_names,
        'images_seen': 0,
        'images_kept': 0,
        'images_dropped_no_intersection_boxes': 0,
        'boxes_kept': 0,
        'boxes_dropped': 0,
        'class_counts': Counter(),
        'splits': {},
    }

    for split, image_dir in splits.items():
        out_images = output_root / source_name / 'images' / split
        out_labels = output_root / source_name / 'labels' / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        split_seen = split_kept = 0
        for image_path in iter_images(image_dir):
            split_seen += 1
            stats['images_seen'] += 1
            label_path = label_path_for_image(image_path, image_dir)
            if not label_path.exists():
                stats['images_dropped_no_intersection_boxes'] += 1
                continue

            kept_lines = []
            for raw in label_path.read_text(encoding='utf-8').splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                if len(parts) < 5:
                    continue
                try:
                    source_id = int(float(parts[0]))
                except ValueError:
                    continue
                if source_id not in source_id_to_canonical_id:
                    stats['boxes_dropped'] += 1
                    continue
                canonical_id = source_id_to_canonical_id[source_id]
                kept_lines.append(' '.join([str(canonical_id)] + parts[1:]))
                stats['boxes_kept'] += 1
                stats['class_counts'][canonical[canonical_id]] += 1

            # V1 policy: only keep images containing at least one intersection annotation.
            if not kept_lines:
                stats['images_dropped_no_intersection_boxes'] += 1
                continue

            shutil.copy2(image_path, out_images / image_path.name)
            (out_labels / f'{image_path.stem}.txt').write_text('\n'.join(kept_lines) + '\n', encoding='utf-8')
            stats['images_kept'] += 1
            split_kept += 1

        stats['splits'][split] = {'seen': split_seen, 'kept': split_kept}

    stats['class_counts'] = dict(stats['class_counts'])
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    canonical = {int(k): v for k, v in cfg['canonical_classes'].items()}
    output_root = Path(args.output)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for source_name, source_cfg in cfg['sources'].items():
        all_stats.append(harmonize_source(source_name, source_cfg, canonical, output_root))

    (output_root / 'harmonization_report.json').write_text(
        json.dumps(all_stats, indent=2), encoding='utf-8'
    )
    print(json.dumps(all_stats, indent=2))


if __name__ == '__main__':
    main()
