from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import yaml

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def find_dataset_yaml(root: Path) -> Path:
    preferred = [root / 'data.yaml', root / 'dataset.yaml', root / 'construction-ppe.yaml']
    for candidate in preferred:
        if candidate.exists():
            return candidate

    yamls = sorted(list(root.glob('*.yaml')) + list(root.glob('*.yml')))
    if yamls:
        return yamls[0]

    # Tolerate archives that extract one extra top-level folder.
    nested = sorted(list(root.glob('*/*.yaml')) + list(root.glob('*/*.yml')))
    if nested:
        return nested[0]

    raise FileNotFoundError(
        f'No dataset YAML found in {root}. Put the exported dataset, including its YAML file, in this folder.'
    )


def normalize_names(names) -> List[str]:
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    raise ValueError('Dataset YAML must contain names as a list or index-keyed dictionary.')


def _candidate_base(yaml_path: Path, meta: dict) -> Path:
    configured = Path(meta.get('path', '.'))
    base = configured if configured.is_absolute() else (yaml_path.parent / configured).resolve()
    if base.exists():
        return base

    # Some downloaded dataset YAML files retain a package-level `path:` value even
    # when users copy the YAML inside the dataset folder. If train/val paths resolve
    # directly from the YAML's folder, prefer that real layout rather than failing.
    train = meta.get('train')
    if isinstance(train, str) and (yaml_path.parent / train).exists():
        return yaml_path.parent.resolve()
    return base


def resolve_dataset(root: Path) -> Tuple[Path, dict, List[str]]:
    yaml_path = find_dataset_yaml(root)
    meta = load_yaml(yaml_path)
    names = normalize_names(meta.get('names'))
    base = _candidate_base(yaml_path, meta)
    return base, meta, names


def split_image_dirs(base: Path, meta: dict) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for target, keys in {'train': ['train'], 'val': ['val', 'valid'], 'test': ['test']}.items():
        value = None
        for key in keys:
            if meta.get(key):
                value = meta[key]
                break
        if value is None:
            continue
        if isinstance(value, list):
            raise ValueError(f'Multiple image roots for {target} are not supported in V1.')
        p = Path(value)
        if not p.is_absolute():
            p = (base / p).resolve()
        result[target] = p
    return result


def label_path_for_image(image_path: Path, image_root: Path) -> Path:
    # Standard YOLO layouts supported:
    #   .../images/train/x.jpg -> .../labels/train/x.txt
    #   .../train/images/x.jpg -> .../train/labels/x.txt
    parts = list(image_path.parts)
    indices = [i for i, part in enumerate(parts) if part == 'images']
    if indices:
        i = indices[-1]
        parts[i] = 'labels'
        return Path(*parts).with_suffix('.txt')

    split_name = image_root.name
    parent = image_root.parent
    if parent.name == 'images':
        return parent.parent / 'labels' / split_name / f'{image_path.stem}.txt'
    return parent / 'labels' / f'{image_path.stem}.txt'


def iter_images(folder: Path):
    if not folder.exists():
        return
    for p in sorted(folder.rglob('*')):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p
