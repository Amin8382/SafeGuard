from pathlib import Path

from PIL import Image

from src.data.merge import Record, choose_split, deduplicate, write_dataset


def test_choose_split_is_deterministic():
    cfg = {'train_pct': 80, 'val_pct': 10, 'test_pct': 10, 'seed_namespace': 'test'}
    h = 'a' * 64
    assert choose_split(h, cfg) == choose_split(h, cfg)


def test_deduplicate_keeps_one_exact_image(tmp_path: Path):
    image = tmp_path / 'a.jpg'
    Image.new('RGB', (32, 32), 'white').save(image)
    label = tmp_path / 'a.txt'
    label.write_text('0 0.5 0.5 0.4 0.4\n', encoding='utf-8')
    from src.data.merge import sha256
    h = sha256(image)
    records = [
        Record('roboflow', 'train', image, label, h),
        Record('ultralytics', 'val', image, label, h),
    ]
    unique, duplicates = deduplicate(records)
    assert len(unique) == 1
    assert len(duplicates) == 1


def test_write_dataset_emits_provenance_and_yaml(tmp_path: Path):
    canonical = {0: 'person', 1: 'helmet', 2: 'vest', 3: 'no_helmet'}
    records = []
    from src.data.merge import sha256
    for i in range(6):
        image = tmp_path / f'i{i}.jpg'
        Image.new('RGB', (40 + i, 40), (i * 20, 10, 20)).save(image)
        label = tmp_path / f'i{i}.txt'
        label.write_text('0 0.5 0.5 0.6 0.8\n1 0.5 0.2 0.2 0.2\n', encoding='utf-8')
        records.append(Record('roboflow', 'train', image, label, sha256(image)))

    output = tmp_path / 'out'
    stats = write_dataset(
        records,
        output,
        canonical,
        {'train_pct': 80, 'val_pct': 10, 'test_pct': 10, 'seed_namespace': 'unit-test'},
        'merged',
    )
    assert stats['images'] == 6
    assert (output / 'safeguard.yaml').exists()
    assert (output / 'provenance.csv').exists()
    assert all((output / 'images' / split).exists() for split in ['train', 'val', 'test'])


def test_dataset_yaml_path_fallback_when_packaged_path_is_stale(tmp_path: Path):
    from src.data.common import resolve_dataset, split_image_dirs
    import yaml

    root = tmp_path / 'dataset'
    (root / 'images' / 'train').mkdir(parents=True)
    meta = {
        'path': 'construction-ppe',
        'train': 'images/train',
        'names': ['helmet', 'Person'],
    }
    (root / 'construction-ppe.yaml').write_text(yaml.safe_dump(meta), encoding='utf-8')
    base, parsed, names = resolve_dataset(root)
    assert base == root.resolve()
    assert names == ['helmet', 'Person']
    assert split_image_dirs(base, parsed)['train'] == (root / 'images' / 'train').resolve()


def test_compare_with_lock_validates_generic_release_metadata():
    from src.data.audit import compare_with_lock

    report = {
        'source': 'roboflow',
        'images_total': 2799,
        'class_count': 10,
        'classes': [
            'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
            'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
        ],
        'splits': {
            'train': {'images': 2603},
            'val': {'images': 114},
            'test': {'images': 82},
        },
    }
    lock = {
        'selected_release': {
            'expected_images': 2799,
            'expected_classes': 10,
            'expected_class_names': report['classes'],
            'expected_splits': {'train': 2603, 'val': 114, 'test': 82},
        }
    }

    result = compare_with_lock(report, lock)

    assert result['all_ok'] is True
    assert all(check['ok'] for check in result['checks'])


def test_roboflow_parent_prefixed_split_paths_resolve_inside_dataset(tmp_path):
    from src.data.common import resolve_dataset, split_image_dirs

    root = tmp_path / 'roboflow'
    (root / 'train' / 'images').mkdir(parents=True)
    (root / 'valid' / 'images').mkdir(parents=True)
    (root / 'test' / 'images').mkdir(parents=True)
    (root / 'data.yaml').write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 1\nnames: ['Person']\n",
        encoding='utf-8',
    )

    base, parsed, _ = resolve_dataset(root)
    splits = split_image_dirs(base, parsed)

    assert splits['train'] == (root / 'train' / 'images').resolve()
    assert splits['val'] == (root / 'valid' / 'images').resolve()
    assert splits['test'] == (root / 'test' / 'images').resolve()
