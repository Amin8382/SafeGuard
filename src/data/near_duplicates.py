from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from PIL import Image
import numpy as np

from .common import iter_images


def dhash(path: Path, size: int = 8) -> int:
    with Image.open(path) as image:
        gray = image.convert('L').resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.uint8)
    bits = 0
    bit_index = 0
    for row in range(size):
        for col in range(size):
            if pixels[row, col] > pixels[row, col + 1]:
                bits |= 1 << bit_index
            bit_index += 1
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def main():
    parser = argparse.ArgumentParser(description='Report visually near-duplicate images without deleting them.')
    parser.add_argument('--root', default='data/processed')
    parser.add_argument('--threshold', type=int, default=4)
    parser.add_argument('--output', default='reports/near_duplicates.json')
    args = parser.parse_args()

    root = Path(args.root)
    images = list(iter_images(root / 'images'))
    hashes = [(path, dhash(path)) for path in images]
    pairs = []
    for (path_a, hash_a), (path_b, hash_b) in combinations(hashes, 2):
        distance = hamming(hash_a, hash_b)
        if distance <= args.threshold:
            pairs.append({
                'image_a': str(path_a),
                'image_b': str(path_b),
                'hamming_distance': distance,
            })

    report = {
        'root': str(root.resolve()),
        'images_scanned': len(images),
        'threshold': args.threshold,
        'candidate_pairs': len(pairs),
        'pairs': pairs,
        'policy': 'report_only_manual_review_required',
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
