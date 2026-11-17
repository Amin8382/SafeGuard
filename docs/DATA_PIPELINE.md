# SafeGuard Data Pipeline

## Why intersection instead of union?

A union would keep every class from both datasets. That creates inconsistent negative supervision: an object can be visible in one dataset but unlabeled because that source never targeted that class. Training YOLO would then sometimes treat the same real object as foreground and sometimes as background.

SafeGuard therefore keeps only semantically shared labels.

```text
Roboflow                     Ultralytics                 Canonical
Person                       Person                      person
Hardhat                      helmet                      helmet
Safety Vest                  vest                        vest
NO-Hardhat                   no_helmet                   no_helmet
```

## Stages

```text
raw/roboflow -------\
                     -> audit -> harmonize -> exact dedup -> deterministic split -> unified dataset
raw/ultralytics ----/
                                                           |-> merged
                                                           |-> roboflow-only
                                                           `-> ultralytics-only
```

### Audit

Checks:
- image/split counts;
- class names/counts;
- missing labels;
- malformed label lines;
- normalized bounding boxes;
- object-size distribution;
- selected release expectations.

### Harmonization

Every source class ID is first resolved to its **class name**, then mapped to a canonical class. This protects the project against a numeric-ID reorder between source releases.

Images with zero intersection annotations are dropped from V1.

### Exact deduplication

SHA-256 is calculated on image bytes across both sources. Exact duplicate images are collapsed before splitting.

### Re-splitting

The original dataset splits are not mixed directly. SafeGuard calculates a deterministic bucket from the image content hash and a fixed namespace. Exact duplicates therefore always resolve to the same final split.

### Near duplicates

Perceptual near-duplicate pairs are reported for manual review. They are not automatically removed because similar construction-site frames can be legitimate separate observations.

### Provenance

`provenance.csv` records:
- final split;
- original source;
- original split;
- SHA-256;
- original path;
- final output path.

This is the audit trail connecting the final training image back to its source.
