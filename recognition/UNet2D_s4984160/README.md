# HipMRI 2D U-Net comparison: dataset module

This directory currently contains the independently reviewed 2D data loader. A 2D model and training/evaluation results have not yet been implemented here.

## Data layout

Pass either `/home/groups/comp3710/HipMRI_Study_open` or its `keras_slices_data` directory. MRI and mask files reside in separate sibling folders:

```text
keras_slices_data/
  keras_slices_train/          keras_slices_seg_train/
  keras_slices_validate/       keras_slices_seg_validate/
  keras_slices_test/           keras_slices_seg_test/
```

Files are paired by patient, week and numeric slice index. Missing, orphan or duplicate files raise an error instead of being silently skipped. A prepared flat local directory with paired `case_...` and `seg_...` files is also supported.

## Patient separation

`patient_split.json` preserves the existing 3D experiment's split: sort the original letter-prefixed 3D patient IDs, shuffle using Python `random.Random(42)`, then select 26/6/6 patients. The verified unique numeric suffix maps those IDs to 2D filenames. Do not sort numeric IDs and take the first 26, or use the provided directory split as our experimental split.

The loader indexes all supplied source folders, then applies this manifest without moving data. With the complete dataset, expected slice-pair counts are 8,700 training, 2,340 validation and 1,620 test. Passing a different manifest is explicit; overlapping, missing or unassigned patients are rejected.

## Preprocessing and usage

Each original MRI slice is Z-score normalized using its original pixels before padding. The mask must contain integer labels 0..5. Shapes `(256,128)` and `(256,144)`, including a trailing singleton dimension, are accepted. Smaller widths are padded on the right to `(256,144)`; MRI padding is zero and label padding is **255**. The optional transform jointly processes image and mask before padding, and must preserve shape and integer labels. Validation/test transforms are disabled by the loader helper.

```python
from dataset import IGNORE_INDEX, get_dataloaders
import torch

train_loader, val_loader, test_loader = get_dataloaders(
    '/home/groups/comp3710/HipMRI_Study_open',
    batch_size=16,
    num_workers=0,
)
criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
image, mask, metadata = next(iter(train_loader))
# image: float32 [B,1,256,144]; mask: int64 [B,256,144]
```

Every loss and metric must exclude `mask == IGNORE_INDEX`. Remove the recorded right padding before restoring predictions to the original image grid. Raw mask values are checked before integer conversion to avoid silently truncating fractional labels. Per-slice normalization here differs from the 3D pipeline's per-volume nonzero-voxel normalization; report this preprocessing difference when interpreting a future 2D/3D comparison.

## Spatial mapping

The directory audit matched all 211 scan keys to 60 numbered slices each. Pixel checks across 69 different slice pairs from four scans found:

```text
slice_2d[i] = original_volume[::-1, 98+i, :], i=0..59
```

The original 3D headers use L/P/S axes; the matched planes are coronal. The slice metadata records this source axis/index/flip. This rule was content-checked on those samples, not every slice. Sampled 2D headers have identity affines; they do not independently provide the original physical geometry. Use original 3D affines for distances and alignment. Two-dimensional data cover only the central 60 planes; future comparisons must use a shared spatial region/grid rather than compare partial coverage with full-volume scores.

## Checks and provenance

Run `python -m unittest discover -s recognition/UNet2D_s4984160/tests -v` from the repository root with NumPy, NiBabel and PyTorch installed. Tests create synthetic NIfTI fixtures; they do not use patient images. Synthetic tests check loading behavior, not full real-data correctness or model performance.

The loader was adapted from the author's submitted draft, with AI assistance reviewing directory pairing, explicit split handling, normalization/padding, input validation and tests. The author's original attachment was preserved. No friend's implementation, model weights, experimental scores or report text are included. Final project documentation will need the full AI-use disclosure, exact environment versions and measured experimental results.
