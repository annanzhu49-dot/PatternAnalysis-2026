
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DATA_ROOT = Path("/home/groups/comp3710/HipMRI_Study_open")


def get_data_pairs(root=DATA_ROOT):
    """Find matching MRI and segmentation mask files."""
    mri_dir = root / "semantic_MRs"
    mask_dir = root / "semantic_labels_only"

    pairs = []

    for mri_path in sorted(mri_dir.glob("*_LFOV.nii.gz")):
        prefix = mri_path.name.replace("_LFOV.nii.gz", "")
        mask_path = mask_dir / f"{prefix}_SEMANTIC.nii.gz"

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        patient_id = prefix.split("_")[0]
        pairs.append((patient_id, mri_path, mask_path))

    return pairs


def split_by_patient(pairs, seed=42):
    """Split patients without leaking scans across datasets."""
    patients = sorted({p[0] for p in pairs})

    rng = random.Random(seed)
    rng.shuffle(patients)

    train_ids = set(patients[:26])
    val_ids = set(patients[26:32])
    test_ids = set(patients[32:])

    train = [p for p in pairs if p[0] in train_ids]
    val = [p for p in pairs if p[0] in val_ids]
    test = [p for p in pairs if p[0] in test_ids]

    return train, val, test


class HipMRIDataset(Dataset):
    def __init__(self, pairs, target_shape=(128, 128, 64)):
        self.pairs = pairs
        self.target_shape = target_shape

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        patient_id, mri_path, mask_path = self.pairs[index]

        # Load 3D MRI and mask
        mri_img = nib.load(str(mri_path))
        mask_img = nib.load(str(mask_path))

        if mri_img.shape != mask_img.shape:
            raise ValueError(f"Shape mismatch: {mri_path.name}")

        if not np.allclose(mri_img.affine, mask_img.affine):
            raise ValueError(f"Affine mismatch: {mri_path.name}")

        mri = np.asarray(mri_img.dataobj, dtype=np.float32)
        mask = np.asarray(mask_img.dataobj).astype(np.int64)

        # Normalize MRI using nonzero voxels
        foreground = mri != 0

        if foreground.any():
            mean = mri[foreground].mean()
            std = mri[foreground].std()
            mri = (mri - mean) / max(float(std), 1e-6)

        # Convert to tensors: [C, D, H, W]
        mri = torch.from_numpy(mri.copy()).unsqueeze(0).unsqueeze(0)
        mask = torch.from_numpy(mask.copy()).unsqueeze(0).unsqueeze(0)

        # Downsample MRI with trilinear interpolation
        mri = F.interpolate(
            mri,
            size=self.target_shape,
            mode="trilinear",
            align_corners=False
        )

        # Downsample mask with nearest-neighbor interpolation
        mask = F.interpolate(
            mask.float(),
            size=self.target_shape,
            mode="nearest"
        )

        # MRI: [1, D, H, W]
        # Mask: [D, H, W]
        return mri.squeeze(0), mask.squeeze(0).squeeze(0).long()


if __name__ == "__main__":
    pairs = get_data_pairs()
    train, val, test = split_by_patient(pairs)

    print("Total scans:", len(pairs))
    print("Train scans:", len(train))
    print("Validation scans:", len(val))
    print("Test scans:", len(test))

    train_ids = {p[0] for p in train}
    val_ids = {p[0] for p in val}
    test_ids = {p[0] for p in test}

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)

    print("Patient-level split: OK")

    dataset = HipMRIDataset(train)
    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Mask labels:", torch.unique(mask))

