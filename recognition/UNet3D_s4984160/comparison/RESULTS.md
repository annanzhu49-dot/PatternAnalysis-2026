## Model comparison: complete volume and shared band

The tables below separate measured scopes. Pending is not a score of zero.
No 2D test result or 3D-versus-2D ranking is available yet. Improved 2D
U-Net is an optional unscheduled row, not an additional active experiment.

### A. Complete-volume 3D reference

| Model / scope | Test patients/scans | Mean scan Dice ± SD | Mean IoU | HD95 (approx. mm) | Pooled volume ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard 3D U-Net; 64×128×128 full grid | 6 / 27 | 0.7732 ± 0.0557 | 0.6336 | 6.50 | 1.1775 |

This table supports the complete-volume 3D result. Native 2D partial-slice
results cannot be placed in this scope or used to claim a direct advantage.

### B. Shared-band prostate comparison (separate evaluation scope)

| Model | Test patients/scans | Mean scan Dice ± SD | Mean patient Dice | Pooled precision / recall | Pooled volume ratio | Shared HD95 / organ invasion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Standard 3D U-Net | 6 / 27 | 0.7732 ± 0.0557 | 0.7678 | 0.7147 / 0.8416 | 1.1775 | Not measured for this scope |
| Standard 2D U-Net | Required: same 6 / 27 | Pending | Pending | Pending | Pending | Pending |
| Improved 2D U-Net (optional; not scheduled) | Required: same 6 / 27 | Pending | Pending | Pending | Pending | Pending |

Shared grid: D=64, H=128, full-volume W indices 49..78 (30 planes).
These nearest-resized mask planes correspond to original axis-1 indices
98,100,...,156 within the supplied 60-slice band [98,157]. This differs
from scoring all 60 native-resolution 2D slices. Actual 2D predictions
must first be mapped to this grid and their GT checked for exact equality.

The full/shared 3D prostate counts happen to agree in this archived run.
That does not justify copying complete-volume HD95 or invasion counts
into the shared-band table. Those metrics require their own spatial
calculation. Labels on crop boundaries can affect surface distances.

### C. Six-class region check (3D only)

| Class | Full-volume mean scan Dice | Shared-band mean scan Dice |
| --- | ---: | ---: |
| Background | 0.9939 | 0.9742 |
| Body | 0.9708 | 0.9663 |
| Bones | 0.8748 | 0.8882 |
| Bladder | 0.9146 | 0.9212 |
| Rectum | 0.8197 | 0.8159 |
| Prostate | 0.7732 | 0.7732 |

### D. Paired-case template on the shared grid

| Test case | 3D shared-band prostate Dice | 2D shared-band Dice | Improved 2D (optional) |
| --- | ---: | --- | --- |
| B037_Week0 | 0.6413 | Pending | Pending |
| L011_Week2 | 0.7088 | Pending | Pending |
| H017_Week3 | 0.7215 | Pending | Pending |
| K042_Week0 | 0.8525 | Pending | Pending |
| H007_Week0 | 0.7888 | Pending | Pending |

Reuse these fixed case identities for the 2D overlays; include all 27
scans in numerical comparisons rather than evaluating only these examples.

### E. Resource profile (separate from accuracy scope)

| Model | Parameters | Precision / hardware | Forward latency | Peak inference allocation |
| --- | ---: | --- | --- | --- |
| Standard 3D U-Net | 351,206 | BF16 / NVIDIA A100-PCIE-40GB | 17.55 ms median per volume | 0.1364 GiB |
| Standard 2D U-Net | Pending checkpoint configuration verification | Pending measurement | Pending full-scan measurement | Pending |
| Improved 2D U-Net (optional) | Pending implementation | Pending | Pending | Pending |

The existing 3D timing uses the full input volume, excludes data loading,
transfer, argmax, metrics and plotting, and is not training memory/time.
A 2D per-slice time is not comparable with a 3D per-volume time. Report
complete-scan 2D timing and use matched hardware/precision for architecture
claims; otherwise label the hardware difference. Shared-band accuracy does
not mean the 3D network was run on a cropped input.

### F. Native-resolution 2D results (not the shared-grid comparison)

| Model / scope | Test patients/scans/slices | Mean scan prostate Dice | Mean IoU |
| --- | --- | --- | --- |
| Standard 2D U-Net; all 60 original unpadded slices per scan | Required: 6 / 27 / 1620 | Pending | Pending |
| Improved 2D U-Net (optional); same native band | Required: 6 / 27 / 1620 | Pending | Pending |

This is the native 2D task, grouped by scan. Its figures and metrics use
original valid pixels (256×128 or 256×144); padding label 255 is ignored.
These results are not substituted into table B without grid alignment.

Protocol and missing-result rules: [COMPARISON_PROTOCOL.md](../COMPARISON_PROTOCOL.md).
Auditable tables: [comparison/metrics.json](metrics.json) and
[comparison/metrics.csv](metrics.csv). Regenerate with
`python prepare_comparison.py`; this only reads archived local records.
