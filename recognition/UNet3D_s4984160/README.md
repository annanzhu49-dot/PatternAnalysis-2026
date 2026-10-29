# Standard 3D U-Net (Normal difficulty)

Final held-out test prostate mean scan Dice: **0.773216** on 27 scans from
six patients. Validation checkpoint-selection Dice: **0.755972**. See
[the final experiment record](experiment_notes/final_test_refine_v2/README.md)
for original results, limitations, configuration and five prediction figures.

## Read the completed experiment

- [完整实验记录（中文）](EXPERIMENT_REPORT.md): each step mapped to its source
  file, data preprocessing/split, actual architecture, two-stage training,
  six-class results, prostate error analysis and a report-ready English paragraph.
- [Original results and figures](experiment_notes/final_test_refine_v2/README.md):
  archived JSON/CSV, configuration, training history and five PNG examples.
- [Evaluation definitions](EVALUATION_USAGE.md): aggregation, HD95 conventions,
  example selection and inference timing boundaries.
- [Comparison protocol](COMPARISON_PROTOCOL.md): full-volume, native-2D and
  shared-grid scopes; pending entries and exact alignment requirements.

| Prostate metric | Held-out test result |
| --- | ---: |
| Mean scan Dice ± SD | 0.7732 ± 0.0557 |
| Mean patient Dice | 0.7678 |
| Mean HD95, approximate resized-grid mm | 6.50 |
| Pooled precision / recall | 0.7147 / 0.8416 |
| Pooled predicted / GT volume ratio | 1.1775 |
| Prostate FP voxels in GT bladder / rectum | 89 / 21 |
| Scans with Dice below 0.70 | 3 / 27 |

The mean meets the 0.70 target; not every scan does. Predicted volume exceeds
GT by 17.75% in aggregate. Figure generation and review of the five displayed examples are complete;
the final aligned 2D comparison remains pending. These reviews are algorithmic
error analyses, not an independent clinical annotation audit.

## Files used for the final experiment

- `modules.py` and `dataset.py`: standard network and 3D preprocessing/split.
- `baseline_model.py`: checkpoint-configured InstanceNorm/BatchNorm factory.
- `train_fast.py`: initial scratch stage (15 epochs in the recorded run).
- `train_refine.py`: refinement from scratch-stage last weights with a new
  optimizer and training-only class weights (11 epochs in the recorded run).
- `evaluate.py`, `evaluation_utils.py`: frozen-checkpoint held-out metrics.
- `predict.py`: render cached evaluation predictions, without rerunning inference.
- `test_evaluation.py`: synthetic evaluator checks.
- `train_scratch_fast.sbatch`, `train_refine.sbatch`, `evaluate_refine.sbatch`:
  original submitted Slurm scripts, with unchanged resource requests.

The three corresponding frozen source directories are
`baseline_scratch_v1/`, `baseline_refine_v2/` and
`evaluation_refine_v2_final/`. The Slurm scripts refer to these directories.
Root `train.py` and earlier pilot notes remain historical; they are not the
training entry point for the reported final checkpoint.

## Running

Requires Python, PyTorch, NumPy, NiBabel, SciPy and Matplotlib. Exact measured
environment versions are in the archived `summary.json`. Dataset default:
`/home/groups/comp3710/HipMRI_Study_open`. Training/evaluation require the
private course data. Checkpoints and raw data are deliberately excluded from Git.

From this directory on the configured cluster, submit the scratch stage with
`sbatch train_scratch_fast.sbatch`. After it completes and its last checkpoint
exists, submit refinement with `sbatch train_refine.sbatch`. Evaluation uses
`sbatch evaluate_refine.sbatch` after the refinement checkpoint is available.
Do not submit all stages concurrently; they depend on preceding outputs.
These scripts retain the original account, partition, environment and working
directory and are intended for the recorded cluster setup.

The final checkpoint remains at `results/refine_v2/best_model.pt` on the
cluster; its SHA256 is recorded in the archived result. Evaluation requires
a fresh output directory to avoid overwriting a previous run. See
[EVALUATION_USAGE.md](EVALUATION_USAGE.md) for metric conventions and outputs.
No retraining was performed when archiving these results.

The current model uses a 64x128x128 resized grid, base width 8 and
InstanceNorm3d, with no residual blocks or deep supervision. The supplied 2D
data only cover a partial coronal band; comparisons must align patient split,
spatial coverage and metric aggregation. Clinical-resolution distance accuracy
is not established by resized-grid metrics.

Code review, training adaptations, evaluation and documentation used AI
assistance; retain the full project AI-use disclosure in the final report.

<!-- BEGIN GENERATED COMPARISON -->
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

Protocol and missing-result rules: [COMPARISON_PROTOCOL.md](COMPARISON_PROTOCOL.md).
Auditable tables: [comparison/metrics.json](comparison/metrics.json) and
[comparison/metrics.csv](comparison/metrics.csv). Regenerate with
`python prepare_comparison.py`; this only reads archived local records.
<!-- END GENERATED COMPARISON -->

## Qualitative error analysis: five held-out cases

These examples use the frozen checkpoint from the reported 3D test run. The
figure titles and metrics were checked against the archived full-volume CSV;
GT-derived boundary-region counts were checked separately. No training,
postprocessing or test-driven threshold selection was performed for this review.

Green/red/blue denote prostate TP/FP/FN. Cyan/orange/magenta denote bladder,
rectum/prostate in the organ overlays. The first three rows show one inferior
GT extent, middle and superior GT extent axial plane; the fourth shows a fixed
coronal plane. Full-volume metrics must not be interpreted as single-plane
metrics. Boundary CSV scores below pool the first/last two GT-bearing planes,
whereas each axial figure row shows one plane. Extents are proxies, not
independently annotated clinical apex/base landmarks. HD95 is approximate
millimetres in the resized grid; counts refer to that grid.

Selection was deterministic and favoured different patients. Strategy names
such as highest HD95 or largest missed fraction are selection priorities, not
claims that every selected case is a global worst case. K042 illustrates a
good overall result with local failure, and H007 illustrates a middle-score case.

| Case | Dice | HD95 (approx. mm) | Pred/GT ratio | Precision | Recall | FP in bladder/rectum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B037_Week0 | 0.6413 | 14.65 | 0.6010 | 0.8542 | 0.5134 | 0 / 0 |
| L011_Week2 | 0.7088 | 8.13 | 1.6787 | 0.5655 | 0.9493 | 32 / 0 |
| H017_Week3 | 0.7215 | 6.72 | 0.9916 | 0.7246 | 0.7185 | 10 / 0 |
| K042_Week0 | 0.8525 | 6.26 | 0.9017 | 0.8990 | 0.8106 | 0 / 0 |
| H007_Week0 | 0.7888 | 4.47 | 1.0833 | 0.7585 | 0.8217 | 0 / 5 |

### B037 Week0: undersegmentation and missing superior extent

![B037 Week0: undersegmentation and missing superior extent](experiment_notes/final_test_refine_v2/B037_Week0_LFOV.png)

The middle axial plane (D=30) retains a matching core but shows a visible blue missed border. At D=39 the GT prostate region is missed; the coronal view also shows incomplete extent coverage. Whole-volume prediction contains 2,003 voxels against 3,333 GT voxels (ratio 0.6010), with 1,622 FN and recall 0.5134. In the superior two-plane region D=38,39, all 87 GT voxels are missed and regional Dice is zero. This supports undercoverage as the dominant failure, consistent with Dice 0.6413 and HD95 14.65 mm. Neither bladder nor rectum invasion explains it: both counts are zero.

### L011 Week2: oversegmentation with local boundary misses

![L011 Week2: oversegmentation with local boundary misses](experiment_notes/final_test_refine_v2/L011_Week2_LFOV.png)

A red rim surrounds the matching core at D=38, while the tiny GT region at D=44 has blue missed voxels. Whole-volume predicted volume is 1.6787 times GT, with precision 0.5655, recall 0.9493 and 547 FP. Of these FP, 32 overlap GT bladder. The superior two-plane region D=43,44 contains TP=3, FP=47 and FN=19 (Dice 0.0833), showing that high overall recall does not guarantee correct local boundaries. The red and blue errors coexist across the two-plane region; its counts do not describe only the displayed D=44 plane. This is not the global highest-HD95 scan: the selection favours a new patient after B037.

### H017 Week3: near-correct volume with spatial mismatch

![H017 Week3: near-correct volume with spatial mismatch](experiment_notes/final_test_refine_v2/H017_Week3_LFOV.png)

The D=33 error map shows both red and blue border discrepancies despite a matching centre. Predicted and GT volumes are nearly equal (472 versus 476 voxels; ratio 0.9916), but FP=130 and FN=134 yield Dice 0.7215. Ten predicted prostate voxels overlap GT bladder across the entire volume; that count cannot be assigned to every visible red border. Volume agreement alone therefore hides contour or position errors. These views do not establish a rigid displacement or prove an imaging cause.

### K042 Week0: good overall overlap, incomplete inferior extent

![K042 Week0: good overall overlap, incomplete inferior extent](experiment_notes/final_test_refine_v2/K042_Week0_LFOV.png)

The middle axial and coronal views show predominantly green overlap, consistent with the relatively high full-volume Dice of 0.8525. The inferior D=26 plane still has blue misses, while the superior view has local red extension. In D=26,27, TP=31 and FN=37 give regional recall 0.4559 and Dice 0.5688. The superior two-plane volume ratio is 1.2596 despite the whole-volume ratio being 0.9017. A high global Dice can therefore conceal errors in small extent regions and even opposite local volume biases.

### H007 Week0: a middle-score case with small-extent failure

![H007 Week0: a middle-score case with small-extent failure](experiment_notes/final_test_refine_v2/H007_Week0_LFOV.png)

The middle plane (D=30) and coronal view have a matching core, but the inferior D=24 GT focus is missed and the superior D=36 view also shows blue error. The overall Dice is 0.7888 with volume ratio 1.0833. Across D=24,25, only 3 of 28 GT voxels are recovered (recall 0.1071; Dice 0.1714). Superior two-plane Dice is 0.6525. Five predicted prostate voxels overlap GT rectum somewhere in the full volume; the remaining FP are not all rectal invasion. Thus extent omission persists beyond the worst-scoring scan.

### Interpretation and engineering recommendation

The cases distinguish undercoverage, excess volume, and offsetting FP/FN
with nearly correct volume. Good overlap in the gland core or a passing overall
Dice does not ensure accurate small extents. Downsampling and local tissue
ambiguity are possible contributors, but no resolution ablation, contrast
measurement or independent landmark annotation establishes them as root causes.
These test observations should motivate future validation experiments, not
tuning on the held-out test set.

For this prototype, manual review should concentrate on extent coverage,
large volume discrepancies and adjacent-organ borders. No automatic-release
policy or clinical acceptance threshold was calibrated. The four selected
planes do not inspect all voxels; no connected-component audit was performed,
so absence of small disconnected artifacts is not established.

The checked numerical evidence and PNG hashes are in
[`case_review_metrics.csv`](experiment_notes/final_test_refine_v2/case_review_metrics.csv).
A more detailed Chinese explanation is in [CASE_ANALYSIS.md](CASE_ANALYSIS.md).

## Final report assembly and submission status

Course specification v2.4, Sections 4, 4.1 and 5.3, requires the report in the
project-specific `README.md`, with a PDF export for Turnitin and a GitHub PR
against upstream `topic-recognition`. This file is the report entry point for
this project; the repository-root README is the shared course introduction.

This remains a report in progress. The complete method notes in
[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) should be consolidated into the
final report body, together with the original one-page feasibility review,
final aligned 2D/3D comparison, literature references and a complete subsection
entitled `Artificial Intelligence Usage Disclosure`. Final code entry points
also need to be aligned with the specification: current root `train.py` is
historical, and current `predict.py` renders cached predictions after
`evaluate.py` performs checkpoint loading/inference. A README export should
include the actual report text and figures rather than rely solely on links.
No final PDF export or PR creation is claimed by this documentation update.
