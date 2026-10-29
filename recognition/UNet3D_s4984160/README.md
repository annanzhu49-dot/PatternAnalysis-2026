# Standard 3D U-Net for HipMRI Segmentation

COMP3710 · Normal difficulty · Student ID: s4984160 · Updated 30 October 2026

## Abstract

This project implements a standard 3D U-Net for six-class pelvic MRI segmentation,
with prostate segmentation as the principal endpoint. A patient-disjoint split
of 211 scans from 38 patients provided 145 training, 39 validation and 27 test
scans. The validation-selected 3D checkpoint achieved test mean scan prostate
Dice **0.7732** (SD 0.0557), mean IoU **0.6336** and mean HD95 **6.50 mm** on
the resized full-volume grid. An independently trained standard 2D U-Net used
the same patient groups. After verifying exact ground-truth equality on a
shared grid, 2D achieved Dice **0.8232**, compared with **0.7732** for 3D, but
produced more prostate voxels inside the reference rectum (**400 versus 21**).
Matched-GPU measurements and five fixed paired cases show that mean overlap,
organ confusion and computational cost must be considered together. This is
a comparison of two implemented pipelines, not a controlled dimensionality
ablation or evidence of clinical readiness.

## 1. Problem, scope and rationale

The task is voxel-wise classification of Background, Body, Bones, Bladder,
Rectum and Prostate (numeric labels 0–5). The prostate is small relative to
the volume, making foreground imbalance and missed or excessive boundaries
important. Dice measures overlap, but can obscure localized extent errors
and confusion with neighbouring organs. The report therefore combines
per-class Dice/IoU with prostate HD95, volume ratio, precision/recall and
erroneous overlap with bladder and rectum.

The encoder–decoder and skip connections follow the U-Net design [1]; the
volumetric form follows the 3D U-Net family [2]. The implemented 3D model is
a compact standard U-Net with instance normalization, rather than an exact
reproduction of every configuration in the original paper. No residual blocks,
attention, deep supervision or improved U-Net are claimed. The 2D model is an
independently trained comparator, not a pretrained component of the 3D model.
The adopted task-level mean prostate Dice targets are 0.70 for 3D and 0.75 for
the native 2D task; these thresholds are not clinical acceptance criteria.

## 2. Feasibility review and final scope

This compact review records the scope supported by the completed experiment;
it is a retrospective update, not a claim of a separately approved feasibility
check. The initial improved-3D ambition was narrowed to Normal standard 3D
U-Net so that training, independent testing, prediction and an interpretable
comparison could be completed within the available cluster access and deadline.

**Data feasibility.** Paired course MRI and semantic labels were available.
Patient IDs support a disjoint split, preventing different weeks from the
same patient appearing in training and test. The supplied 2D data cover only
a coronal band, so comparing native 2D scores against full-volume 3D scores
would be inappropriate. A verified shared-grid evaluation resolves the spatial
comparison while retaining each model's native task results separately.

**Compute feasibility.** Full-resolution 3D training is costly. Resizing to
64×128×128, using base width 8, batch size one, RAM caching and BF16 allowed
the standard model to run on a course A100. Small-foreground learning remained
difficult: the first 15 epochs had zero validation prostate Dice. Continuing
from the last checkpoint with training-derived class weights recovered a useful
baseline. Queue time was an operational constraint, distinct from model runtime.

**Risk and decision criteria.** The principal risks were class imbalance,
patient leakage, spatial misalignment and stopping at an apparently good mean
without inspecting failures. Mitigations were a fixed patient split, validation
checkpoint selection, strict loading/configuration checks, exact shared-GT
verification and retained failure cases. The completed test means meet the
adopted numerical targets, but small patient count, low-resolution geometry and
remaining organ confusion limit generalization. The achievable deliverable is
a reproducible research baseline and qualified comparison; improved architectures
and clinical validation remain outside the implemented scope.

## 3. Data and preprocessing — `dataset.py`

| Split | Patients | Scans | Supplied 2D slices |
| --- | ---: | ---: | ---: |
| Training | 26 | 145 | 8,700 |
| Validation | 6 | 39 | 2,340 |
| Test | 6 | 27 | 1,620 |

Original prefixed patient IDs were sorted, shuffled with `random.Random(42)`,
then partitioned 26/6/6. All weeks from one patient stay in one group. The test
patients are B037, H007, H017, K042, L011 and M013. The 2D manifest maps the
numeric IDs back to the identical groups. The archived manifests, rather than
an independently randomized slice split, define the evaluation cohort.

For 3D, `semantic_MRs/` and `semantic_labels_only/` are paired by filename.
Shapes and affines are checked. Of 211 volumes, 210 have shape (256,256,128)
and one has shape (256,256,144). MRI nonzero voxels are normalized per volume
by Z-score, with zero background retained. Axes change from (X,Y,Z) to
(D,H,W)=(Z,X,Y); MRI is trilinearly resized (`align_corners=False`) and labels
nearest-resized to (64,128,128). Images have one input channel; labels remain
integers 0–5. Test preprocessing is identical to validation preprocessing.

For 2D, each scan supplies 60 coronal slices, corresponding to original axis-1
indices 98–157, with an axis-0 reversal. Each slice is Z-score normalized over
its valid original pixels. Width 128 is right-padded to 144 (MRI value zero,
label 255); native width 144 needs no padding. The loss ignores label 255 and
evaluation removes padding. This slice-level normalization differs from 3D.

### Shared evaluation geometry

The observed mapping is `slice[i] = original_volume[::-1,98+i,:]`. Predictions
are unpadded and flipped back before nearest-grid alignment. The common grid
has D=64, H=128 and 30 planes, corresponding to full-grid W=49–78 and original
Y=98,100,…,156. It is not all 60 native planes. All 27 mapped GT arrays were
checked for exact voxel equality; re-inferred 3D overlap counts reproduced
the original archive. Shared HD95 and organ confusion were independently
recomputed. The 3D model still takes its full-volume input, so this is matched
output-region evaluation, not matched input context.

## 4. Architectures — `modules.py`

| Property | Standard 3D U-Net | Standard 2D U-Net |
| --- | --- | --- |
| Input | One full resized volume or training block | One normalized padded coronal slice |
| Encoder/bottleneck channels | 8, 16, 32, 64 | 16, 32, 64, 128, 256 |
| Pooling levels | 3 | 4 |
| Normalization | Affine InstanceNorm3d | BatchNorm2d |
| Parameters | 351,206 | 1,942,374 |
| Output | Six-class logits | Six-class logits |

Both use double convolution blocks, ReLU, max pooling, transposed-convolution
upsampling and concatenated skip connections. Each 3D block uses 3×3×3
convolutions. For a full input, spatial sizes progress from 64×128×128 to
8×16×16 at the bottleneck, then return to the input size. Although the component
definition defaults to BatchNorm3d, `support/baseline_model.py` builds the actual
trained model with `InstanceNorm3d(affine=True, track_running_stats=False)`.
Evaluation reconstructs this configuration from the checkpoint. Prediction
uses argmax without postprocessing. The different widths, depths, normalization
and parameter counts are explicit comparison confounders.

## 5. Training and model selection — `train.py`

### 5.1 Standard 3D baseline

The initial stage is `support/training/scratch.py`; root `train.py` is the final
refinement stage. The first stage stopped after 15 epochs under its validation
plateau rule. Prostate validation Dice was zero, although other classes had
learned. Refinement loaded epoch-15 **last_model.pt**, reset the optimizer,
and trained a further 11 epochs. It did not restore the complete optimizer/RNG
state of a paused run, and it did not initialize from the test set.

| Setting | Recorded configuration |
| --- | --- |
| Optimizer | AdamW, learning rate 0.0005, weight decay 1e-5 |
| Batch / precision | 1 / CUDA BF16 autocast |
| Sampling probabilities | 50% full volume, 25% rectum-centred block, 25% prostate-centred block |
| Block size | 32×64×64; random centre offset; full-volume fallback if target absent |
| Data execution | Preprocessed RAM cache, one thread, zero extra workers |
| Objective | 0.5 weighted cross-entropy + 0.5 weighted foreground soft-Dice loss |
| Foreground Dice weights, classes 1–5 | [1,1,1,2,2] |
| Checkpoint selection | Highest validation mean scan prostate Dice; foreground mean breaks ties |

Sampling probabilities describe random choices, not exact per-epoch quotas.
Initial CE weights were [1,1,1,2,3,3]. Refinement used training-only counts:
`sqrt(body_count/class_count)` clamped to [0.25,25], yielding approximately
[0.7792,1.0000,3.3000,8.1902,16.0901,18.8062]. Class weights address imbalance,
while full-volume examples maintain exposure to background and organ context.
This rationale is not evidence from a controlled sampling ablation.

Refinement validation prostate Dice was zero for its first five epochs, then
0.4395, 0.6710, 0.3671, 0.7415, 0.7029 and 0.7560. The last three met the
consecutive ≥0.70 stopping rule with nonzero rectum Dice. Epoch 11 (cumulative
epoch 26) was selected with validation Dice **0.755972**. The configured later
scheduler/plateau rules did not activate. Improvements cannot be attributed
solely to class weights: further optimization, optimizer reset and the loss
change occurred together.

### 5.2 Independently trained 2D comparator

The 2D model used Adam (learning rate 0.001), batch size 16 and an equal mixture
of valid-pixel CE and mean foreground soft-Dice loss. Ignored padding is masked
in both terms. The scheduler halves learning rate after four stale validation
epochs; early stopping uses patience eight and minimum ten epochs, within a
maximum of 30. Seed 42, one thread and zero workers were recorded. Job 716367
ran 13 epochs; epoch 5 had the best validation prostate Dice **0.834130**.
That checkpoint was frozen before independent testing. Training batch soft Dice
and hard-label whole-scan evaluation are different quantities.

### 5.3 3D training evidence

![Existing 3D training loss and validation Dice](support/artifacts/training_3d/training_curves.png)

The plot uses the original 15 initial and 11 refinement epochs. Training
objectives have phase-specific weights, so they are plotted as separate lines.
Validation loss was not logged; the recorded validation metric is prostate
Dice. The selected point is cumulative epoch 26. Source values and hashes
are preserved in [the 3D training record](support/artifacts/training_3d/README.md).

### 5.4 2D baseline training evidence

![2D loss and validation Dice](../UNet2D_s4984160/support/artifacts/training_716367/training_plot.png)

The baseline stopped after 13 epochs without improving on epoch 5 (validation
prostate Dice 0.8341). Its held-out native score was subsequently 0.8254.
Training loss continued falling while validation performance plateaued, so
extra training was not assumed to help. The test set was not used to select
the epoch. Original training records are preserved in the
[2D training record](../UNet2D_s4984160/support/artifacts/training_716367/README.md).


## 6. Independent evaluation — `evaluate.py`

Frozen test evaluation used 27 scans from six unseen patients. Job 715670
evaluated 3D, job 716656 evaluated 2D and re-inferred the paired 3D reference,
and job 717015 measured resources and demonstrated direct 3D inference.
The evaluator strictly checks model configuration, checkpoint SHA256, patient
manifests and geometry. No weights, thresholds or postprocessing were selected
using test results. Aggregate success does not make each prediction correct.

For each class and scan, Dice = `2TP/(2TP+FP+FN)` and IoU =
`TP/(TP+FP+FN)`. The primary result is mean scan Dice; SD describes variation
between scans, not a confidence interval. Patient Dice first averages scans
within each patient, then averages patients equally. Pooled precision, recall
and volume ratio use summed voxel counts; these are not means of per-case
ratios. Both-empty class masks are undefined and excluded with valid counts,
rather than assigned perfect scores; one-sided empty overlap is zero.

Prostate HD95 is the 95th percentile of concatenated bidirectional six-connected
surface distances, averaged across valid cases. An empty mask makes HD95
undefined, with an explicit status. All 27 full-volume 3D cases had valid values.
Distances use original affine spacing adjusted for resize and are approximate
resized-grid millimetres, not clinical original-resolution contour accuracy.
Native 2D distances use the corresponding 3D affine, not identity 2D affines.

Organ invasion counts prostate predictions whose GT class is bladder or rectum.
They cover these specific confusion regions, not all false positives. Counts
are comparable between models only on the same grid. The predicted/GT volume
ratio exceeds one for excess total volume, but cannot localize errors.

The 3D full-volume prostate totals were TP=19,690, FP=7,859 and FN=3,706;
pooled Dice was 0.772990. Only 110 of its 7,859 false-positive voxels lay in GT
bladder/rectum (1.40%), so most excess prediction cannot be described as
invasion of those two organs. Three of 27 3D cases scored below 0.70, with
scan Dice ranging from 0.6413 to 0.8850. The following tables preserve native,
shared and complete-volume scopes separately.

<!-- BEGIN GENERATED COMPARISON -->
## Model comparison: complete volume and shared band

Frozen held-out evaluation job **716656** completed. All 27 shared GT grids
matched voxel-for-voxel, and re-inferred 3D overlap counts reproduced the archive.
The 2D native task and paired shared-grid task remain separate below.
Only the two implemented standard models are compared.

### A. Complete-volume 3D reference

| Scope | Patients/scans | Mean scan Dice ± SD | Mean IoU | HD95 (approx. mm) | Pooled volume ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard 3D; full 64×128×128 grid | 6 / 27 | 0.7732 ± 0.0557 | 0.6336 | 6.50 | 1.1775 |

### B. Paired shared-grid prostate results

| Model | Mean scan Dice ± SD | Mean patient Dice | Mean IoU | HD95 (approx. mm) | Pooled precision / recall | Pred/GT ratio | FP in bladder / rectum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard 3D U-Net | 0.7732 ± 0.0557 | 0.7678 | 0.6336 | 6.50 | 0.7147 / 0.8416 | 1.1775 | 89 / 21 |
| Standard 2D U-Net | 0.8232 ± 0.0645 | 0.7877 | 0.7044 | 5.18 | 0.8421 / 0.8093 | 0.9611 | 110 / 400 |

2D has higher Dice in **21/27 scans**; its mean scan gain is
**5.00 percentage points**. The gain falls to
**1.98 points** when weighting the six patients equally;
4/6 patients favour 2D. Multiple weeks from one patient are correlated,
so 27 scans are not 27 independent patients. These are descriptive results,
without a statistical superiority claim.

Shared scope: D=64, H=128, original full-grid W=49..78 (30 planes),
corresponding to original Y=98,100,...,156. Shared HD95 and invasion counts
were computed independently in this paired job, not copied from full-volume
3D metrics. One 2D prediction (H007 Week0) touches the cropped W boundary;
no GT or 3D prostate does. Distances refer to approximate resized spacing
and can be affected by ROI surface truncation.

### C. Six-class scope check

| Class | 3D full Dice / IoU | 3D shared Dice / IoU | 2D shared Dice / IoU |
| --- | ---: | ---: | ---: |
| Background | 0.9939 / 0.9878 | 0.9742 / 0.9499 | 0.9966 / 0.9932 |
| Body | 0.9708 / 0.9434 | 0.9663 / 0.9348 | 0.9837 / 0.9679 |
| Bones | 0.8748 / 0.7781 | 0.8882 / 0.7994 | 0.9314 / 0.8717 |
| Bladder | 0.9146 / 0.8432 | 0.9212 / 0.8544 | 0.9668 / 0.9359 |
| Rectum | 0.8197 / 0.6972 | 0.8159 / 0.6928 | 0.8708 / 0.7734 |
| Prostate | 0.7732 / 0.6336 | 0.7732 / 0.6336 | 0.8232 / 0.7044 |

### D. Fixed-case shared-grid comparison

| Case | 3D Dice | 2D Dice | 2D minus 3D |
| --- | ---: | ---: | ---: |
| B037_Week0 | 0.6413 | 0.7195 | +0.0782 |
| L011_Week2 | 0.7088 | 0.8458 | +0.1370 |
| H017_Week3 | 0.7215 | 0.8158 | +0.0942 |
| K042_Week0 | 0.8525 | 0.8378 | -0.0147 |
| H007_Week0 | 0.7888 | 0.6686 | -0.1202 |

The paired overlays are in the [completed 2D result record](../UNet2D_s4984160/support/artifacts/final_test_2d/README.md).
They reuse the same five cases and positions, without selecting new favourable examples.

### E. Resource profile and confounders

| Model | Parameters | Precision | Forward ms/case | Prediction ms/case | Peak allocated / reserved GiB |
| --- | ---: | --- | ---: | ---: | ---: |
| Standard 2D | 1,942,374 | FP32 | 18.63 | 21.75 | 0.2710 / 0.3418 |
| Standard 3D | 351,206 | FP32 | 17.05 | 18.24 | 0.1815 / 0.2402 |
| Standard 3D | 351,206 | BF16 | 9.75 | 10.91 | 0.1246 / 0.1582 |

Job **717015** measured all rows separately on the same **NVIDIA A100-PCIE-40GB**;
each model started with zero other PyTorch GPU allocations. TF32 was disabled.
Times are medians across 27 per-case medians, with three warmups and three repeats.
2D means all 60 native slices (batch 16); 3D means the full 64×128×128 volume (batch 1).
Forward excludes transfers; prediction includes transfers, forward, argmax and output copy.
Both exclude disk loading, preprocessing and initial model loading. Peaks include resident
weights and inference tensors/cache, but exclude CUDA driver/context and training memory.
Allocated and reserved are distinct PyTorch memory measures.

Use FP32 rows for the matched-precision comparison; BF16 records the 3D deployment mode.
The earlier 3D-only timing (17.55 ms, 0.1364 GiB) remains in the original archive;
this table uses the new isolated profiling protocol, without replacing segmentation scores.
Coverage, resolution, normalization, model size and training differ, so these are pipeline
costs, not an isolated causal test of convolution dimension.

Raw [resource records and prediction example](../UNet2D_s4984160/support/artifacts/resource_717015/README.md).

### F. Native 2D task (all 60 unpadded slices per scan)

| Model | Patients/scans/slices | Mean scan Dice ± SD | Mean patient Dice | Mean IoU | Native-band HD95 (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard 2D | 6 / 27 / 1620 | 0.8254 ± 0.0640 | 0.7909 | 0.7075 | 4.90 |

The native 2D mean exceeds the 0.75 target; 4/27 scans remain below 0.75.
Native metrics use the original valid pixels and original 3D affine spacing
for the supplied partial band. They are not full-volume 3D scores and must
not be substituted into the shared-grid table.

### Interpretation

2D improves mean overlap, precision and mean HD95, with less total excess
volume. However, shared-grid rectal invasion is **400 voxels versus 21**
for 3D, and bladder invasion is 110 versus 89. 3D has higher pooled recall
(0.8416 versus 0.8093). H007 Week0 is an important 2D failure, not hidden
by the better average. Keep both models frozen; use the remaining errors
for reporting and future validation experiments, not test-set tuning.

Rules: [comparison protocol](support/docs/comparison_protocol.md).
Auditable outputs: [metrics.json](support/comparison/metrics.json),
[metrics.csv](support/comparison/metrics.csv), and
[patient_comparison.csv](support/comparison/patient_comparison.csv).
Regenerate with `python -m support.prepare_comparison` (local records only).
<!-- END GENERATED COMPARISON -->

## Qualitative error analysis: five held-out cases

Columns are MRI, GT, 3D prediction, 2D prediction, and 2D prostate TP/FP/FN.
Green/red/blue are TP/FP/FN; cyan/orange/magenta are bladder/rectum/prostate.
Gray shading is outside the evaluated band, not predicted background. Extent
rows are GT-based proxies, not independently annotated clinical landmarks.
Case identities and view positions were fixed from the earlier 3D review.

### B037 Week0: improved average overlap, but superior omission remains

![B037 paired predictions](../UNet2D_s4984160/support/artifacts/final_test_2d/figures/B037_Week0_LFOV.png)

Shared Dice rises from 0.6413 to 0.7195 and HD95 falls from 14.65 to 12.49 mm.
The 2D middle-plane core has green overlap with visible blue missed edges;
the superior D=39 GT focus remains blue/missed. Volume ratios are 0.6481 for
2D and 0.6010 for 3D: both undersegment. A better average has not repaired this
specific extent failure.

### L011 Week2: less excess prediction, residual superior error

![L011 paired predictions](../UNet2D_s4984160/support/artifacts/final_test_2d/figures/L011_Week2_LFOV.png)

Shared Dice improves from 0.7088 to 0.8458. The middle-plane 2D contour is less
expanded than 3D, consistent with the volume ratio falling from 1.6787 to
1.1880. HD95 improves from 8.13 to 4.71 mm; bladder invasion falls from 32 to
3 voxels. The small superior D=44 focus is still missed, so improvement is
not uniform across boundaries.

### H017 Week3: better precision with remaining undercoverage

![H017 paired predictions](../UNet2D_s4984160/support/artifacts/final_test_2d/figures/H017_Week3_LFOV.png)

Shared Dice improves from 0.7215 to 0.8158 and HD95 from 6.72 to 3.36 mm.
The 2D prediction is smaller overall (volume ratio 0.8130), but its precision
is higher (0.9096 versus 0.7246). Core overlap improves while blue errors
remain at small extents. Bladder invasion is zero for 2D versus ten for 3D;
rectal invasion is eight versus zero. Volume agreement alone does not rank
the models, and improved overlap does not remove every organ-confusion error.

### K042 Week0: 3D retains a small Dice advantage

![K042 paired predictions](../UNet2D_s4984160/support/artifacts/final_test_2d/figures/K042_Week0_LFOV.png)

Shared Dice is 0.8378 for 2D versus 0.8525 for 3D. 2D has lower recall
(0.7607 versus 0.8106) and a smaller volume ratio (0.8158 versus 0.9017),
consistent with undercoverage despite a mostly matching core. Its HD95 is
slightly lower (5.68 versus 6.26 mm), illustrating that Dice and HD95 need
not produce the same ranking.

### H007 Week0: important 2D failure and rectal confusion

![H007 paired predictions](../UNet2D_s4984160/support/artifacts/final_test_2d/figures/H007_Week0_LFOV.png)

Shared Dice falls from 0.7888 for 3D to 0.6686 for 2D; HD95 worsens from 4.47
to 9.06 mm. The middle-plane 2D error map has red extension toward the GT
rectal region, while blue extent misses persist. Whole shared-volume counts
confirm **191 rectal-invasion voxels for 2D versus five for 3D**; these counts
are not limited to the displayed slice. This one case supplies 47.75% of all
2D rectal-invasion voxels. It also touches the shared ROI boundary. This
failure is retained rather than hidden by the better 2D average.


The original detailed 3D-only review remains in [support/docs/case_analysis.md](support/docs/case_analysis.md). The paired review uses fixed cases and does not replace all-scan numerical evaluation.


## 7. Discussion, limitations and engineering recommendation

Both implemented standard models meet their adopted mean-Dice targets. On the
verified shared grid, 2D improves mean overlap and precision, while 3D retains
higher pooled recall and much less rectal invasion. These observations do not
support the claim that 3D must outperform 2D. The 2D model has about 5.5 times
as many parameters, different normalization and higher native in-plane detail;
the experiment does not isolate why the ranking differs.

The mean scan Dice difference of 5.00 percentage points falls to 1.98 points
under equal patient weighting. There are only six test patients, with correlated
weekly scans; no confidence interval, statistical superiority test, multiple-seed
study or external validation is claimed. The five examples are deliberately
selected diagnostic cases, not a representative random sample. Selected slices
do not establish absence of errors elsewhere, and no complete connected-component
or clinical annotation audit was performed. GT-derived inferior/superior extent
planes are not independent clinical apex/base landmarks. Downsampling and crop
surfaces also limit boundary interpretation; H007's 2D output touches the ROI edge.

Under this resource protocol the compact 3D model is slightly faster and uses
less allocated memory than 2D at matched FP32. BF16 further reduces measured
3D cost, but numerical precision and warmup/protocol matter; this should not be
presented as a universal speed law. Neither timing includes a complete disk-to-
image workflow, and no training-memory comparison was measured.

For the implemented research workflow, retain the 3D model as the full-volume
baseline and the 2D model as a useful limited-band comparator. Choosing solely
by mean Dice would miss H007's rectal confusion; choosing solely by organ
invasion would overlook 3D's lower precision and excess volume. Both require
review of low-confidence or boundary-sensitive predictions. No clinical-use
recommendation follows from this course experiment.

Future validation experiments should prioritize full-resolution or localized
3D refinement for extent omissions, and balanced negative/adjacent-organ
examples for the 2D rectal-confusion failure. Capacity/normalization-matched
experiments and repeated patient splits would clarify the comparison. These
are proposals, not completed improvements; preserve the current test results
and develop any new decisions using training/validation data and a new final
evaluation protocol.

## 8. Reproducibility and implementation guide

```text
UNet3D_s4984160/
├── dataset.py     Data pairing, patient split and preprocessing
├── modules.py     Standard U-Net components
├── train.py       Recorded refinement stage
├── evaluate.py    Frozen-checkpoint evaluation
├── predict.py     Direct inference or cached rendering
├── README.md      Consolidated experiment report
└── support/       Factories, initial training, scripts, records and history
```

The reading order is **Dataset → Modules → Train → Evaluate → Predict**.
Initial training is invoked as `python -m support.training.scratch`; model
normalization is constructed by `support/baseline_model.py`. The independently
trained 2D implementation is in `../UNet2D_s4984160/`. The cluster used the
existing `comp3710` account/partition, one GPU and a 1.5-hour limit, without
extra CPU or RAM requests. Archived source provenance is retained in
`support/history/source_snapshots.json`; root paths have since been organized.

| Software | Recorded 3D evaluation version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| NumPy | 2.4.6 |
| NiBabel | 5.4.2 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.2 |

3D frozen weight SHA256:
`77ecc9f051d5d1213c2e62474d582050fe5b425a08182e3d2becc73540737156`.

2D epoch-5 frozen weight SHA256:
`3a666e057827897d294d9df9604ac325617fa874bf541402011537847024174b`.

Weights and course data remain on the cluster; they are not bundled in Git.
Original source/configuration, patient manifests, CSV/JSON, hashes and figures
support offline checking. Re-running inference requires access to those exact
weights and private course data. Seeds do not guarantee bit-identical output
across devices or software versions. CPU/synthetic execution checks establish
implementation behaviour, not real-data segmentation quality.

### 8.1 Running the current layout

Requires Python, PyTorch, NumPy, NiBabel, SciPy and Matplotlib. The original
evaluation environment versions are recorded in
[summary.json](support/artifacts/final_test_3d/summary.json). Private course data and
weights are not included. Dataset default:
`/home/groups/comp3710/HipMRI_Study_open`.

From this project directory on the configured cluster:

```bash
sbatch support/scripts/train_scratch.sbatch
# After initial training finishes and its last checkpoint exists:
sbatch support/scripts/train_refine.sbatch
# After the selected refinement checkpoint exists:
sbatch support/scripts/evaluate.sbatch
```

Do not submit the dependent stages concurrently. The Slurm account, partition,
GPU and 1.5-hour requests are unchanged; only Python entry-point paths changed.
These examples are for a new run, not an instruction to restart the completed
experiment. Existing evaluation requires a fresh output directory; see
[evaluation usage](support/docs/evaluation.md).

Useful local commands, from this project directory:

```bash
python -m support.training.scratch --help
python train.py --help
python -m unittest discover -s support/tests -v
python -m support.prepare_comparison
```

### Direct checkpoint prediction example

On a GPU node, from this project directory with the private course data:

```bash
python predict.py \
  --checkpoint results/refine_v2/best_model.pt \
  --mri /home/groups/comp3710/HipMRI_Study_open/semantic_MRs/B037_Week0_LFOV.nii.gz \
  --mask /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/B037_Week0_SEMANTIC.nii.gz \
  --expected-sha256 77ecc9f051d5d1213c2e62474d582050fe5b425a08182e3d2becc73540737156 \
  --device cuda --precision bf16 --threads 1 --output-dir results/direct_example
```

This loads the saved configuration/weights, applies `dataset.py` preprocessing,
runs inference, prints Dice/HD95 and saves a prediction overlay plus resized
D,H,W labels and metadata. GT is used for evaluation/visualization, not input
to the network. A new output directory is required. The original cached-render
command still works. See [prediction and resource usage](support/docs/resource_profile.md).

Job **717015** verified this direct entry point on B037 Week0: Dice **0.6413**,
HD95 **14.65 mm**, and all six classes' voxel counts match the archived test.
This retained failure case shows undercoverage; it does not change the overall
test Dice of 0.7732. The [direct prediction image and metadata](../UNet2D_s4984160/support/artifacts/resource_717015/README.md)
provide the completed execution evidence.

The final checkpoint remains on the cluster at `results/refine_v2/best_model.pt`;
its SHA256 and original metrics are in
[the result record](support/artifacts/final_test_3d/README.md). Historical early code is
under `support/history/`, not the current training entry point. Additional references:
[case analysis in Chinese](support/docs/case_analysis.md),
[comparison protocol](support/docs/comparison_protocol.md), and
[2D/3D data mapping](support/docs/data_mapping.md).


## 9. Artificial Intelligence Usage Disclosure

ChatGPT and Codex assisted with experimental planning, reviewing and revising
the student's draft data/model/training code, writing evaluation and prediction
utilities, debugging, preparing cluster scripts, collecting outputs, checking
spatial alignment and metrics, and drafting/organizing this report. Codex also
performed authorized terminal and Git operations. This assistance was
substantial; the work is not represented as unaided implementation or writing.

A peer's shared discussion was consulted for general 2D workflow ideas. It is
not the source of the trained checkpoints or reported measurements, and the
project does not present the peer's implementation or results as its own.
The independently run project code/configurations and archived cluster outputs
are the evidence for the results reported here.

Numbers in the tables come from archived real training/test outputs, with
aggregate recomputation and checkpoint/cohort consistency checks. Literature
links were checked against the original publications. These checks reduce
transcription risk but do not constitute an independent expert or clinical
audit. AI-assisted interpretation is constrained by saved metrics and displayed
images; no missing experiment, validation-loss curve or clinical endorsement
has been invented. The student remains responsible for reviewing the report,
understanding the implementation and confirming the accuracy of this disclosure.

## 10. References

1. Ronneberger, O., Fischer, P., and Brox, T. (2015). *U-Net: Convolutional
   Networks for Biomedical Image Segmentation*. MICCAI.
   [Original paper](https://arxiv.org/abs/1505.04597).
2. Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., and Ronneberger, O.
   (2016). *3D U-Net: Learning Dense Volumetric Segmentation from Sparse
   Annotation*. MICCAI. [Original paper](https://arxiv.org/abs/1606.06650).
3. COMP3710 course materials (2026). *Report specification v2.4* and the supplied
   HipMRI study MRI/semantic labels and 2D slices. Course-provided data and
   assignment documentation; no independent public dataset citation or external
   data license is asserted here.

## Conclusion

The completed standard 3D U-Net provides a full-volume baseline with held-out
prostate Dice 0.7732. The independently trained 2D comparator has higher shared-
grid mean Dice but greater rectal confusion. Verified spatial alignment,
patient-level reporting, five retained cases and isolated resource measurements
make the trade-off visible. The main outcome is a documented, reproducible
baseline and qualified comparison, with remaining boundary errors and a small
test cohort explicitly limiting the conclusion.
