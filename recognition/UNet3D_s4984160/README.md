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
GT by 17.75% in aggregate. Figure generation is complete, but detailed visual
case interpretation and the final aligned 2D comparison remain to be completed.

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
