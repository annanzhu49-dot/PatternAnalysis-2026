"""Verify archived 3D scores and prepare comparison tables without inference.

Only reads local, archived test results. Missing 2D results stay null. This
does not load a checkpoint, submit a job, or rank incomparable evaluations.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'support/artifacts/final_test_3d'
OUTPUT = ROOT / 'support/comparison'
START = '<!-- BEGIN GENERATED COMPARISON -->'
END = '<!-- END GENERATED COMPARISON -->'


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def score_rows(rows):
    """Recompute case/patient/pooled scores from archived integer counts."""
    result = []
    for label in range(6):
        group = [r for r in rows if int(r['class_id']) == label]
        keys = [r['case'] for r in group]
        if not group or len(set(keys)) != len(keys):
            raise ValueError(f'Missing/duplicate class {label} rows')
        totals = {k: sum(int(r[k]) for r in group)
                  for k in ('tp', 'fp', 'fn', 'gt_voxels', 'pred_voxels')}
        dice, iou, by_patient = [], [], {}
        for row in group:
            tp, fp, fn = (int(row[k]) for k in ('tp', 'fp', 'fn'))
            if int(row['gt_voxels']) != tp + fn or int(row['pred_voxels']) != tp + fp:
                raise ValueError('Inconsistent archived counts')
            d = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
            j = tp / (tp + fp + fn) if tp + fp + fn else None
            for name, value in [('dice', d), ('iou', j)]:
                stored = float(row[name]) if row[name] else None
                if (stored is None) != (value is None) or (
                        value is not None and not math.isclose(value, stored, abs_tol=1e-12)):
                    raise ValueError(f'{row["case"]}: {name} disagrees with counts')
            if d is not None:
                dice.append(d)
                by_patient.setdefault(row['patient_id'], []).append(d)
            if j is not None:
                iou.append(j)
        tp, fp, fn = (totals[k] for k in ('tp', 'fp', 'fn'))
        result.append(dict(class_id=label, class_name=group[0]['class_name'],
                           **totals, mean_case_dice=mean(dice),
                           std_case_dice=statistics.pstdev(dice) if dice else None,
                           mean_case_iou=mean(iou), valid_dice_cases=len(dice),
                           mean_patient_dice=mean([mean(v) for v in by_patient.values()]),
                           pooled_dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                           pooled_precision=tp/(tp+fp) if tp+fp else None,
                           pooled_recall=tp/(tp+fn) if tp+fn else None,
                           pooled_volume_ratio=(tp+fp)/(tp+fn) if tp+fn else None))
    return result


def verify_summary(observed, expected):
    for actual, frozen in zip(observed, expected):
        for key, value in actual.items():
            if isinstance(value, float):
                if not math.isclose(value, frozen[key], abs_tol=1e-12):
                    raise ValueError(f'Summary mismatch: class {actual["class_id"]}, {key}')
            elif value != frozen[key]:
                raise ValueError(f'Summary mismatch: {key}')


def main():
    summary = json.loads((ARCHIVE / 'summary.json').read_text(encoding='utf-8'))
    full_rows = read_csv(ARCHIVE / 'per_case_metrics.csv')
    shared_rows = read_csv(ARCHIVE / 'shared_2d_region_metrics.csv')
    full, shared = score_rows(full_rows), score_rows(shared_rows)
    verify_summary(full, summary['class_metrics'])
    verify_summary(shared, summary['shared_2d_region']['class_metrics'])
    keys = {(r['case'], r['patient_id'], r['class_id']) for r in full_rows}
    if keys != {(r['case'], r['patient_id'], r['class_id']) for r in shared_rows}:
        raise ValueError('Full/shared scan identities differ')
    if len(keys) != 27 * 6 or len({r['patient_id'] for r in full_rows}) != 6:
        raise ValueError('Unexpected test cohort')
    # On the recorded 256-wide original axis, nearest mask resize uses every
    # second source index: W=49..78 -> original Y=98,100,...,156.
    if summary['target_shape'] != [64, 128, 128]:
        raise ValueError('The documented shared-grid protocol needs this target shape')
    if summary['shared_2d_region']['original_indices_inclusive'] != [98, 157]:
        raise ValueError('Unexpected supplied 2D band')
    for key in ('prostate_fp_in_bladder', 'prostate_fp_in_rectum'):
        if sum(int(r[key]) for r in full_rows if r['class_id'] == '5') != summary[key]:
            raise ValueError(f'Confusion count mismatch: {key}')
    distances = [float(r['hd95_mm']) for r in full_rows
                 if r['class_id'] == '5' and r['hd95_status'] == 'ok']
    if len(distances) != summary['prostate_hd95_valid_cases'] or not math.isclose(
            mean(distances), summary['prostate_mean_hd95_mm'], abs_tol=1e-12):
        raise ValueError('Full-volume HD95 mismatch')

    common = dict(region='shared_band', grid='D64_H128_W30; full-grid W49..78',
                  patients=6, scans=27, source_indices='98,100,...,156',
                  aggregation='mean_scan; pooled metrics separately labelled')
    metric_keys = ['mean_case_dice', 'std_case_dice', 'mean_patient_dice',
                   'mean_case_iou', 'pooled_dice', 'pooled_precision',
                   'pooled_recall', 'pooled_volume_ratio']
    entries = []
    for region, metrics in [('full_volume', full[5]), ('shared_band', shared[5])]:
        item = dict(common, model='standard_3d_unet', region=region,
                    status='measured', **{k: metrics[k] for k in metric_keys},
                    hd95_mm=None, fp_in_bladder=None, fp_in_rectum=None)
        if region == 'full_volume':
            item.update(grid='D64_H128_W128', source_indices='full-grid W indices 0..127',
                        hd95_mm=summary['prostate_mean_hd95_mm'],
                        fp_in_bladder=summary['prostate_fp_in_bladder'],
                        fp_in_rectum=summary['prostate_fp_in_rectum'])
        entries.append(item)
    for model, status in [('standard_2d_unet', 'awaiting_frozen_test_evaluation'),
                          ('improved_2d_unet', 'optional_not_scheduled')]:
        entries.append(dict(common, model=model, status=status,
                            **{k: None for k in metric_keys},
                            hd95_mm=None, fp_in_bladder=None, fp_in_rectum=None))
        entries.append(dict(common, model=model, status=status, region='native_2d_band',
                            grid='60 slices; original unpadded 256x128 or 256x144',
                            source_indices='98..157 inclusive; all 60 planes',
                            **{k: None for k in metric_keys},
                            hd95_mm=None, fp_in_bladder=None, fp_in_rectum=None))
    sources = {name: hashlib.sha256((ARCHIVE/name).read_bytes()).hexdigest()
               for name in ['summary.json', 'per_case_metrics.csv',
                            'shared_2d_region_metrics.csv', 'split.json']}
    report = dict(checkpoint_sha256=summary['checkpoint_sha256'],
                  test_case_ids=sorted({r['case'] for r in full_rows}),
                  test_patient_ids=sorted({r['patient_id'] for r in full_rows}),
                  sources_sha256=sources, entries=entries,
                  shared_hd95_and_organ_confusion='not measured in shared-band CSV',
                  shared_2d_alignment_status='pending actual 2D predictions and exact GT grid check',
                  ranking_status='no paired ranking until aligned 2D test evaluation')
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'metrics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with (OUTPUT/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader(); writer.writerows(entries)

    lines = ['## Model comparison: complete volume and shared band', '',
             'The tables below separate measured scopes. Pending is not a score of zero.',
             'No 2D test result or 3D-versus-2D ranking is available yet. Improved 2D',
             'U-Net is an optional unscheduled row, not an additional active experiment.', '',
             '### A. Complete-volume 3D reference', '',
             '| Model / scope | Test patients/scans | Mean scan Dice ± SD | Mean IoU | HD95 (approx. mm) | Pooled volume ratio |',
             '| --- | ---: | ---: | ---: | ---: | ---: |',
             f'| Standard 3D U-Net; 64×128×128 full grid | 6 / 27 | {full[5]["mean_case_dice"]:.4f} ± {full[5]["std_case_dice"]:.4f} | {full[5]["mean_case_iou"]:.4f} | {summary["prostate_mean_hd95_mm"]:.2f} | {full[5]["pooled_volume_ratio"]:.4f} |', '',
             'This table supports the complete-volume 3D result. Native 2D partial-slice',
             'results cannot be placed in this scope or used to claim a direct advantage.', '',
             '### B. Shared-band prostate comparison (separate evaluation scope)', '',
             '| Model | Test patients/scans | Mean scan Dice ± SD | Mean patient Dice | Pooled precision / recall | Pooled volume ratio | Shared HD95 / organ invasion |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |',
             f'| Standard 3D U-Net | 6 / 27 | {shared[5]["mean_case_dice"]:.4f} ± {shared[5]["std_case_dice"]:.4f} | {shared[5]["mean_patient_dice"]:.4f} | {shared[5]["pooled_precision"]:.4f} / {shared[5]["pooled_recall"]:.4f} | {shared[5]["pooled_volume_ratio"]:.4f} | Not measured for this scope |',
             '| Standard 2D U-Net | Required: same 6 / 27 | Pending | Pending | Pending | Pending | Pending |',
             '| Improved 2D U-Net (optional; not scheduled) | Required: same 6 / 27 | Pending | Pending | Pending | Pending | Pending |', '',
             'Shared grid: D=64, H=128, full-volume W indices 49..78 (30 planes).',
             'These nearest-resized mask planes correspond to original axis-1 indices',
             '98,100,...,156 within the supplied 60-slice band [98,157]. This differs',
             'from scoring all 60 native-resolution 2D slices. Actual 2D predictions',
             'must first be mapped to this grid and their GT checked for exact equality.', '',
             'The full/shared 3D prostate counts happen to agree in this archived run.',
             'That does not justify copying complete-volume HD95 or invasion counts',
             'into the shared-band table. Those metrics require their own spatial',
             'calculation. Labels on crop boundaries can affect surface distances.', '',
             '### C. Six-class region check (3D only)', '',
             '| Class | Full-volume mean scan Dice | Shared-band mean scan Dice |',
             '| --- | ---: | ---: |']
    for a, b in zip(full, shared):
        lines.append(f'| {a["class_name"]} | {a["mean_case_dice"]:.4f} | {b["mean_case_dice"]:.4f} |')
    lines += ['', '### D. Paired-case template on the shared grid', '',
              '| Test case | 3D shared-band prostate Dice | 2D shared-band Dice | Improved 2D (optional) |',
              '| --- | ---: | --- | --- |']
    for case in read_csv(ARCHIVE/'case_review_metrics.csv'):
        row = next(r for r in shared_rows if r['case'] == case['case'] and r['class_id'] == '5')
        lines.append(f'| {case["case"].replace("_LFOV.nii.gz", "")} | {float(row["dice"]):.4f} | Pending | Pending |')
    lines += ['', 'Reuse these fixed case identities for the 2D overlays; include all 27',
              'scans in numerical comparisons rather than evaluating only these examples.', '',
              '### E. Resource profile (separate from accuracy scope)', '',
              '| Model | Parameters | Precision / hardware | Forward latency | Peak inference allocation |',
              '| --- | ---: | --- | --- | --- |',
              f'| Standard 3D U-Net | {summary["parameters"]:,} | BF16 / {summary["hardware"]["gpu"]} | {summary["median_forward_latency_ms"]:.2f} ms median per volume | {summary["peak_allocated_gpu_gb"]:.4f} GiB |',
              '| Standard 2D U-Net | Pending checkpoint configuration verification | Pending measurement | Pending full-scan measurement | Pending |',
              '| Improved 2D U-Net (optional) | Pending implementation | Pending | Pending | Pending |', '',
              'The existing 3D timing uses the full input volume, excludes data loading,',
              'transfer, argmax, metrics and plotting, and is not training memory/time.',
              'A 2D per-slice time is not comparable with a 3D per-volume time. Report',
              'complete-scan 2D timing and use matched hardware/precision for architecture',
              'claims; otherwise label the hardware difference. Shared-band accuracy does',
              'not mean the 3D network was run on a cropped input.', '',
              '### F. Native-resolution 2D results (not the shared-grid comparison)', '',
              '| Model / scope | Test patients/scans/slices | Mean scan prostate Dice | Mean IoU |',
              '| --- | --- | --- | --- |',
              '| Standard 2D U-Net; all 60 original unpadded slices per scan | Required: 6 / 27 / 1620 | Pending | Pending |',
              '| Improved 2D U-Net (optional); same native band | Required: 6 / 27 / 1620 | Pending | Pending |', '',
              'This is the native 2D task, grouped by scan. Its figures and metrics use',
              'original valid pixels (256×128 or 256×144); padding label 255 is ignored.',
              'These results are not substituted into table B without grid alignment.', '',
              'Protocol and missing-result rules: [comparison protocol](support/docs/comparison_protocol.md).',
              'Auditable tables: [comparison/metrics.json](support/comparison/metrics.json) and',
              '[comparison/metrics.csv](support/comparison/metrics.csv). Regenerate with',
              '`python -m support.prepare_comparison`; this only reads archived local records.', '']
    section = '\n'.join(lines)
    readme_path = ROOT/'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    block = START+'\n'+section+END
    if START in readme:
        prefix, tail = readme.split(START, 1)
        _, suffix = tail.split(END, 1)
        readme = prefix+block+suffix
    else:
        marker = '## Qualitative error analysis: five held-out cases'
        readme = readme.replace(marker, block+'\n\n'+marker)
    readme_path.write_text(readme, encoding='utf-8')
    print('Verified both 162-row scopes against frozen summary; wrote tables and README. No inference/jobs.')


if __name__ == '__main__':
    main()
