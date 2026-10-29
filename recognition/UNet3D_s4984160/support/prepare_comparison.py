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
ARCHIVE_2D = ROOT.parent / 'UNet2D_s4984160/support/artifacts/final_test_2d'
RESOURCE = ROOT.parent / 'UNet2D_s4984160/support/artifacts/resource_717015'
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
    if len(observed) != 6 or len(expected) != 6:
        raise ValueError('Expected all six classes')
    for actual, frozen in zip(observed, expected):
        for key, value in actual.items():
            if isinstance(value, float):
                if not math.isclose(value, frozen[key], abs_tol=1e-12):
                    raise ValueError(f'Summary mismatch: class {actual["class_id"]}, {key}')
            elif value != frozen[key]:
                raise ValueError(f'Summary mismatch: {key}')


def load_completed_2d(reference, reference_full_rows, reference_shared_rows, entries, report):
    """Verify the completed paired experiment independently from its CSV counts."""
    if not (ARCHIVE_2D/'summary.json').exists():
        return None
    summary = json.loads((ARCHIVE_2D/'summary.json').read_text(encoding='utf-8'))
    if not (summary['test_evaluated'] and summary['shared_gt_all_voxels_equal']
            and summary['reference_counts_match_archive']):
        raise ValueError('Completed 2D test/alignment evidence missing')
    if summary['reference_checkpoint_sha256'] != reference['checkpoint_sha256']:
        raise ValueError('Paired experiment used a different 3D reference')
    if summary['checkpoint_epoch'] != 5 or summary['checkpoint_sha256'] != (
            '3a666e057827897d294d9df9604ac325617fa874bf541402011537847024174b'):
        raise ValueError('Unexpected selected 2D checkpoint')
    if summary['shared_grid'] != [64,128,30] or summary['shared_W_indices'] != list(range(49,79)):
        raise ValueError('Unexpected paired grid')
    expected = {(r['case'], r['patient_id'], r['class_id']) for r in reference_full_rows}
    names = dict(native_2d='native_2d_per_case_metrics.csv',
                 shared_2d='shared_2d_per_case_metrics.csv',
                 shared_3d='reference_3d_shared_metrics.csv')
    scopes = {}
    for scope, name in names.items():
        rows = read_csv(ARCHIVE_2D/name)
        if {(r['case'], r['patient_id'], r['class_id']) for r in rows} != expected:
            raise ValueError(f'{scope}: cohort differs from original 3D test')
        metrics = score_rows(rows)
        verify_summary(metrics, summary[scope]['class_metrics'])
        prostate = [r for r in rows if r['class_id'] == '5']
        distances = [float(r['hd95_mm']) for r in prostate if r['hd95_status']=='ok']
        frozen = summary[scope]
        if (frozen['scans'], frozen['patients']) != (27,6):
            raise ValueError('Unexpected scope cohort')
        if len(distances) != frozen['prostate_hd95_valid_cases'] or not math.isclose(
                mean(distances), frozen['prostate_mean_hd95_mm'], abs_tol=1e-12):
            raise ValueError(f'{scope}: HD95 disagrees')
        if len(prostate)-len(distances) != frozen['prostate_hd95_invalid_cases']:
            raise ValueError('HD95 validity count differs')
        for key in ('prostate_fp_in_bladder','prostate_fp_in_rectum'):
            if sum(int(r[key]) for r in prostate) != frozen[key]:
                raise ValueError(f'{scope}: invasion count differs')
        scopes[scope] = dict(metrics=metrics, rows=rows,
                             cases={r['case']:r for r in prostate})
    prior = {(r['case'],r['class_id']):r for r in reference_shared_rows}
    for r in scopes['shared_3d']['rows']:
        original = prior[(r['case'],r['class_id'])]
        if any(int(r[k]) != int(original[k]) for k in ('tp','fp','fn')):
            raise ValueError('Reinferred shared 3D counts differ from original archive')
    gt = {(r['case'],r['class_id']):r['gt_voxels'] for r in scopes['shared_3d']['rows']}
    if any(r['gt_voxels'] != gt[(r['case'],r['class_id'])] for r in scopes['shared_2d']['rows']):
        raise ValueError('2D/3D shared GT count mismatch')
    paired = read_csv(ARCHIVE_2D/'paired_cases.csv')
    if len(paired)!=27 or len({r['case'] for r in paired})!=27:
        raise ValueError('Missing or duplicate paired cases')
    for r in paired:
        if int(r['gt_mismatch_voxels'])!=0:
            raise ValueError('Shared GT mismatch')
        for key,scope in [('native_2d_dice','native_2d'),('shared_2d_dice','shared_2d'),('shared_3d_dice','shared_3d')]:
            if not math.isclose(float(r[key]),float(scopes[scope]['cases'][r['case']]['dice']),abs_tol=1e-12):
                raise ValueError('Paired Dice differs from scope rows')
        if not math.isclose(float(r['delta_2d_minus_3d']),float(r['shared_2d_dice'])-float(r['shared_3d_dice']),abs_tol=1e-12):
            raise ValueError('Paired delta differs')
    for key,field in [('gt_scans','gt_touches_shared_W_boundary'),
                      ('prediction_2d_scans','prediction_2d_touches_shared_W_boundary'),
                      ('prediction_3d_scans','prediction_3d_touches_shared_W_boundary')]:
        if sum(r[field]=='True' for r in paired)!=summary['shared_boundary_touches'][key]:
            raise ValueError('Boundary contact count differs')
    for item in entries:
        if item['model']=='improved_2d_unet':continue
        if item['model']=='standard_3d_unet' and item['region']=='full_volume':continue
        scope = 'shared_3d' if item['model']=='standard_3d_unet' else (
            'shared_2d' if item['region']=='shared_band' else 'native_2d')
        data=summary[scope]; metrics=scopes[scope]['metrics'][5]
        item.update(status='measured', **{k:metrics[k] for k in metrics if k in item},
                    hd95_mm=data['prostate_mean_hd95_mm'],
                    fp_in_bladder=data['prostate_fp_in_bladder'],fp_in_rectum=data['prostate_fp_in_rectum'])
    patient_rows=[]
    for patient in sorted({r['patient_id'] for r in paired}):
        group=[r for r in paired if r['patient_id']==patient]
        a=mean([float(r['shared_2d_dice']) for r in group]);b=mean([float(r['shared_3d_dice']) for r in group])
        patient_rows.append(dict(patient_id=patient,scans=len(group),shared_2d_dice=a,shared_3d_dice=b,delta_2d_minus_3d=a-b))
    deltas=[float(r['delta_2d_minus_3d']) for r in paired]
    result=dict(mean_scan_delta=mean(deltas),median_scan_delta=statistics.median(deltas),
                mean_patient_delta=mean([r['delta_2d_minus_3d'] for r in patient_rows]),
                scans_2d_higher=sum(d>1e-8 for d in deltas),scans_3d_higher=sum(d< -1e-8 for d in deltas),
                patients_2d_higher=sum(r['delta_2d_minus_3d']>1e-8 for r in patient_rows),
                native_2d_scans_below_075=sum(float(r['native_2d_dice'])<.75 for r in paired))
    native=scopes['native_2d']['metrics'][5]['mean_case_dice']
    if summary['native_2d_target_met'] != (native>=.75):raise ValueError('Target status differs')
    report.update(checkpoint_2d_sha256=summary['checkpoint_sha256'],
                  sources_2d_sha256={name:hashlib.sha256((ARCHIVE_2D/name).read_bytes()).hexdigest()
                    for name in ['summary.json',*names.values(),'paired_cases.csv']},
                  shared_hd95_and_organ_confusion='Independently measured in frozen paired job 716656',
                  shared_2d_alignment_status='Exact GT equality for all 27 scans; reference counts match archive',
                  shared_boundary_touches=summary['shared_boundary_touches'],
                  paired_descriptive_results=result,
                  ranking_status='Descriptive comparison of these pipelines, not an isolated dimensionality ablation')
    return dict(summary=summary,scopes=scopes,paired=paired,patients=patient_rows,descriptive=result)


def verified_resources(reference, full_rows):
    """Verify whole-case aggregates and the direct prediction against raw records."""
    profile = json.loads((RESOURCE/'summary.json').read_text())
    paired = json.loads((ARCHIVE_2D/'summary.json').read_text())
    if (profile['checkpoint_3d_sha256'] != reference['checkpoint_sha256'] or
            profile['checkpoint_2d_sha256'] != paired['checkpoint_sha256']):
        raise ValueError('Resource checkpoint mismatch')
    if not profile['models_measured_separately'] or not profile['no_training'] or profile['tf32']:
        raise ValueError('Unexpected resource protocol')
    rows = read_csv(RESOURCE/'per_case_profile.csv')
    modes = {(p['model'], p['precision']) for p in profile['profiles']}
    if len(rows) != 81 or modes != {('standard_2d_unet','fp32'), ('standard_3d_unet','fp32'), ('standard_3d_unet','bf16')}:
        raise ValueError('Incomplete resource profiles')
    expected_cases = {r['case'] for r in full_rows}
    for p in profile['profiles']:
        group = [r for r in rows if (r['model'], r['precision']) == (p['model'], p['precision'])]
        if len(group) != 27 or {r['case'] for r in group} != expected_cases or p['scans'] != 27:
            raise ValueError('Resource cohort mismatch')
        if p['non_model_allocated_bytes_before_load'] != 0:
            raise ValueError('Unexpected resident GPU allocation')
        expected_parameters = paired['parameters_2d'] if p['model']=='standard_2d_unet' else reference['parameters']
        if p['parameters'] != expected_parameters:
            raise ValueError('Parameter count mismatch')
        for field, aggregate in [('forward_case_ms_mean', statistics.mean),
                                 ('forward_case_ms_median', statistics.median),
                                 ('prediction_case_ms_mean', statistics.mean),
                                 ('prediction_case_ms_median', statistics.median),
                                 ('peak_allocated_gib', max), ('peak_reserved_gib', max)]:
            source = field.replace('_mean','').replace('_median','')
            value = aggregate(float(r[source]) for r in group)
            if not math.isclose(value, p[field], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Resource aggregate differs: {field}')
    demo = json.loads((RESOURCE/'predict_demo/summary.json').read_text())
    if demo['checkpoint_sha256'] != reference['checkpoint_sha256'] or demo['precision'] != 'bf16':
        raise ValueError('Direct prediction checkpoint/precision mismatch')
    for metric in demo['class_metrics']:
        original = next(r for r in full_rows if r['case']==demo['case'] and int(r['class_id'])==metric['class_id'])
        for key in ('tp','fp','fn','gt_voxels','pred_voxels'):
            if int(original[key]) != metric[key]:
                raise ValueError(f'Direct prediction count differs: {key}')
        if metric['class_id']==5 and not math.isclose(float(original['hd95_mm']), demo['prostate_hd95_mm'], abs_tol=1e-10):
            raise ValueError('Direct prediction HD95 differs')
    for marker in ('profile_complete.txt','predict_demo/prediction_complete.txt'):
        if not (RESOURCE/marker).read_text().strip():
            raise ValueError('Missing resource completion marker')
    if (RESOURCE/'resource_717015.err').stat().st_size:
        raise ValueError('Resource error log is not empty')
    sources = {str(path.relative_to(RESOURCE)).replace('\\','/'): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(RESOURCE.rglob('*')) if path.is_file() and path.name!='README.md'}
    return dict(job_id=717015, measurement=profile, direct_prediction=demo, sources_sha256=sources)


def resource_table(resources):
    if resources is None:
        return ['### E. Resource profile and confounders', '',
                'Isolated resource measurements are not archived; no values are inferred.', '']
    profile=resources['measurement']
    lines=['### E. Resource profile and confounders','',
           '| Model | Parameters | Precision | Forward ms/case | Prediction ms/case | Peak allocated / reserved GiB |',
           '| --- | ---: | --- | ---: | ---: | ---: |']
    for p in profile['profiles']:
        name='Standard 2D' if p['model']=='standard_2d_unet' else 'Standard 3D'
        lines.append(f'| {name} | {p["parameters"]:,} | {p["precision"].upper()} | {p["forward_case_ms_median"]:.2f} | {p["prediction_case_ms_median"]:.2f} | {p["peak_allocated_gib"]:.4f} / {p["peak_reserved_gib"]:.4f} |')
    lines += ['', 'Job **717015** measured all rows separately on the same **NVIDIA A100-PCIE-40GB**;',
              'each model started with zero other PyTorch GPU allocations. TF32 was disabled.',
              'Times are medians across 27 per-case medians, with three warmups and three repeats.',
              '2D means all 60 native slices (batch 16); 3D means the full 64×128×128 volume (batch 1).',
              'Forward excludes transfers; prediction includes transfers, forward, argmax and output copy.',
              'Both exclude disk loading, preprocessing and initial model loading. Peaks include resident',
              'weights and inference tensors/cache, but exclude CUDA driver/context and training memory.',
              'Allocated and reserved are distinct PyTorch memory measures.','',
              'Use FP32 rows for the matched-precision comparison; BF16 records the 3D deployment mode.',
              'The earlier 3D-only timing (17.55 ms, 0.1364 GiB) remains in the original archive;',
              'this table uses the new isolated profiling protocol, without replacing segmentation scores.',
              'Coverage, resolution, normalization, model size and training differ, so these are pipeline',
              'costs, not an isolated causal test of convolution dimension.','',
              'Raw [resource records and prediction example](../UNet2D_s4984160/support/artifacts/resource_717015/README.md).','']
    return lines


def completed_tables(reference, full, shared, completed, resources):
    s=completed['summary']; scopes=completed['scopes']; d=completed['descriptive']
    a=scopes['shared_2d']['metrics'][5];b=scopes['shared_3d']['metrics'][5];n=scopes['native_2d']['metrics'][5]
    lines=['## Model comparison: complete volume and shared band','',
           'Frozen held-out evaluation job **716656** completed. All 27 shared GT grids',
           'matched voxel-for-voxel, and re-inferred 3D overlap counts reproduced the archive.',
           'The 2D native task and paired shared-grid task remain separate below.',
           'Improved 2D U-Net is optional and was not implemented or scheduled.','',
           '### A. Complete-volume 3D reference','',
           '| Scope | Patients/scans | Mean scan Dice ± SD | Mean IoU | HD95 (approx. mm) | Pooled volume ratio |',
           '| --- | ---: | ---: | ---: | ---: | ---: |',
           f'| Standard 3D; full 64×128×128 grid | 6 / 27 | {full[5]["mean_case_dice"]:.4f} ± {full[5]["std_case_dice"]:.4f} | {full[5]["mean_case_iou"]:.4f} | {reference["prostate_mean_hd95_mm"]:.2f} | {full[5]["pooled_volume_ratio"]:.4f} |','',
           '### B. Paired shared-grid prostate results','',
           '| Model | Mean scan Dice ± SD | Mean patient Dice | Mean IoU | HD95 (approx. mm) | Pooled precision / recall | Pred/GT ratio | FP in bladder / rectum |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,scope in [('Standard 3D U-Net','shared_3d'),('Standard 2D U-Net','shared_2d')]:
        c=scopes[scope]['metrics'][5];v=s[scope]
        lines.append(f'| {name} | {c["mean_case_dice"]:.4f} ± {c["std_case_dice"]:.4f} | {c["mean_patient_dice"]:.4f} | {c["mean_case_iou"]:.4f} | {v["prostate_mean_hd95_mm"]:.2f} | {c["pooled_precision"]:.4f} / {c["pooled_recall"]:.4f} | {c["pooled_volume_ratio"]:.4f} | {v["prostate_fp_in_bladder"]} / {v["prostate_fp_in_rectum"]} |')
    lines += ['| Improved 2D (optional, not run) | — | — | — | — | — | — | — |','',
              f'2D has higher Dice in **{d["scans_2d_higher"]}/27 scans**; its mean scan gain is',
              f'**{100*d["mean_scan_delta"]:.2f} percentage points**. The gain falls to',
              f'**{100*d["mean_patient_delta"]:.2f} points** when weighting the six patients equally;',
              f'{d["patients_2d_higher"]}/6 patients favour 2D. Multiple weeks from one patient are correlated,',
              'so 27 scans are not 27 independent patients. These are descriptive results,',
              'without a statistical superiority claim.','',
              'Shared scope: D=64, H=128, original full-grid W=49..78 (30 planes),',
              'corresponding to original Y=98,100,...,156. Shared HD95 and invasion counts',
              'were computed independently in this paired job, not copied from full-volume',
              '3D metrics. One 2D prediction (H007 Week0) touches the cropped W boundary;',
              'no GT or 3D prostate does. Distances refer to approximate resized spacing',
              'and can be affected by ROI surface truncation.','',
              '### C. Six-class scope check','',
              '| Class | 3D full Dice | 3D shared Dice | 2D shared Dice |',
              '| --- | ---: | ---: | ---: |']
    for f,r,t in zip(full,shared,scopes['shared_2d']['metrics']):
        lines.append(f'| {f["class_name"]} | {f["mean_case_dice"]:.4f} | {r["mean_case_dice"]:.4f} | {t["mean_case_dice"]:.4f} |')
    lines += ['','### D. Fixed-case shared-grid comparison','',
              '| Case | 3D Dice | 2D Dice | 2D minus 3D |', '| --- | ---: | ---: | ---: |']
    for case in read_csv(ARCHIVE/'case_review_metrics.csv'):
        r=next(r for r in completed['paired'] if r['case']==case['case'])
        lines.append(f'| {case["case"].replace("_LFOV.nii.gz", "")} | {float(r["shared_3d_dice"]):.4f} | {float(r["shared_2d_dice"]):.4f} | {float(r["delta_2d_minus_3d"]):+.4f} |')
    lines += ['', 'The paired overlays are in the [completed 2D result record](../UNet2D_s4984160/support/artifacts/final_test_2d/README.md).',
              'They reuse the same five cases and positions, without selecting new favourable examples.','',
              *resource_table(resources),
              '### F. Native 2D task (all 60 unpadded slices per scan)','',
              '| Model | Patients/scans/slices | Mean scan Dice ± SD | Mean patient Dice | Mean IoU | Native-band HD95 (mm) |',
              '| --- | ---: | ---: | ---: | ---: | ---: |',
              f'| Standard 2D | 6 / 27 / 1620 | {n["mean_case_dice"]:.4f} ± {n["std_case_dice"]:.4f} | {n["mean_patient_dice"]:.4f} | {n["mean_case_iou"]:.4f} | {s["native_2d"]["prostate_mean_hd95_mm"]:.2f} |',
              '| Improved 2D (optional, not run) | — | — | — | — | — |','',
              f'The native 2D mean exceeds the 0.75 target; {d["native_2d_scans_below_075"]}/27 scans remain below 0.75.',
              'Native metrics use the original valid pixels and original 3D affine spacing',
              'for the supplied partial band. They are not full-volume 3D scores and must',
              'not be substituted into the shared-grid table.','',
              '### Interpretation','',
              '2D improves mean overlap, precision and mean HD95, with less total excess',
              'volume. However, shared-grid rectal invasion is **400 voxels versus 21**',
              'for 3D, and bladder invasion is 110 versus 89. 3D has higher pooled recall',
              '(0.8416 versus 0.8093). H007 Week0 is an important 2D failure, not hidden',
              'by the better average. Keep both models frozen; use the remaining errors',
              'for reporting and future validation experiments, not test-set tuning.','',
              'Rules: [comparison protocol](support/docs/comparison_protocol.md).',
              'Auditable outputs: [metrics.json](support/comparison/metrics.json),',
              '[metrics.csv](support/comparison/metrics.csv), and',
              '[patient_comparison.csv](support/comparison/patient_comparison.csv).',
              'Regenerate with `python -m support.prepare_comparison` (local records only).','']
    return '\n'.join(lines)


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
    completed = load_completed_2d(summary,full_rows,shared_rows,entries,report)
    OUTPUT.mkdir(exist_ok=True)
    resources = verified_resources(summary, full_rows) if completed and RESOURCE.exists() else None
    if resources:
        report['resource_profile'] = resources
        with (OUTPUT/'resources.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(resources['measurement']['profiles'][0]))
            writer.writeheader();writer.writerows(resources['measurement']['profiles'])
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'metrics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with (OUTPUT/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader(); writer.writerows(entries)
    if completed:
        with (OUTPUT/'patient_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(completed['patients'][0]))
            writer.writeheader();writer.writerows(completed['patients'])

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
    if completed:
        section = completed_tables(summary,full,shared,completed,resources)
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
    print('Verified archived scopes'+(' and completed paired 2D test records' if completed else '')+
          '; wrote tables and README. No inference/jobs.')


if __name__ == '__main__':
    main()
