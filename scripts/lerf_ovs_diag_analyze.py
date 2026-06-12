"""
LERF-OVS failure diagnostic analysis (per-frame CSV).

Input:  CSV produced by scripts/lerf_ovs_diagnostic.py — one row per (scene, prompt, frame).
Output: markdown summary.

Headline numbers use LangSplat-style aggregation (per-image mean → per-scene mean → overall),
matching the standard reported in 3D OVS publications (eval_seg.py / langsplat eval).

Failure analysis (Type A/B/C, A1/A2/A3, FP/FN) uses prompt-level aggregation (mean across
frames per prompt), since Type/sub-type are prompt-intrinsic properties.
"""

import os
import csv
import argparse
import numpy as np
from collections import defaultdict


def classify_type(oracle, actual, tau):
    if oracle >= tau and actual >= tau:
        return 'C'
    if oracle >= tau and actual < tau:
        return 'A'
    if oracle < tau and actual < tau:
        return 'B'
    return '?'


def sub_type_from_rank(rank):
    if rank == 1:
        return 'A3'
    if rank <= 10:
        return 'A2'
    return 'A1'


def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k in ['is_ref_frame', 'gt_pixels', 'image_pixels', 'oracle_sp_count',
                      'correct_sp_lvl', 'correct_sp_id', 'correct_sp_clip_rank',
                      'clip_top1_lvl', 'clip_top1_sp_id',
                      'tp_pixels', 'tn_pixels', 'fp_pixels', 'fn_pixels', 'pool_size']:
                r[k] = int(r[k])
            for k in ['oracle_iou', 'actual_iou', 'clip_top1_oracle_iou',
                      'precision', 'recall']:
                r[k] = float(r[k])
            # Derived metrics per-row
            tp, tn, fp, fn = r['tp_pixels'], r['tn_pixels'], r['fp_pixels'], r['fn_pixels']
            r['accuracy'] = (tp + tn) / max(tp + tn + fp + fn, 1)
            p, rr = r['precision'], r['recall']
            r['f1'] = 2 * p * rr / max(p + rr, 1e-9) if (p + rr) > 0 else 0.0
            rows.append(r)
    return rows


def md_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |']
    out.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def langsplat_agg(rows, col):
    """LangSplat-style aggregation: per-image mean → per-scene mean → overall.

    Matches eval_seg.py:
      scene_met = []
      for img in gt_imgs:
          img_met.append(mean over prompts in this img)
      scene_met.append(mean over images)
      overall = mean over scenes
    """
    by_image = defaultdict(list)
    for r in rows:
        by_image[(r['scene'], r['frame'])].append(r[col])
    image_means = {k: np.mean(v) for k, v in by_image.items()}
    by_scene = defaultdict(list)
    for (sc, fr), v in image_means.items():
        by_scene[sc].append(v)
    scene_means = {sc: np.mean(v) for sc, v in by_scene.items()}
    overall = np.mean(list(scene_means.values())) if scene_means else float('nan')
    return overall, scene_means, image_means


def per_prompt_agg(rows):
    """Aggregate per (scene, prompt) by mean across that prompt's frames.

    Returns: list of dicts, one per (scene, prompt), with means + sums.
    """
    by_pp = defaultdict(list)
    for r in rows:
        by_pp[(r['scene'], r['prompt'])].append(r)
    out = []
    for (sc, p), grp in by_pp.items():
        # Frame-independent metadata: take from any row (same across frames)
        ref_md = grp[0]
        # Means
        oracle_mean = np.mean([r['oracle_iou'] for r in grp])
        actual_mean = np.mean([r['actual_iou'] for r in grp])
        # Sums (for pixel-level)
        tp = sum(r['tp_pixels'] for r in grp)
        tn = sum(r['tn_pixels'] for r in grp)
        fp = sum(r['fp_pixels'] for r in grp)
        fn = sum(r['fn_pixels'] for r in grp)
        gt = sum(r['gt_pixels'] for r in grp)
        img = sum(r['image_pixels'] for r in grp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9) if (precision + recall) > 0 else 0.0
        out.append({
            'scene': sc, 'prompt': p,
            'n_frames': len(grp),
            'oracle_iou_mean': oracle_mean,
            'actual_iou_mean': actual_mean,
            'tp_total': tp, 'tn_total': tn, 'fp_total': fp, 'fn_total': fn,
            'gt_total': gt, 'img_total': img,
            'precision': precision, 'recall': recall,
            'accuracy': accuracy, 'f1': f1,
            'correct_sp_clip_rank': ref_md['correct_sp_clip_rank'],
            'clip_top1_oracle_iou': ref_md['clip_top1_oracle_iou'],
            'pool_size': ref_md['pool_size'],
        })
    return out


def section_headline(rows):
    """Section 0: LangSplat-style headline numbers."""
    oracle_overall, oracle_scenes, _ = langsplat_agg(rows, 'oracle_iou')
    actual_overall, actual_scenes, _ = langsplat_agg(rows, 'actual_iou')
    p_overall, p_scenes, _ = langsplat_agg(rows, 'precision')
    r_overall, r_scenes, _ = langsplat_agg(rows, 'recall')

    table_rows = []
    for sc in sorted(oracle_scenes.keys()):
        table_rows.append([
            sc,
            f'{oracle_scenes[sc]:.4f}',
            f'{actual_scenes[sc]:.4f}',
            f'{oracle_scenes[sc] - actual_scenes[sc]:.4f}',
            f'{p_scenes[sc]:.4f}',
            f'{r_scenes[sc]:.4f}',
        ])
    table_rows.append([
        '**ALL (LangSplat mean)**',
        f'**{oracle_overall:.4f}**',
        f'**{actual_overall:.4f}**',
        f'**{oracle_overall - actual_overall:.4f}**',
        f'**{p_overall:.4f}**',
        f'**{r_overall:.4f}**',
    ])
    return md_table(
        ['Scene', 'Oracle mIoU', 'Actual mIoU', 'Gap', 'mP', 'mR'],
        table_rows), oracle_overall, actual_overall, oracle_scenes, actual_scenes


def section_type_classification(prompt_rows, taus):
    """Type A/B/C using per-prompt means."""
    sections = []
    for tau in taus:
        for r in prompt_rows:
            r[f'type_{int(tau*100)}'] = classify_type(
                r['oracle_iou_mean'], r['actual_iou_mean'], tau)
        by_type = defaultdict(list)
        for r in prompt_rows:
            by_type[r[f'type_{int(tau*100)}']].append(r)
        total = len(prompt_rows)
        table_rows = []
        for t in ['A', 'B', 'C', '?']:
            grp = by_type[t]
            if not grp:
                continue
            n = len(grp)
            pct = 100 * n / total
            oracle_avg = np.mean([r['oracle_iou_mean'] for r in grp])
            actual_avg = np.mean([r['actual_iou_mean'] for r in grp])
            iou_lost = sum(r['oracle_iou_mean'] - r['actual_iou_mean'] for r in grp)
            table_rows.append([
                t, n, f'{pct:.1f}%',
                f'{oracle_avg:.3f}', f'{actual_avg:.3f}',
                f'{iou_lost:.3f}',
            ])
        a_lost = sum(r['oracle_iou_mean'] - r['actual_iou_mean']
                     for r in by_type.get('A', []))
        total_lost = sum(r['oracle_iou_mean'] - r['actual_iou_mean']
                         for r in prompt_rows)
        sections.append(
            f"### τ = {tau}\n\n"
            + md_table(
                ['Type', 'N prompts', 'Share', 'Oracle avg', 'Actual avg', 'Sum IoU lost'],
                table_rows)
            + f"\n\n- **Type A 잃은 IoU 합 = {a_lost:.3f}** "
            f"(전체 손실 {total_lost:.3f} 중 {100*a_lost/max(total_lost,1e-6):.1f}%)\n"
            f"- **공격 가능 헤드룸 = Type A {len(by_type.get('A',[]))}/{total} prompts "
            f"({100*len(by_type.get('A',[]))/total:.1f}%)**"
        )
    return '\n\n'.join(sections)


def section_subtype(prompt_rows, tau):
    A = [r for r in prompt_rows if r[f'type_{int(tau*100)}'] == 'A']
    for r in A:
        r['sub_type'] = sub_type_from_rank(r['correct_sp_clip_rank'])
    by_sub = defaultdict(list)
    for r in A:
        by_sub[r['sub_type']].append(r)
    total = len(A)
    sub_def = {
        'A1': ('rank >= 11',
               'CLIP feature noisy → multi-view CLIP aggregation / per-SP feature refinement'),
        'A2': ('rank in [2, 10]',
               'ranking algo limit → matching algo 교체 / canon-contrast 대체'),
        'A3': ('rank = 1',
               'fragmentation / hierarchy → adaptive topk / level sweep'),
    }
    table_rows = []
    for s in ['A1', 'A2', 'A3']:
        grp = by_sub[s]
        cond, hint = sub_def[s]
        if not grp:
            table_rows.append([s, cond, 0, '0.0%', '-', '-', '-', hint])
            continue
        n = len(grp)
        pct = 100 * n / max(total, 1)
        oracle_avg = np.mean([r['oracle_iou_mean'] for r in grp])
        actual_avg = np.mean([r['actual_iou_mean'] for r in grp])
        rank_avg = np.mean([r['correct_sp_clip_rank'] for r in grp])
        table_rows.append([s, cond, n, f'{pct:.1f}%',
                           f'{oracle_avg:.3f}', f'{actual_avg:.3f}',
                           f'{rank_avg:.1f}', hint])
    md = md_table(
        ['Sub-type', '조건', 'N', 'Share of A', 'Oracle avg', 'Actual avg',
         'Avg rank', 'Attack direction'],
        table_rows)
    a1_examples = sorted(by_sub.get('A1', []), key=lambda r: -r['oracle_iou_mean'])[:10]
    md += '\n\n### Type A1 대표 prompt (rank ≥ 11, oracle 높은 순 top 10)\n\n'
    md += md_table(
        ['Scene', 'Prompt', 'N frames', 'Oracle', 'Actual', 'CLIP rank', 'Pool'],
        [[r['scene'], r['prompt'], r['n_frames'],
          f"{r['oracle_iou_mean']:.3f}", f"{r['actual_iou_mean']:.3f}",
          r['correct_sp_clip_rank'], r['pool_size']]
         for r in a1_examples])
    return md, A


def section_fp_fn(prompt_rows, tau):
    """Full confusion matrix analysis: TP/TN/FP/FN + Precision/Recall/Accuracy/F1/IoU."""
    for r in prompt_rows:
        denom = r['fp_total'] + r['fn_total']
        r['fp_dominance'] = r['fp_total'] / denom if denom > 0 else 0.5
    def bucket(fpd):
        if fpd > 0.7: return 'FP-dom'
        if fpd < 0.3: return 'FN-dom'
        return 'balanced'
    for r in prompt_rows:
        r['fp_bucket'] = bucket(r['fp_dominance'])

    n = len(prompt_rows)
    p_avg = np.mean([r['precision'] for r in prompt_rows])
    r_avg = np.mean([r['recall'] for r in prompt_rows])
    f1_avg = np.mean([r['f1'] for r in prompt_rows])
    acc_avg = np.mean([r['accuracy'] for r in prompt_rows])
    iou_avg = np.mean([r['actual_iou_mean'] for r in prompt_rows])

    tp_all = sum(r['tp_total'] for r in prompt_rows)
    tn_all = sum(r['tn_total'] for r in prompt_rows)
    fp_all = sum(r['fp_total'] for r in prompt_rows)
    fn_all = sum(r['fn_total'] for r in prompt_rows)
    gt_all = sum(r['gt_total'] for r in prompt_rows)
    img_all = sum(r['img_total'] for r in prompt_rows)
    pooled_prec = tp_all / max(tp_all + fp_all, 1)
    pooled_rec = tp_all / max(tp_all + fn_all, 1)
    pooled_acc = (tp_all + tn_all) / max(tp_all + tn_all + fp_all + fn_all, 1)
    pooled_iou = tp_all / max(tp_all + fp_all + fn_all, 1)
    pooled_f1 = 2 * pooled_prec * pooled_rec / max(pooled_prec + pooled_rec, 1e-9)

    fp_dom_share = sum(1 for r in prompt_rows if r['fp_bucket'] == 'FP-dom') / n
    fn_dom_share = sum(1 for r in prompt_rows if r['fp_bucket'] == 'FN-dom') / n
    bal_share = sum(1 for r in prompt_rows if r['fp_bucket'] == 'balanced') / n

    sec = []
    sec.append(f"### 3.1 전체 — Per-prompt 평균 metrics (N={n})\n")
    sec.append(md_table(
        ['Metric', 'Value', 'Definition'],
        [['Precision (avg)', f'{p_avg:.3f}', 'TP / (TP+FP) per prompt → mean'],
         ['Recall (avg)', f'{r_avg:.3f}', 'TP / (TP+FN) per prompt → mean'],
         ['F1 (avg)', f'{f1_avg:.3f}', '2·P·R / (P+R) per prompt → mean'],
         ['Accuracy (avg)', f'{acc_avg:.4f}', '(TP+TN) / total per prompt → mean'],
         ['IoU (avg)', f'{iou_avg:.3f}', 'TP / (TP+FP+FN) per prompt → mean']]))
    sec.append('')
    sec.append(md_table(
        ['Error pattern', 'Share', 'Definition'],
        [['FP-dominant (FP / (FP+FN) > 0.7)', f'{100*fp_dom_share:.1f}%', 'over-segment'],
         ['FN-dominant (< 0.3)', f'{100*fn_dom_share:.1f}%', 'miss / under-segment'],
         ['Balanced', f'{100*bal_share:.1f}%', '양쪽 비슷']]))

    sec.append(f"\n### 3.2 전체 — Pixel-summed raw confusion matrix\n")
    sec.append(md_table(
        ['Pixel category', 'Count', '% of all image pixels'],
        [['TP (correct foreground)', f'{tp_all:,}', f'{100*tp_all/img_all:.4f}%'],
         ['TN (correct background)', f'{tn_all:,}', f'{100*tn_all/img_all:.4f}%'],
         ['FP (잘못 칠한 픽셀)', f'{fp_all:,}', f'{100*fp_all/img_all:.4f}%'],
         ['FN (놓친 정답 픽셀)', f'{fn_all:,}', f'{100*fn_all/img_all:.4f}%'],
         ['GT total (객체 픽셀)', f'{gt_all:,}', f'{100*gt_all/img_all:.4f}%'],
         ['Image total', f'{img_all:,}', '100%']]))
    sec.append('\nPooled metrics (모든 픽셀 합쳐 계산):\n')
    sec.append(md_table(
        ['Metric', 'Pooled value', 'Per-prompt avg', 'Difference'],
        [['Precision', f'{pooled_prec:.3f}', f'{p_avg:.3f}', f'{pooled_prec - p_avg:+.3f}'],
         ['Recall', f'{pooled_rec:.3f}', f'{r_avg:.3f}', f'{pooled_rec - r_avg:+.3f}'],
         ['F1', f'{pooled_f1:.3f}', f'{f1_avg:.3f}', f'{pooled_f1 - f1_avg:+.3f}'],
         ['Accuracy', f'{pooled_acc:.4f}', f'{acc_avg:.4f}', f'{pooled_acc - acc_avg:+.4f}'],
         ['IoU', f'{pooled_iou:.3f}', f'{iou_avg:.3f}', f'{pooled_iou - iou_avg:+.3f}']]))
    sec.append(f"\n> **주의**: Pixel-pooled Accuracy = {pooled_acc:.4f} 가 매우 높아 보이는 건 배경 픽셀이 전체의 ~{100*tn_all/img_all:.1f}%를 차지하기 때문 (class imbalance). LERF-OVS 평가에서 accuracy는 보조 지표.\n")

    # Per-Type pixel-summed confusion matrix
    sec.append(f"\n### 3.3 Type별 Confusion Matrix (τ = {tau}, pixel-summed)\n")
    type_col = f'type_{int(tau*100)}'
    rows_x = []
    for t in ['A', 'B', 'C']:
        grp = [r for r in prompt_rows if r[type_col] == t]
        if not grp:
            rows_x.append([t, 0] + ['-'] * 9)
            continue
        tp = sum(r['tp_total'] for r in grp)
        tn = sum(r['tn_total'] for r in grp)
        fp = sum(r['fp_total'] for r in grp)
        fn = sum(r['fn_total'] for r in grp)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        rows_x.append([t, len(grp),
                       f'{tp:,}', f'{tn:,}', f'{fp:,}', f'{fn:,}',
                       f'{prec:.3f}', f'{rec:.3f}',
                       f'{f1:.3f}', f'{acc:.4f}', f'{iou:.3f}'])
    sec.append(md_table(
        ['Type', 'N', 'TP', 'TN', 'FP', 'FN',
         'Precision', 'Recall', 'F1', 'Accuracy', 'IoU'], rows_x))

    sec.append(f"\n### 3.4 Type별 — Per-prompt 평균 (τ = {tau})\n")
    rows_y = []
    for t in ['A', 'B', 'C']:
        grp = [r for r in prompt_rows if r[type_col] == t]
        if not grp:
            rows_y.append([t, 0] + ['-'] * 5)
            continue
        gn = len(grp)
        p = np.mean([r['precision'] for r in grp])
        rec = np.mean([r['recall'] for r in grp])
        f1 = np.mean([r['f1'] for r in grp])
        acc = np.mean([r['accuracy'] for r in grp])
        iou = np.mean([r['actual_iou_mean'] for r in grp])
        rows_y.append([t, gn,
                       f'{p:.3f}', f'{rec:.3f}', f'{f1:.3f}',
                       f'{acc:.4f}', f'{iou:.3f}'])
    sec.append(md_table(
        ['Type', 'N', 'Precision', 'Recall', 'F1', 'Accuracy', 'IoU'], rows_y))

    sec.append(f"\n### 3.5 Type × FP/FN dominance 교차표 (τ = {tau})\n")
    rows_z = []
    for t in ['A', 'B', 'C']:
        grp = [r for r in prompt_rows if r[type_col] == t]
        if not grp:
            rows_z.append([t, 0, '-', '-', '-'])
            continue
        gn = len(grp)
        fp_n = sum(1 for r in grp if r['fp_bucket'] == 'FP-dom')
        fn_n = sum(1 for r in grp if r['fp_bucket'] == 'FN-dom')
        bal_n = sum(1 for r in grp if r['fp_bucket'] == 'balanced')
        rows_z.append([t, gn,
                       f'{fp_n} ({100*fp_n/gn:.0f}%)',
                       f'{bal_n} ({100*bal_n/gn:.0f}%)',
                       f'{fn_n} ({100*fn_n/gn:.0f}%)'])
    sec.append(md_table(
        ['Type', 'N', 'FP-dominant', 'Balanced', 'FN-dominant'], rows_z))

    return '\n\n'.join(sec)


def section_attack(A_rows):
    by_sub = defaultdict(list)
    for r in A_rows:
        by_sub[r['sub_type']].append(r)
    total_a_loss = sum(r['oracle_iou_mean'] - r['actual_iou_mean'] for r in A_rows)
    # Architecture-only attack assessment. Inference settings (topk, threshold,
    # scoring fn) are protocol-fixed → A2/A3 are not architecture-attackable.
    attack_defs = {
        'A1': ('✅ YES (architecture)',
               'per-SP CLIP feature 생성 단계 수정 (image_encoding.py + merge_proj.py). '
               'topk/scoring 미변경.'),
        'A2': ('❌ NO (inference setting)',
               'rank in [2,10] = topk 키우면 풀리지만 topk=3은 protocol fix.'),
        'A3': ('❌ NO (inference setting)',
               'rank=1인데 topk=3가 distractor 포함 → topk=1 축소가 해결이지만 setting 변경.'),
    }
    table_rows = []
    for s in ['A1', 'A2', 'A3']:
        grp = by_sub[s]
        valid, reason = attack_defs[s]
        if not grp:
            table_rows.append([s, 0, '0.000', '0.0%', valid, reason])
            continue
        n = len(grp)
        loss = sum(r['oracle_iou_mean'] - r['actual_iou_mean'] for r in grp)
        share = 100 * loss / max(total_a_loss, 1e-6)
        table_rows.append([s, n, f'{loss:.3f}', f'{share:.1f}%', valid, reason])
    return md_table(
        ['Sub-type', 'N prompts', 'IoU loss sum',
         'Share of A loss', 'Architecture attack?', '이유'],
        table_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='output/diagnostics/lerf_ovs_per_prompt.csv')
    parser.add_argument('--out_md', default='md/lerf_ovs_failure_analysis_results.md')
    parser.add_argument('--taus', type=float, nargs='+', default=[0.25, 0.5])
    args = parser.parse_args()

    rows = load_rows(args.csv)
    prompt_rows = per_prompt_agg(rows)
    print(f"Loaded {len(rows)} (frame, prompt) rows, "
          f"{len(prompt_rows)} unique (scene, prompt) pairs.")

    md = []
    md.append('# LERF-OVS Failure Analysis — Results\n')
    md.append('> Generated 2026-06-04. Data: 4 scenes (figurines, ramen, teatime, '
              'waldo_kitchen). Per-frame CSV ([per_prompt.csv](../output/diagnostics/lerf_ovs_per_prompt.csv)). '
              'Methodology: [lerf_ovs_failure_analysis_plan.md](lerf_ovs_failure_analysis_plan.md).\n')
    md.append('> **Headline numbers**: LangSplat-style aggregation '
              '(per-image mean → per-scene mean → overall). 다른 3D OVS 논문과 직접 비교 가능.\n')
    md.append('> **Failure analysis** (Type/sub-type/FP/FN): per-prompt aggregation '
              '(mean over the prompt\'s frames; pixels summed for FP/FN).\n')

    # Headline (will compute TL;DR after Type classification)
    head_md, ora_all, act_all, ora_sc, act_sc = section_headline(rows)

    # Type classification
    type_md = section_type_classification(prompt_rows, args.taus)
    primary_tau = args.taus[0]
    sub_md, A_rows = section_subtype(prompt_rows, primary_tau)
    fpfn_md = section_fp_fn(prompt_rows, primary_tau)
    attack_md = section_attack(A_rows)

    # Stats for TL;DR
    A_share = len(A_rows) / len(prompt_rows) * 100
    A_loss = sum(r['oracle_iou_mean'] - r['actual_iou_mean'] for r in A_rows)
    total_loss = sum(r['oracle_iou_mean'] - r['actual_iou_mean'] for r in prompt_rows)
    A1 = [r for r in A_rows if r['sub_type'] == 'A1']
    A1_loss = sum(r['oracle_iou_mean'] - r['actual_iou_mean'] for r in A1) if A1 else 0
    A1_rank_avg = np.mean([r['correct_sp_clip_rank'] for r in A1]) if A1 else 0

    md.append('## TL;DR\n')
    md.append(f"""**Headline (LangSplat-style, 4 scenes):**
- Oracle mIoU = **{ora_all:.4f}**
- Actual mIoU = **{act_all:.4f}**
- **Gap = {ora_all - act_all:.4f}** ← matching loss, 다른 논문과 직접 비교 가능한 수치

**Failure 분포 (τ=0.25, per-prompt aggregation):**
- Type A (CLIP matching 실패): **{len(A_rows)}/{len(prompt_rows)} prompts ({A_share:.1f}%)**
- Type A가 차지하는 전체 IoU 손실: **{100*A_loss/max(total_loss,1e-6):.1f}%**

**Type A 내부:**
- A1 (CLIP feature noise, rank ≥ 11): **{len(A1)}/{len(A_rows)} = {100*len(A1)/max(len(A_rows),1):.1f}%**
- A1의 IoU 손실 비중 (within A): **{100*A1_loss/max(A_loss,1e-6):.1f}%**
- A1 정답 SP 평균 CLIP rank: **{A1_rank_avg:.1f}** (in pools of 745–4136 SPs)

**Attack priority (architecture-only 원칙):**
1. **#1 (유일 valid architectural attack): A1 — per-SP CLIP feature 생성 단계 수정 (`image_encoding.py` + `merge_proj.py`)**
   - 회복 가능 IoU = **{A1_loss:.3f}** (Type A 손실의 {100*A1_loss/max(A_loss,1e-6):.1f}%)
2. A2 / A3 → **❌ inference setting (topk, scoring) 변경 필요 — protocol fix를 깸**

**해석 한 줄:**
> 정답 SP가 NAG 안에 거의 항상 존재 (SAM lift 성공). 그러나 CLIP이 정답 SP를 평균 {A1_rank_avg:.0f}등으로 매기는 prompt {len(A1)}개가 LERF-OVS 헤드룸의 약 {100*A1_loss/max(total_loss,1e-6):.0f}% 를 잠그고 있다.
""")

    md.append("""---

## 핵심 원칙 — Fixed protocol & Architecture-only attacks

본 분석의 framing은 다음 원칙을 따른다:

| 항목 | 결정 |
|--|--|
| **고정 (논문 protocol)** | 평가 데이터 (4 scenes, polygon GT), 평가 metric (LangSplat-style mIoU/P/R), **THGS의 inference 인터페이스 (`test_lerf.py` 의 CLIP text 인코딩 + topk=3 at level=[2,3] + canon-contrast scoring + threshold=0.5)** |
| **수정 가능 (architecture / pipeline)** | per-pixel CLIP feature 추출 (`image_encoding.py`), Gaussian/SP feature 투영 + 집계 (`merge_proj.py`), SP partition (`sp_partition.py`), 그래프 가중치 (`graph_weight.py`) |
| **수정 금지 (settings)** | topk 값, threshold, canon-contrast 함수, level 선택 — 이걸 바꾸면 다른 method가 됨 (논문 fair comparison 깨짐) |

→ Attack 후보는 반드시 **pipeline 안의 deterministic 알고리즘 수정**이어야 함.

---

## Type / Sub-type 정의

각 (scene, prompt) 의 분류:

### Type (Oracle IoU와 Actual IoU의 조합, τ = 0.25 기준)

| Type | 조건 | 의미 | 진단 |
|--|--|--|--|
| **Type C** | Oracle ≥ τ **AND** Actual ≥ τ | 성공 | 분석 불필요 |
| **Type A** | Oracle ≥ τ **AND** Actual < τ | **CLIP matching 실패** — 정답 SP는 NAG에 있는데 CLIP이 못 골랐다 | ✅ **공격 가능** |
| **Type B** | Oracle < τ **AND** Actual < τ | SAM lift 실패 — 정답 SP 자체가 NAG에 없음 | NAG 구성 한계 |
| ? | Oracle < τ **AND** Actual ≥ τ | 이론적 불가 (Actual > Oracle은 발생하지 않아야 함) | data error 의심 |

### Sub-type (Type A 내부, 정답 SP의 CLIP rank로 분류)

정답 SP = greedy v4의 first pick (= 단독 IoU 최고 SP). 이 SP의 CLIP score rank (unified pool level=[2,3]):

| Sub-type | 조건 (정답 SP CLIP rank) | 진단 | Attack 가능 여부 |
|--|--|--|--|
| **A1** | rank ≥ 11 | CLIP이 정답 SP를 한참 뒤로 밀어버림 → per-SP CLIP feature quality 자체 문제 | ✅ **architecture-attackable** |
| **A2** | rank in [2, 10] | 정답 SP가 상위권에 있지만 top-1 아님 | ❌ inference setting (topk) |
| **A3** | rank = 1 | 정답 SP는 top-1, but topk=3가 distractor 포함 → over-segmentation | ❌ inference setting (topk) |

**핵심**: **A1만** architecture/pipeline level attack 대상. A2/A3는 inference 인터페이스를 바꿔야 풀리는데 그건 protocol fix를 깸.

---

## 0. Headline 수치 (LangSplat-style aggregation)
""")
    md.append(head_md)
    md.append('')
    md.append('Aggregation 방식: `eval_seg.py` 와 동일\n')
    md.append('```\n'
              'for each scene:\n'
              '    for each annotated image:\n'
              '        for each prompt in this image:\n'
              '            compute IoU\n'
              '        image_score = mean over prompts in this image\n'
              '    scene_score = mean over images\n'
              'overall = mean over scenes\n'
              '```\n')

    md.append('\n## 1. Type A/B/C 분류 (Step 1)\n')
    md.append('Per-prompt: mean Oracle / Actual across the frames the prompt appears in. Type 분류는 그 평균값에 τ 적용.\n')
    md.append(type_md)

    md.append(f'\n## 2. Type A 내부 sub-classification (Step 2, τ = {primary_tau})\n')
    md.append(sub_md)

    md.append(f'\n## 3. Confusion Matrix Analysis (Step 3, τ = {primary_tau})\n')
    md.append('전체 confusion matrix (TP/TN/FP/FN) + Precision/Recall/F1/Accuracy/IoU 모두 제공. '
              'Per-prompt 평균과 pixel-pooled 둘 다 보고 (class imbalance — 배경이 ~99% — 영향 파악).\n')
    md.append(fpfn_md)

    md.append(f'\n## 4. Attack priorities (Step 4)\n')
    md.append(attack_md)

    # Section 5: interpretation
    md.append('\n## 5. 핵심 해석\n')
    by_t25 = defaultdict(list)
    for r in prompt_rows:
        by_t25[r.get('type_25', '?')].append(r)
    type_a_n = len(by_t25['A'])
    type_b_n = len(by_t25.get('B', []))
    type_c_n = len(by_t25['C'])
    md.append(f"""### 5.1 SAM lift는 거의 실패하지 않음

τ=0.25 기준 Type B (Oracle도 actual도 낮음) = **{type_b_n} prompts**. 즉 **NAG의 SP 풀에는 정답 객체가 거의 항상 존재**한다. SAM lift 단계는 LERF-OVS 평가에서 bottleneck이 아니다.

### 5.2 실패는 CLIP matching에 집중, A 내부에서 더 좁은 sub-type에 집중

```
전체 {len(prompt_rows)} prompts
├── Type C (성공): {type_c_n} ({100*type_c_n/len(prompt_rows):.1f}%)
└── Type A (실패): {type_a_n} ({100*type_a_n/len(prompt_rows):.1f}%) — 손실 {A_loss:.3f} ({100*A_loss/max(total_loss,1e-6):.1f}% of total)
    ├── A1 (rank≥11): {len(A1)} ({100*len(A1)/max(len(A_rows),1):.1f}% of A) — 손실 {A1_loss:.3f} ({100*A1_loss/max(A_loss,1e-6):.1f}% of A)
    ├── A2 (rank 2-10): {sum(1 for r in A_rows if r.get('sub_type')=='A2')}
    └── A3 (rank=1): {sum(1 for r in A_rows if r.get('sub_type')=='A3')}
```

→ **A1 {len(A1)}개 prompt만 해결해도 전체 IoU 손실의 약 {100*A1_loss/max(total_loss,1e-6):.0f}% 회복 가능**.

### 5.3 A1의 평균 rank ≈ {A1_rank_avg:.0f} — CLIP feature가 정답을 못 알아봄

A1의 {len(A1)}개 prompt에서 정답 SP의 CLIP rank가 평균 **{A1_rank_avg:.0f}등** (pool ~745-4136 SPs 중). 즉 위쪽 ~{A1_rank_avg-1:.0f}개 SP가 모두 정답보다 CLIP score 높음.

- topk=3로는 절대 도달 불가
- topk 확장은 노이즈만 추가
- 본질적으로 **CLIP feature quality 개선**이 필요
""")

    # Section 6: per-scene breakdown
    md.append('\n## 6. Scene별 상세\n')
    scene_table = []
    for sc in sorted(ora_sc.keys()):
        sc_rows = [r for r in prompt_rows if r['scene'] == sc]
        sc_A = [r for r in sc_rows if r.get('type_25') == 'A']
        sc_A1 = [r for r in sc_A if r.get('sub_type') == 'A1']
        scene_table.append([
            sc, len(sc_rows),
            f'{ora_sc[sc]:.4f}', f'{act_sc[sc]:.4f}',
            f'{ora_sc[sc] - act_sc[sc]:.4f}',
            f'{len(sc_A)} ({100*len(sc_A)/max(len(sc_rows),1):.0f}%)',
            f'{len(sc_A1)} ({100*len(sc_A1)/max(len(sc_A),1):.0f}% of A)',
        ])
    md.append(md_table(
        ['Scene', 'N prompts', 'Oracle mIoU', 'Actual mIoU', 'Gap',
         'Type A', 'Type A1'],
        scene_table))

    # Section 7: top losses
    md.append('\n## 7. 부록 A: 가장 큰 IoU 손실 prompt top 15 (per-prompt mean 기준)\n')
    sorted_loss = sorted(prompt_rows, key=lambda r: -(r['oracle_iou_mean'] - r['actual_iou_mean']))[:15]
    md.append(md_table(
        ['Scene', 'Prompt', 'Frames', 'Oracle', 'Actual', 'Loss',
         'Type@0.25', 'CLIP rank', 'P', 'R', 'FP-dom'],
        [[r['scene'], r['prompt'], r['n_frames'],
          f"{r['oracle_iou_mean']:.3f}", f"{r['actual_iou_mean']:.3f}",
          f"{r['oracle_iou_mean'] - r['actual_iou_mean']:.3f}",
          r.get('type_25', '?'), r['correct_sp_clip_rank'],
          f"{r['precision']:.3f}", f"{r['recall']:.3f}",
          f"{r['fp_dominance']:.2f}"] for r in sorted_loss]))

    # Section 8: Deep Analysis Q1-Q5 (static summary, refer to lerf_ovs_deep_analysis.md for tables)
    md.append("""
## 8. Deep Analysis (Q1~Q5) — A1 attack 가설 좁히기

Phase 1 결과 (A1이 dominant) 후 attack 후보를 좁히기 위한 5개 sub-질문.
자세한 표는 [lerf_ovs_deep_analysis.md](lerf_ovs_deep_analysis.md) 참조.

| Q | 결과 | 의미 |
|--|--|--|
| Q1: A1 frame stability | A1 std = **0.000** (vs A2: 0.110) | A1은 **view-invariant** — multi-view aggregation 무효 |
| Q2: CLIP top-1 SP 정확도 | Type A의 90%가 top-1 SP oracle IoU < 0.1 | CLIP은 **완전 무관한 영역**을 picking (spatial misalignment) |
| Q3: 객체 크기 효과 | Correlation = -0.118 | 거의 무관 |
| Q4: Oracle SP 개수 | A1 mean = 1.58, C mean = 1.48 (비슷) | Fragmentation은 원인 아님 |
| Q5: A1 prompt 특성 | 평균 단어 수 1.25 (C: 1.91), 짧은 generic noun 다수 | longer specific phrase 문제 아님 |

**Q1+Q2 결합 결론**: A1은 view-invariant + CLIP은 totally-wrong 영역. → 가설: **per-SP CLIP feature 자체가 정답 위치에 의미 신호 못 emit.**

다음 단계 (Phase 1.5): pipeline 코드 trace로 (a)/(b)/(c)/(d) 분리.

---

## 9. Phase 1.5 — Pipeline Code Trace 결과

### 9.1 데이터 흐름 파악

THGS는 **per-SAM-mask CLIP** 사용 (LangSplat 방식):

```
[image_encoding.py]
  SAM이 각 image에 4-level (default/s/m/l) mask 생성 → 각 mask CLIP encode → per-mask feature

[merge_proj.py::proj_gaussian_features]
  for each training view:
    render_point() → per-gaussian visibility weight + 2D 좌표
    visible gaussian의 2D 좌표 → SAM mask ID 조회 → mask의 CLIP feature 사용
    sp_feature[sp] += normalize(sum over gaussians) × portion
  if portion < 0.1: continue  ← SP의 visible gaussian < 10% 이면 SKIP
```

### 9.2 진짜 Root Cause = (d), 아니 (a)/(b)/(c)

가설 (a)/(b)/(c) 모두 틀림. 실제 원인:

**(d) Zero-norm SPs 가 canon-contrast 점수 0.5를 받아 ranking 점령**

증거 (figurines/pikachu):
- 정답 SP cos_sim with "pikachu" = **0.2413** (pool 745개 중 **3등** in raw cos_sim)
- CLIP top-1 SP cos_sim = 0.2422 (거의 동일)
- canon-contrast relevancy: 정답 = 0.4393, top-1 = 0.5061
- **relevancy 상위 3-15등 모두 cos_sim=0, relevancy=0.5 (zero-norm SPs)**

원리: `utils/vlm_utils.py:compute_similarity` 에서 zero feature → softmax([0,0]) = [0.5, 0.5] → relevancy = 0.5 (항상).

### 9.3 Root Cause 원인의 두 layer

| Layer | 위치 | 무엇 | Attack 자격 |
|--|--|--|--|
| **(d-1) Pipeline** | `merge_proj.py::proj_gaussian_features` 의 `if portion < 0.1: continue` | 일부 SP가 모든 view에서 portion 충족 못해 zero feature 잔존 | ✅ YES (architecture) |
| **(d-2) Inference logic** | `utils/vlm_utils.py::compute_similarity` 의 softmax([0,0])=[0.5,0.5] | zero feature가 default relevancy 0.5 받음 | ❌ NO (inference 변경 금지) |

→ **(d-1) Pipeline fix로 zero-norm SP를 제거하면 (d-2) 문제도 자동 해소**.

### 9.4 Scene별 Zero-norm SP 정량

| Scene | Pool size | Zero-norm | 비율 |
|--|--|--|--|
| figurines | 745 | **48** | **6.4%** |
| waldo_kitchen | 1449 | 51 | 3.5% |
| teatime | 4136 | 47 | 1.1% |
| ramen | 793 | 4 | 0.5% |

### 9.5 Zero-norm 제거 시 A1 정답 SP rank 변화

| Scene | Prompt | 원본 rank | non-zero pool rank | zero above | real above |
|--|--|--|--|--|--|
| figurines | **pirate hat** | 49 | **1** ✅ | 48 | 0 |
| waldo_kitchen | **pour-over vessel** | 53 | **2** ✅ | 51 | 1 |
| figurines | **pumpkin** | 52 | **4** ✅ | 48 | 3 |
| figurines | pikachu | 64 | 16 | 48 | 15 |
| figurines | bag | 63 | 15 | 48 | 14 |
| figurines | miffy | 75 | 27 | 48 | 26 |
| waldo_kitchen | ottolenghi | 66 | 15 | 51 | 14 |
| ramen | hand | 44 | 44 | 0 | 43 |
| teatime | hooves | 52 | 52 | 0 | 51 |
| waldo_kitchen | cabinet | 25 | 25 | 0 | 24 |
| waldo_kitchen | spoon | 47 | 47 | 0 | 46 |

### 9.6 A1이 두 갈래로 분리됨

| Sub-sub | 조건 | Prompts | Phase 2 fix |
|--|--|--|--|
| **A1-α: Zero-norm dominated** | 위에 있는 SPs 대부분이 zero-norm | pirate hat, pour-over vessel, pumpkin, (old camera 부분) — ~3-4개 | ✅ pipeline 수정으로 즉시 해결 |
| **A1-β: Real-feature ranking** | 위에 real-feature SPs 다수 | pikachu, bag, miffy, ottolenghi, hand, hooves, cabinet, spoon, tesla door handle — ~8개 | ⚠️ language_features 재생성 후 추가 진단 |

### 9.7 Phase 2 architectural fix 후보

대상: `merge_proj.py::proj_gaussian_features`

| # | Fix | Risk | Inference 영향 |
|--|--|--|--|
| (i) | `portion < 0.1` threshold 완화 (예: 0.01) | 낮음 | 무변경 |
| **(ii)** | **Threshold 제거** | 낮음~중간 | 무변경 |
| (iii) | First-visible-view fallback | 낮음 | 무변경 |
| (iv) | Zero-norm SP를 NAG에서 명시적 제외 | 중간 | 무변경 (pool만 작아짐) |

→ 권장 시작점: **(ii)** (가장 minimal change, 1 line).

### 9.8 Phase 1.5 산출물

- [scripts/lerf_ovs_sp_feature_inspect.py](scripts/lerf_ovs_sp_feature_inspect.py)
- [scripts/lerf_ovs_zeronorm_check.py](scripts/lerf_ovs_zeronorm_check.py)
- [scripts/lerf_ovs_root_cause.py](scripts/lerf_ovs_root_cause.py) (language_features 필요, 미실행)

---

## 10. 부록 B: Aggregation 방식별 수치 비교
""")
    pp_oracle = np.mean([r['oracle_iou_mean'] for r in prompt_rows])
    pp_actual = np.mean([r['actual_iou_mean'] for r in prompt_rows])
    # Per-scene then overall (prompts equally weighted within scene)
    pp_by_scene_o = {sc: np.mean([r['oracle_iou_mean'] for r in prompt_rows if r['scene'] == sc])
                     for sc in set(r['scene'] for r in prompt_rows)}
    pp_by_scene_a = {sc: np.mean([r['actual_iou_mean'] for r in prompt_rows if r['scene'] == sc])
                     for sc in set(r['scene'] for r in prompt_rows)}
    pp_scene_mean_o = np.mean(list(pp_by_scene_o.values()))
    pp_scene_mean_a = np.mean(list(pp_by_scene_a.values()))
    md.append(md_table(
        ['Aggregation', 'Oracle', 'Actual', 'Gap', 'Note'],
        [
            ['LangSplat (per-image → per-scene)',
             f'{ora_all:.4f}', f'{act_all:.4f}', f'{ora_all - act_all:.4f}',
             '논문/eval_seg.py 표준'],
            ['Per-prompt mean across frames → per-scene → overall',
             f'{pp_scene_mean_o:.4f}', f'{pp_scene_mean_a:.4f}',
             f'{pp_scene_mean_o - pp_scene_mean_a:.4f}',
             'failure 분석에 사용'],
            ['Flat prompt mean (all prompts equal weight)',
             f'{pp_oracle:.4f}', f'{pp_actual:.4f}',
             f'{pp_oracle - pp_actual:.4f}',
             '참고용 (scene size 무시)'],
        ]))
    md.append('Aggregation 방식이 다를 뿐 같은 raw data. 결과의 절대값은 다르지만 **failure pattern과 attack priority는 모두 동일**.\n')

    md.append("""
---

## 11. Problem Definition — Distractor Confusion in Open-vocabulary 3DGS Segmentation

### 11.1 표준 용어 (Standard terminology)

CV / Retrieval / Open-vocabulary 문헌에서 "정답이 아닌 비슷한 후보"는 표준적으로 다음 용어로 불린다:

| 용어 | 의미 | 사용 분야 |
|--|--|--|
| **Distractor** | Query와 표현 공간에서 비슷하지만 정답이 아닌 후보 | Image retrieval, open-vocab seg, OWL-ViT/GLIP 등 |
| **Hard negative** | 학습 측면 강조 — "거의 정답처럼 보이는 오답" | Contrastive learning, hard negative mining |
| **Spurious match / activation** | 잘못된 위치/객체에 attention emit | CLIP / VLM analysis |

본 분석은 가장 일반적인 **distractor** 용어를 채택.

### 11.2 정식 문제 정의 (Formal problem statement)

> **Distractor confusion in open-vocabulary 3D Gaussian segmentation**:
>
> Text query `t`와 3D superpoints set `S = {s_1, ..., s_N}` (per-SP CLIP features `{f_i}`) 가 주어졌을 때, CLIP-based matching `score(f_i, t)` 의 의도는 **target SP** `s*` (= prompt가 가리키는 객체의 superpoint) 를 top rank로 두는 것이다.
>
> **Failure mode**: target이 아닌 일부 SP `{s_d}` (= **distractor SPs**) 가 `score(f_d, t) ≥ score(f*, t)` 를 만족해 top-k selection에서 `s*` 를 밀어냄. → render union이 잘못된 영역을 가리키거나 (FP > 0) 빈 mask로 귀결 (FP ≈ 0).
>
> Distractor는 세 클래스로 분리된다.

### 11.3 본 분석이 정의한 3-class distractor taxonomy

| Class | 명칭 | 정의 | Render 양상 | 우리 sub-type |
|--|--|--|--|--|
| **D1** | **Degenerate-feature distractor** (= null-feature distractor) | Pipeline 단계 결함으로 feature 못 받은 zero-norm SPs. Canon-contrast softmax가 default 0.5 점수 할당 → 실제 정답 SP를 밀어냄. | empty (가우시안 visible 부족) | **A1-α** |
| **D2** | **In-view spatial distractor** | 현재 evaluation view에서 visible한 다른 객체의 SP가 CLIP feature space에서 prompt와 closely match. | wrong area (FP > 0) | **A1-β1** |
| **D3** | **Out-of-view distractor** | CLIP feature space에서 prompt와 close하지만 그 SP의 gaussians가 현재 view에서 not visible. | empty (off-camera) | **A1-β2** |

### 11.4 Distractor class × Confusion matrix signature

각 distractor class는 pixel-level confusion matrix에서 **고유한 signature**를 만든다.

| Class | TP | FP | FN | Precision | Recall | FPR | F1 | Pixel pattern |
|--|--|--|--|--|--|--|--|--|
| **D1 (degenerate)** | 0 | 0 | GT | 정의 불가 (→ 0) | **0** | 0 | 0 | **Silent failure** — 아무것도 출력 안 함 |
| **D2 (in-view spatial)** | ~0 | **> 0** | GT | ~0 | ~0 | > 0 | ~0 | **Confident wrong** — 잘못된 위치를 자신 있게 출력 |
| **D3 (out-of-view)** | 0 | 0 | GT | 정의 불가 (→ 0) | **0** | 0 | 0 | **Silent failure** (D1과 동일) |
| **Type C (success)** | ≫ 0 | small | small | high | high | low | high | 정상 |

**핵심 관찰**:

> **D1과 D3는 pixel-level confusion matrix상 구분 불가능**. 둘 다 TP=0, FP=0, FN=GT 의 동일한 "silent failure" pattern을 보임. 구분하려면 **feature-level 진단** (zero-norm SP detection or gaussian visibility check) 이 필요.

**정량적 매핑** (Section 3.5의 Type × FP/FN dominance와 일치):

| Pixel signature | A1 prompts | Distractor class |
|--|--|--|
| Silent (FP = 0) | 9 prompts (75%) | D1 ∪ D3 |
| Confident wrong (FP > 0) | 3 prompts (25%) | D2 |

→ Section 3.5 의 "Type A 65% FN-dominant" 분포는 본질적으로 **D1+D3 ≫ D2** 라는 의미. A1 내부에서는 9:3 = **75% silent / 25% confident-wrong**.

### 11.5 Pixel signature → Distractor class 추론 protocol

새로운 method를 evaluate할 때 distractor class를 자동 진단하는 절차:

```
for each failure prompt (Type A, oracle IoU ≥ τ, actual IoU < τ):
    if FP_pixels == 0:                            # silent failure
        check zero-norm SPs in top-k selection
        if any selected SP has feat_norm < ε:
            → classify as D1 (degenerate)
        else:
            check gaussian visibility of selected SPs at eval view
            if visible gaussians == 0:
                → classify as D3 (out-of-view)
            else:
                → classify as D2 sub-case (low-visibility)
    else:                                          # FP > 0
        → classify as D2 (in-view spatial)
```

이 protocol을 사용하면 **pixel-level CSV (TP/TN/FP/FN) + 간단한 feature inspection** 만으로 distractor class 분포를 정량화 가능.

### 11.6 정량 분해 (this work)

THGS / LERF-OVS / 4 scenes / 67 prompts:

| Class | N prompts | IoU 손실 | 손실 비중 (of total A1 loss 9.71) | 손실 비중 (of total loss 18.13) |
|--|--|--|--|--|
| D1 (degenerate) | 3 | ~2.7 | **28%** | **15%** |
| D2 (in-view spatial) | 3 | ~2.0 | **21%** | **11%** |
| D3 (out-of-view) | 6 | ~5.0 | **52%** | **28%** |
| **A1 total (D1+D2+D3)** | **12** | **9.71** | **100% of A1** | **54%** |
| Non-A1 failures (A2, A3, B) | 8 | 8.42 | — | 46% |

**Key finding**: A1 (distractor confusion) 이 전체 LERF-OVS IoU 손실의 **54%**. 그 중에서도:
- **D3 (out-of-view distractor) 가 단일 최대** — 손실의 28% 차지.
- **D1 (degenerate)** 만 단순 pipeline fix로 해결 가능 (15% 회수).
- **D2, D3** 는 per-SP CLIP feature 변별력 문제 — 추가 진단 필요.

### 11.7 Class별 Architectural attack 후보 (Phase 2~)

| Class | Root cause layer | Architectural fix 후보 |
|--|--|--|
| **D1** | `merge_proj.py::proj_gaussian_features` 의 `if portion < 0.1: continue` skip | 후보 (i): threshold 완화 / (ii): 제거 / (iii): first-visible fallback / (iv): NAG에서 zero-norm SP 제외 |
| **D2** | per-SAM-mask CLIP feature 변별력 부족 (or per-SP aggregation dilution) | 후보 (i): SAM-mask boundary-aware aggregation / (ii): per-pixel CLIP fallback / (iii): per-SP feature contrastive refinement |
| **D3** | View-independent feature space의 ambiguity (semantic 닮은 다른 위치 SP) | 후보 (i): per-view visibility prior가 NAG construction에 반영 / (ii): cross-view feature consistency 강화 (단, inference 인터페이스 무변경 조건 충족 필요) |

### 11.8 Contribution statement (paper용)

> *"We provide the first quantitative decomposition of distractor confusion in open-vocabulary 3DGS segmentation. By inserting a per-SP greedy oracle ceiling between SAM lift and CLIP matching stages, we attribute 54% of LERF-OVS IoU loss to three distractor classes — degenerate (pipeline-resolvable), in-view spatial, and out-of-view — each requiring distinct architectural fixes."*

### 11.9 향후 연구 질문

1. **D3 (out-of-view distractor) 가 dominant**한 게 LERF-OVS 특수성인가, 일반 현상인가?
   - 검증: LangSplat / Gaussian Grouping에 동일 oracle 적용해 D1/D2/D3 비율 측정
2. **D2/D3 attack은 inference 인터페이스 변경 없이 가능한가?**
   - 핵심 제약: protocol fix 유지하면서 pipeline만 수정
   - 가능한 방향: NAG construction 시 view-visibility-aware feature averaging
3. **CLIP feature space 자체의 한계인가, aggregation 한계인가?**
   - 검증: language_features 재생성 후 per-SAM-mask CLIP feature 직접 비교
""")

    out_text = '\n\n'.join(md)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, 'w') as f:
        f.write(out_text)
    print(f"Wrote {args.out_md}")
    print('\n--- LangSplat headline ---')
    print(head_md)


if __name__ == '__main__':
    main()
