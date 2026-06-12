"""
Deeper analysis of LERF-OVS failure diagnostic.

Answers 5 questions beyond Type/sub-type/FP-FN analysis:

Q1. Frame stability: do A1 prompts fluctuate or fail consistently across frames?
Q2. CLIP top-1 SP characterization: when CLIP picks wrong, how wrong (semantic near-miss vs spatial mis-align)?
Q3. Object size effect: does gt_pixels correlate with failure rate?
Q4. Oracle SP count: do failing prompts need more SPs?
Q5. A1 prompt commonalities: word count, specificity, etc.

Outputs markdown to stdout (or appendable section file).
"""

import csv
import argparse
import numpy as np
from collections import defaultdict


def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k in ['is_ref_frame', 'gt_pixels', 'oracle_sp_count',
                      'correct_sp_lvl', 'correct_sp_id', 'correct_sp_clip_rank',
                      'clip_top1_lvl', 'clip_top1_sp_id',
                      'tp_pixels', 'fp_pixels', 'fn_pixels', 'pool_size']:
                r[k] = int(r[k])
            for k in ['oracle_iou', 'actual_iou', 'clip_top1_oracle_iou',
                      'precision', 'recall']:
                r[k] = float(r[k])
            rows.append(r)
    return rows


def md_table(headers, body):
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in body:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def aggregate_per_prompt(rows):
    by_pp = defaultdict(list)
    for r in rows:
        by_pp[(r['scene'], r['prompt'])].append(r)
    out = []
    for (sc, p), grp in by_pp.items():
        ref = grp[0]
        oracle_mean = np.mean([r['oracle_iou'] for r in grp])
        actual_mean = np.mean([r['actual_iou'] for r in grp])
        out.append({
            'scene': sc, 'prompt': p,
            'n_frames': len(grp),
            'oracle_iou_mean': oracle_mean,
            'actual_iou_mean': actual_mean,
            'oracle_iou_std': float(np.std([r['oracle_iou'] for r in grp])) if len(grp) > 1 else 0.0,
            'actual_iou_std': float(np.std([r['actual_iou'] for r in grp])) if len(grp) > 1 else 0.0,
            'actual_iou_min': min(r['actual_iou'] for r in grp),
            'actual_iou_max': max(r['actual_iou'] for r in grp),
            'gt_pixels_mean': float(np.mean([r['gt_pixels'] for r in grp])),
            'gt_pixels_total': sum(r['gt_pixels'] for r in grp),
            'oracle_sp_count': ref['oracle_sp_count'],
            'correct_sp_clip_rank': ref['correct_sp_clip_rank'],
            'clip_top1_oracle_iou': ref['clip_top1_oracle_iou'],
            'pool_size': ref['pool_size'],
            'iou_lost': oracle_mean - actual_mean,
        })
    # Type classification
    for r in out:
        r['type_25'] = ('A' if r['oracle_iou_mean'] >= 0.25 and r['actual_iou_mean'] < 0.25
                        else 'B' if r['oracle_iou_mean'] < 0.25 and r['actual_iou_mean'] < 0.25
                        else 'C' if r['oracle_iou_mean'] >= 0.25 and r['actual_iou_mean'] >= 0.25
                        else '?')
        if r['type_25'] == 'A':
            r['sub_type'] = ('A1' if r['correct_sp_clip_rank'] >= 11
                             else 'A2' if r['correct_sp_clip_rank'] > 1
                             else 'A3')
        else:
            r['sub_type'] = '-'
    return out


# Q1: Frame stability
def q1_frame_stability(rows, pp_rows):
    print('## Q1. Frame stability — view-dependent failure 인가?\n')
    print('Per-prompt: actual IoU std across that prompt\'s frames. '
          'Higher std = more view-dependent.\n')
    # Filter to prompts with ≥2 frames
    multi = [r for r in pp_rows if r['n_frames'] >= 2]
    print(f'N prompts with ≥2 frames: {len(multi)} / {len(pp_rows)}\n')
    # Group by Type
    by_type = defaultdict(list)
    for r in multi:
        by_type[r['type_25']].append(r)
    print(md_table(
        ['Type', 'N (≥2 frames)', 'Actual IoU std avg', 'Actual IoU std max',
         'Actual IoU (max-min) avg'],
        [[t, len(grp),
          f'{np.mean([r["actual_iou_std"] for r in grp]):.3f}',
          f'{max([r["actual_iou_std"] for r in grp]):.3f}' if grp else '-',
          f'{np.mean([r["actual_iou_max"] - r["actual_iou_min"] for r in grp]):.3f}']
         for t, grp in [('A', by_type['A']), ('C', by_type['C']), ('B', by_type.get('B', []))]
         if grp]))
    # Sub-type
    A_multi = [r for r in multi if r['type_25'] == 'A']
    sub = defaultdict(list)
    for r in A_multi:
        sub[r['sub_type']].append(r)
    print('\nType A 내부 sub-type 별 stability:\n')
    print(md_table(
        ['Sub-type', 'N (≥2 frames)', 'Actual IoU std avg', 'Actual range avg', 'Failure pattern'],
        [[s, len(grp),
          f'{np.mean([r["actual_iou_std"] for r in grp]):.3f}',
          f'{np.mean([r["actual_iou_max"] - r["actual_iou_min"] for r in grp]):.3f}',
          'consistent (low std)' if np.mean([r["actual_iou_std"] for r in grp]) < 0.05 else 'fluctuating']
         for s, grp in sub.items() if grp]))
    # Top fluctuators
    fluct = sorted(A_multi, key=lambda r: -r['actual_iou_std'])[:10]
    print('\n### Type A에서 가장 흔들리는 prompt top 10 (std 큰 순)\n')
    print(md_table(
        ['Scene', 'Prompt', 'Frames', 'Oracle mean', 'Actual mean', 'Actual std',
         'Actual range', 'Sub-type'],
        [[r['scene'], r['prompt'], r['n_frames'],
          f'{r["oracle_iou_mean"]:.3f}', f'{r["actual_iou_mean"]:.3f}',
          f'{r["actual_iou_std"]:.3f}',
          f'[{r["actual_iou_min"]:.2f}, {r["actual_iou_max"]:.2f}]',
          r['sub_type']] for r in fluct]))


# Q2: CLIP top-1 SP characterization
def q2_clip_top1(pp_rows):
    print('\n## Q2. CLIP top-1 SP characterization — 틀린 pick은 얼마나 틀렸나?\n')
    # When CLIP top-1 is "good" (= picks correct SP) vs "wrong" (= picks something else)
    # clip_top1_oracle_iou tells us how good the top-1 SP is vs GT.
    # If top-1 SP IS the correct SP (rank=1), top1_oracle_iou = the correct SP's oracle IoU.
    # If wrong (rank>1), top1_oracle_iou = oracle IoU of whatever wrong SP CLIP picked.

    by_type = defaultdict(list)
    for r in pp_rows:
        by_type[r['type_25']].append(r)

    print('Per Type: distribution of CLIP top-1 SP\'s oracle IoU '
          '(= "if you blindly trust CLIP top-1, what\'s your IoU?")\n')
    print(md_table(
        ['Type', 'N', 'top1_oracle_iou mean', 'median', 'min', 'max',
         '> 0.5 share', '< 0.1 share'],
        [[t, len(grp),
          f'{np.mean([r["clip_top1_oracle_iou"] for r in grp]):.3f}',
          f'{np.median([r["clip_top1_oracle_iou"] for r in grp]):.3f}',
          f'{min(r["clip_top1_oracle_iou"] for r in grp):.3f}',
          f'{max(r["clip_top1_oracle_iou"] for r in grp):.3f}',
          f'{100*sum(1 for r in grp if r["clip_top1_oracle_iou"] > 0.5)/len(grp):.0f}%',
          f'{100*sum(1 for r in grp if r["clip_top1_oracle_iou"] < 0.1)/len(grp):.0f}%']
         for t, grp in [('A', by_type['A']), ('C', by_type['C']), ('B', by_type.get('B', []))]
         if grp]))

    # For Type A: when CLIP top-1 is wrong, is the SP near or far from the answer?
    A = by_type['A']
    A_top1_iou = np.array([r['clip_top1_oracle_iou'] for r in A])
    print(f'\n**Type A의 top-1 SP의 oracle IoU 분포**:\n')
    print(f'- 0.0  : {sum(1 for x in A_top1_iou if x < 0.05)} prompts ({100*sum(1 for x in A_top1_iou if x < 0.05)/len(A_top1_iou):.0f}%) — totally wrong region')
    print(f'- 0.05-0.3: {sum(1 for x in A_top1_iou if 0.05 <= x < 0.3)} prompts — barely overlapping')
    print(f'- 0.3-0.5: {sum(1 for x in A_top1_iou if 0.3 <= x < 0.5)} prompts — semantic near-miss')
    print(f'- > 0.5: {sum(1 for x in A_top1_iou if x >= 0.5)} prompts — top-1 is actually OK')

    # Verdict
    pct_totally_wrong = 100*sum(1 for x in A_top1_iou if x < 0.05)/len(A_top1_iou)
    if pct_totally_wrong > 60:
        verdict = '**CLIP top-1 가 완전 엉뚱한 위치를 picking** → spatial misalignment dominant'
    elif pct_totally_wrong > 30:
        verdict = '**섞임** (일부 엉뚱, 일부 near-miss)'
    else:
        verdict = '**Semantic near-miss dominant** → CLIP은 비슷한 걸 찾지만 정답 SP는 아님'
    print(f'\n**결론**: {verdict}\n')


# Q3: Object size effect
def q3_object_size(pp_rows):
    print('\n## Q3. 객체 크기 효과 — 작은 객체가 더 잘 실패하나?\n')
    # Bin by gt_pixels_total (sum across all frames)
    sizes = sorted([r['gt_pixels_total'] for r in pp_rows])
    n = len(sizes)
    q1, q2, q3 = sizes[n // 4], sizes[n // 2], sizes[3 * n // 4]
    print(f'gt_pixels (per prompt, summed across frames) quartiles: '
          f'Q1={q1}, Q2(median)={q2}, Q3={q3}\n')

    def bin_size(px):
        if px < q1: return 'XS'
        if px < q2: return 'S'
        if px < q3: return 'M'
        return 'L'
    for r in pp_rows:
        r['size_bin'] = bin_size(r['gt_pixels_total'])

    by_bin = defaultdict(list)
    for r in pp_rows:
        by_bin[r['size_bin']].append(r)

    rows = []
    for b in ['XS', 'S', 'M', 'L']:
        grp = by_bin[b]
        if not grp:
            continue
        n_a = sum(1 for r in grp if r['type_25'] == 'A')
        n_a1 = sum(1 for r in grp if r.get('sub_type') == 'A1')
        rows.append([
            b, len(grp),
            f'{np.mean([r["oracle_iou_mean"] for r in grp]):.3f}',
            f'{np.mean([r["actual_iou_mean"] for r in grp]):.3f}',
            f'{np.mean([r["iou_lost"] for r in grp]):.3f}',
            f'{n_a}/{len(grp)} ({100*n_a/len(grp):.0f}%)',
            f'{n_a1}/{len(grp)} ({100*n_a1/len(grp):.0f}%)',
        ])
    print(md_table(
        ['Size bin', 'N', 'Oracle mean', 'Actual mean', 'Loss mean', 'Type A share', 'A1 share'],
        rows))

    # Correlation
    pixels = np.array([r['gt_pixels_total'] for r in pp_rows])
    losses = np.array([r['iou_lost'] for r in pp_rows])
    if len(pp_rows) > 2:
        corr = float(np.corrcoef(pixels, losses)[0, 1])
        print(f'\nCorrelation(gt_pixels_total, iou_lost) = **{corr:.3f}**')
        if abs(corr) < 0.15:
            print('→ 객체 크기와 실패 강도는 사실상 무관 (|corr| < 0.15).')
        elif corr < -0.15:
            print('→ 큰 객체일수록 손실 적음 (음의 상관).')
        else:
            print('→ 작은 객체일수록 손실 적음? (이건 의외)')


# Q4: Oracle SP count
def q4_sp_count(pp_rows):
    print('\n## Q4. Oracle SP 개수 — 실패 prompt가 더 많은 SP를 필요로 하나?\n')
    by_type = defaultdict(list)
    for r in pp_rows:
        by_type[r['type_25']].append(r)
    rows = []
    for t in ['A', 'B', 'C']:
        grp = by_type.get(t, [])
        if not grp:
            continue
        counts = [r['oracle_sp_count'] for r in grp]
        rows.append([t, len(grp),
                     f'{np.mean(counts):.2f}',
                     sum(1 for c in counts if c == 1),
                     sum(1 for c in counts if c == 2),
                     sum(1 for c in counts if c == 3)])
    print(md_table(['Type', 'N', 'SP count mean', '1 SP', '2 SP', '3 SP'], rows))

    # A1 only
    A1 = [r for r in pp_rows if r.get('sub_type') == 'A1']
    if A1:
        counts = [r['oracle_sp_count'] for r in A1]
        print(f'\n**A1 (={len(A1)} prompts) Oracle SP count distribution**: '
              f'mean={np.mean(counts):.2f}, '
              f'1 SP={sum(1 for c in counts if c==1)}, '
              f'2 SP={sum(1 for c in counts if c==2)}, '
              f'3 SP={sum(1 for c in counts if c==3)}')


# Q5: A1 prompt commonalities
def q5_a1_commonalities(pp_rows):
    print('\n## Q5. A1 prompt들의 공통점 — 단어 수, 길이 등\n')
    A1 = [r for r in pp_rows if r.get('sub_type') == 'A1']
    C = [r for r in pp_rows if r['type_25'] == 'C']
    if not A1:
        print('A1 = empty')
        return

    def word_count(s): return len(s.split())
    def char_count(s): return len(s)

    a1_wc = [word_count(r['prompt']) for r in A1]
    c_wc = [word_count(r['prompt']) for r in C]
    a1_cc = [char_count(r['prompt']) for r in A1]
    c_cc = [char_count(r['prompt']) for r in C]

    print(md_table(
        ['Group', 'N', 'Word count mean', 'Char count mean',
         'Word count distribution'],
        [['A1', len(A1), f'{np.mean(a1_wc):.2f}', f'{np.mean(a1_cc):.1f}',
          str(dict(sorted({wc: a1_wc.count(wc) for wc in set(a1_wc)}.items())))],
         ['C (success)', len(C), f'{np.mean(c_wc):.2f}', f'{np.mean(c_cc):.1f}',
          str(dict(sorted({wc: c_wc.count(wc) for wc in set(c_wc)}.items())))]]))

    print(f'\n**A1 prompts (12)**: {sorted([r["prompt"] for r in A1])}')
    print(f'\n**C prompts (samples)**: {sorted([r["prompt"] for r in C])[:15]}...')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='output/diagnostics/lerf_ovs_per_prompt.csv')
    parser.add_argument('--out_md', default='md/lerf_ovs_deep_analysis.md')
    args = parser.parse_args()

    rows = load_rows(args.csv)
    pp_rows = aggregate_per_prompt(rows)

    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    print('# LERF-OVS Deep Failure Analysis\n')
    print('> Generated 2026-06-04. Builds on Phase 1 diagnostic. '
          'Answers 5 sub-questions to refine attack hypothesis before Phase 2.\n')
    print('Data source: `output/diagnostics/lerf_ovs_per_prompt.csv` '
          '(208 (scene, prompt, frame) rows, 67 unique prompts).\n')

    q1_frame_stability(rows, pp_rows)
    q2_clip_top1(pp_rows)
    q3_object_size(pp_rows)
    q4_sp_count(pp_rows)
    q5_a1_commonalities(pp_rows)

    sys.stdout = old_stdout
    md = buf.getvalue()
    print(md)
    with open(args.out_md, 'w') as f:
        f.write(md)
    print(f'\n[Wrote {args.out_md}]')


if __name__ == '__main__':
    main()
