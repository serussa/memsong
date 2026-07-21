#!/usr/bin/env python3
"""
Long-form PER Evaluation — 全局音素对齐 + 分段统计。

改进自 Muse calc_per.py：
  - 不使用截断（原版按 ref/hyp 字符数最小值截断）。
  - 一次全局 Levenshtein 对齐得到整句音素操作序列。
  - 按参考音素位置将歌词均分 Early / Middle / Late 三段。
  - 在同一次对齐中分别统计每段的 S/D/I，得到段级 PER。
  - Insertion 根据相邻参考位置归入对应分段。

用法:
  python calc_per_long.py
      --hyp_file    <ASR 转写 JSONL>
      --gt_file     <GT 歌词 JSONL (zh.jsonl / en.jsonl)>
      --output_dir  <输出目录>
      --model_name  <模型名>

输出:
  {output_dir}/songs.jsonl    每首歌的明细（PER / 段级 PER / 音素细节）
  {output_dir}/summary.csv    模型级汇总（均值 ± 标准差）

测试:
  python calc_per_long.py --test
"""

import argparse
import csv
import json
import os
import re
import sys
from statistics import mean, stdev

# 复用 Muse 管线的音素转换函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import phoneme_utils


# ================================================================
#  信号过滤（与原始 calc_per.py 完全一致，仅供对照时对齐输入）
# ================================================================
def signal_filter(text: str) -> str:
    """去除标点，统一转为空格。"""
    pattern = r'[ ,。"，:;&—''\'.\]\[()?\n-]'
    text = re.sub(pattern, ' ', text)
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


# ================================================================
#  全局 Levenshtein 对齐（带完整回溯）
# ================================================================
def levenshtein_alignment(ref, hyp):
    """
    Compute full DP matrix and traceback.

    Returns:
        list of (op_type, ref_idx, hyp_idx) in FORWARD order.
        op_type: 'M'  match / 'S'  substitution / 'D'  deletion / 'I'  insertion
        ref_idx/hyp_idx: -1 when not applicable (I / D respectively).
    """
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        ri = ref[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,   # match / substitution
                dp[i - 1][j] + 1,           # deletion
                dp[i][j - 1] + 1,           # insertion
            )

    # Traceback  →  forward-order ops
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(('M' if ref[i - 1] == hyp[j - 1] else 'S', i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('D', i - 1, -1))
            i -= 1
        else:
            ops.append(('I', -1, j - 1))
            j -= 1

    ops.reverse()
    return ops


# ================================================================
#  分段归属
# ================================================================
def _which_segment(ref_idx, early_end, mid_end):
    if ref_idx < early_end:
        return 'Early'
    elif ref_idx < mid_end:
        return 'Middle'
    return 'Late'


def segment_ops(ops, ref_len):
    """
    将一次全局对齐的操作列表按参考音素位置分入 Early / Middle / Late。

    - M/S/D  直接由 ref_idx 定位。
    - I      根据相邻参考位置归属：
        · 全部参考音素之前  → Early
        · 全部参考音素之后  → Late
        · 两参考音素之间    → 归入前一个参考音素所在段

    Returns:
        seg_counts:   {'Early': {'S':int,'D':int,'I':int}, …}
        seg_ref_cnt:  {'Early': int, …}  每段的参考音素数
    """
    ret = {s: {'S': 0, 'D': 0, 'I': 0} for s in ['Early', 'Middle', 'Late']}
    ref_cnt = {s: 0 for s in ['Early', 'Middle', 'Late']}

    if ref_len == 0:
        return ret, ref_cnt

    early_end = ref_len // 3
    mid_end = 2 * ref_len // 3

    for r in range(ref_len):
        ref_cnt[_which_segment(r, early_end, mid_end)] += 1

    cur_ref = 0  # 下一个待消费的参考音素下标
    for op, r_idx, _ in ops:
        if op in ('M', 'S'):
            if op == 'S':
                ret[_which_segment(r_idx, early_end, mid_end)]['S'] += 1
            cur_ref = r_idx + 1
        elif op == 'D':
            ret[_which_segment(r_idx, early_end, mid_end)]['D'] += 1
            cur_ref = r_idx + 1
        else:  # 'I'
            if cur_ref == 0:
                ret['Early']['I'] += 1                          # 开头多余的
            elif cur_ref >= ref_len:
                ret['Late']['I'] += 1                            # 结尾多余的
            else:
                ret[_which_segment(cur_ref - 1, early_end, mid_end)]['I'] += 1

    return ret, ref_cnt


# ================================================================
#  PER 计算
# ================================================================
def compute_per_from_ops(ops, ref_len):
    """从全局对齐操作计算整体 PER = (S + D + I) / N。"""
    if ref_len == 0:
        return 0.0
    errors = sum(1 for op, _, _ in ops if op != 'M')
    return errors / ref_len


def compute_segment_per(seg_counts, seg_ref_cnt):
    """计算每段 PER。"""
    per = {}
    for seg in ['Early', 'Middle', 'Late']:
        n = seg_ref_cnt[seg]
        if n == 0:
            per[seg] = 0.0
        else:
            c = seg_counts[seg]
            per[seg] = (c['S'] + c['D'] + c['I']) / n
    return per


# ================================================================
#  单曲主入口
# ================================================================
def compute_long_per(ref_text, hyp_text):
    """
    全流水线：过滤 → 音素转换 → 全局对齐 → 分段统计。

    Returns:
        dict with keys:
          ref_phonemes, hyp_phonemes (list),
          alignment_ops,
          overall_per, original_per,
          seg_counts, seg_ref_cnt, seg_per,
          ldg  (Late - Early)
    """
    ref_ph = phoneme_utils.get_phonemes(ref_text, with_sp=False, remove_tones=True)
    hyp_ph = phoneme_utils.get_phonemes(hyp_text, with_sp=False, remove_tones=True)

    ops = levenshtein_alignment(ref_ph, hyp_ph)

    overall = compute_per_from_ops(ops, len(ref_ph))
    original = phoneme_utils.calc_per(ref_ph, hyp_ph)

    seg_counts, seg_ref_cnt = segment_ops(ops, len(ref_ph))
    seg_per = compute_segment_per(seg_counts, seg_ref_cnt)

    return {
        'ref_phonemes': ref_ph,
        'hyp_phonemes': hyp_ph,
        'alignment_ops': ops,
        'overall_per': overall,
        'original_per': original,
        'seg_counts': seg_counts,
        'seg_ref_cnt': seg_ref_cnt,
        'seg_per': seg_per,
        'ldg': seg_per['Late'] - seg_per['Early'],
    }


# ================================================================
#  批量匹配 GT + ASR
# ================================================================
def extract_idx(filename):
    """从文件名提取末尾的数字序号。"""
    matches = re.findall(r'\d+', os.path.splitext(filename)[0])
    return int(matches[-1]) if matches else None


def match_gt_and_hyp(gt_file, hyp_file, offset=0):
    """
    匹配 GT 歌词与 ASR 转写结果。

    Returns:
        list of (file_index, ref_text, hyp_text, file_name)
    """
    gt = {}
    with open(gt_file, encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                idx = rec.get('file_index')
                if idx is not None:
                    gt[idx] = rec['lyrics']
            except Exception:
                continue

    hyp = {}
    info = {}
    with open(hyp_file, encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                idx = rec.get('file_idx')
                if idx is None:
                    idx = extract_idx(rec.get('file_name', ''))
                if idx is not None:
                    hyp[idx] = rec.get('hyp_text', '')
                    info[idx] = rec.get('file_name', '')
            except Exception:
                continue

    pairs = []
    for idx in sorted(gt):
        gt_idx = idx + offset
        if gt_idx in hyp:
            pairs.append((
                idx,
                signal_filter(gt[idx]),
                signal_filter(hyp[gt_idx]),
                info.get(gt_idx, ''),
            ))
    return pairs


# ================================================================
#  CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Long-form PER: segment-level phoneme error rate from global alignment')
    parser.add_argument('--hyp_file', help='ASR transcription JSONL')
    parser.add_argument('--gt_file', help='Ground truth lyrics JSONL (zh.jsonl / en.jsonl)')
    parser.add_argument('--model_name', default='model', help='Model name for CSV')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--offset', type=int, default=0, help='Index offset between GT and ASR')
    parser.add_argument('--test', action='store_true', help='Run minimal test and exit')
    args = parser.parse_args()

    if args.test:
        run_minimal_test()
        return

    # 参数检查
    if not all([args.hyp_file, args.gt_file, args.output_dir]):
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 匹配 ──
    pairs = match_gt_and_hyp(args.gt_file, args.hyp_file, args.offset)
    if not pairs:
        print('No matching pairs found between GT and ASR.')
        return
    print(f'Matched {len(pairs)} pairs')

    # ── 逐曲评估 ──
    songs = []
    metrics = {k: [] for k in ['overall_per', 'early_per', 'middle_per', 'late_per', 'ldg', 'original_per']}

    for idx, ref_text, hyp_text, fname in pairs:
        try:
            res = compute_long_per(ref_text, hyp_text)
            record = {
                'file_name': fname,
                'file_index': idx,
                'overall_per': round(res['overall_per'], 4),
                'early_per':   round(res['seg_per']['Early'], 4),
                'middle_per':  round(res['seg_per']['Middle'], 4),
                'late_per':    round(res['seg_per']['Late'], 4),
                'ldg':         round(res['ldg'], 4),
                'original_per': round(res['original_per'], 4),
                'ref_phonemes':  ' '.join(res['ref_phonemes']),
                'hyp_phonemes':  ' '.join(res['hyp_phonemes']),
                'ref_text':    ref_text,
                'hyp_text':    hyp_text,
            }
            songs.append(record)

            for k in metrics:
                if k == 'early_per':
                    metrics[k].append(res['seg_per']['Early'])
                elif k == 'middle_per':
                    metrics[k].append(res['seg_per']['Middle'])
                elif k == 'late_per':
                    metrics[k].append(res['seg_per']['Late'])
                elif k == 'ldg':
                    metrics[k].append(res['ldg'])
                elif k == 'original_per':
                    metrics[k].append(res['original_per'])
                else:
                    metrics[k].append(res['overall_per'])
        except Exception as e:
            print(f'Error idx={idx} ({fname}): {e}', file=sys.stderr)
            continue

    if not songs:
        print('No songs successfully evaluated.')
        return

    # ── 输出 per-song JSONL ──
    songs_path = os.path.join(args.output_dir, 'songs.jsonl')
    with open(songs_path, 'w', encoding='utf-8') as f:
        for s in songs:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'Per-song details → {songs_path}')

    # ── 输出汇总 CSV ──
    def _sd(v):
        return stdev(v) if len(v) >= 2 else 0.0

    csv_path = os.path.join(args.output_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'metric', 'mean', 'std', 'count'])
        for mk in ['overall_per', 'early_per', 'middle_per', 'late_per', 'ldg', 'original_per']:
            v = metrics[mk]
            w.writerow([args.model_name, mk, f'{mean(v):.4f}', f'{_sd(v):.4f}', len(v)])
    print(f'Summary CSV     → {csv_path}')

    # ── 终端打印 ──
    print()
    print(f'====== {args.model_name}  ({len(songs)} songs) ======')
    for mk, label in [('overall_per', 'Overall PER'),
                      ('early_per',   'Early PER'),
                      ('middle_per',  'Middle PER'),
                      ('late_per',    'Late PER'),
                      ('ldg',         'LDG (Late − Early)'),
                      ('original_per','Original PER (Muse)')]:
        v = metrics[mk]
        print(f'  {label:25s}  {mean(v):.4f} ± {_sd(v):.4f}')


# ================================================================
#  最小测试
# ================================================================
def run_minimal_test():
    """验证全局对齐与分段逻辑的正确性。"""
    print('=' * 60)
    print('Minimal test: long-form PER')
    print('=' * 60)

    cases = [
        # (desc, ref, hyp)
        ('exact match',       'Hello World',       'Hello World'),
        ('one substitution',  'Hello World',       'Hello Word'),
        ('insertion middle',  'Hello World',       'Hello Big World'),
        ('deletion end',      'Hello World',       'Hello'),
        ('empty hyp',         'Hello',             ''),
        ('empty ref',         '',                   'Hello'),
        ('twice as long hyp', 'Hello World',       'Hello Hello World World'),
        ('all wrong',         'Hello World',       'Nope Nada'),
    ]

    for desc, ref, hyp in cases:
        res = compute_long_per(ref, hyp)
        rp = res['ref_phonemes']
        hp = res['hyp_phonemes']
        print(f'\n[{desc}]')
        print(f'  ref({len(rp):2d}): {" ".join(rp)}')
        print(f'  hyp({len(hp):2d}): {" ".join(hp)}')
        print(f'  Overall={res["overall_per"]:.4f}  '
              f'E={res["seg_per"]["Early"]:.4f}  '
              f'M={res["seg_per"]["Middle"]:.4f}  '
              f'L={res["seg_per"]["Late"]:.4f}  '
              f'LDG={res["ldg"]:+.4f}')
        for seg in ['Early', 'Middle', 'Late']:
            c = res['seg_counts'][seg]
            print(f'    {seg:8s}: S={c["S"]} D={c["D"]} I={c["I"]}  '
                  f'N={res["seg_ref_cnt"][seg]}')

    # 与原始 calc_per 交叉验证
    print('\n--- Cross-validation vs original calc_per ---')
    for desc, ref, hyp in cases:
        res = compute_long_per(ref, hyp)
        ok = '✓' if abs(res['overall_per'] - res['original_per']) < 1e-10 else '✗'
        print(f'  {desc:20s}  overall={res["overall_per"]:.4f}  '
              f'original={res["original_per"]:.4f}  {ok}')

    print('\nDone.')


if __name__ == '__main__':
    main()
