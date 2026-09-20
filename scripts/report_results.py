"""Summarize completed runs without importing torch or touching model parameters."""
import argparse
import json
import statistics
from pathlib import Path

NAMES = {
    'ar': '普通生成', 'ar_matched': '普通生成（严格额外计算）',
    'extra_compute': '多算一次、不改历史',
    'hard': '在线确定草稿回改', 'soft': '在线概率草稿回改',
    'soft_feedback': '概率草稿、不回改',
    'posthoc_hard': '写完后确定草稿回改',
    'posthoc_soft': '写完后概率草稿回改',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    root = Path(args.root)
    records = [json.loads(p.read_text()) for p in sorted(root.glob('*/result.json'))]
    if not records:
        raise SystemExit('No completed runs')
    for field in ('dataset_sha256', 'source_sha256', 'evaluation_split', 'parameters'):
        if len({r[field] for r in records}) != 1:
            raise SystemExit(f'Inconsistent {field}; refusing to merge unlike experiments')
    for seed in {r['seed'] for r in records}:
        if len({r['initial_weights_sha256'] for r in records if r['seed'] == seed}) != 1:
            raise SystemExit(f'Initial weights differ for seed {seed}')
    split = records[0]['evaluation_split']
    lines = [f'# 实验报告：{root.name}', '',
             f'- 评估划分：`{split}`；每版参数量：{records[0]["parameters"]}。',
             f'- 数据 SHA256：`{records[0]["dataset_sha256"]}`。',
             f'- 训练源码 SHA256：`{records[0]["source_sha256"]}`。',
             '- 检查点仅按验证集选择。均值±标准差表示种子间差异，不代表统计显著性。', '',
             '| 版本 | 完成次数 | 整题准确率 | 位置准确率 | 平均训练秒数 | 峰值内存 MiB | 每题前向次数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    grouped = {}
    for mode in NAMES:
        group = [r for r in records if r['mode'] == mode]
        if not group:
            continue
        grouped[mode] = group
        values = [r['evaluation']['sequence_accuracy'] * 100 for r in group]
        sd = f' ± {statistics.stdev(values):.2f}' if len(values) > 1 else '（单次）'
        lines.append(f'| {NAMES[mode]} | {len(group)} | {statistics.mean(values):.2f}{sd}% '
                     f'| {statistics.mean(r["evaluation"]["token_accuracy"] for r in group) * 100:.2f}% '
                     f'| {statistics.mean(r["training_seconds"] for r in group):.1f} '
                     f'| {max(r["peak_rss_mib"] for r in group):.1f} '
                     f'| {group[0]["evaluation"]["forward_calls_per_example"]} |')
    lines += ['', '## 完成度与修订效果', '']
    for mode, group in grouped.items():
        for r in group:
            m = r['evaluation']
            lines.append(f'- {NAMES[mode]} / seed={r["seed"]}：完成 {r["completed_steps"]} 步，'
                         f'选中第 {r["best_step"]} 步，停止原因 `{r["stop_reason"]}`；'
                         f'旧位置错改对 {m["wrong_to_right"]} 次、对改错 {m["right_to_wrong"]} 次，'
                         f'总回看机会 {m["revision_opportunities"]}。')
    lines += ['', '## 解释限制', '',
              '- 普通生成与概率不回改的每步调用较少；不能只比较准确率就宣称同计算预算优势。',
              '- 多算一次、确定回改和概率回改具有相同主要网络调用次数，仍需结合实际耗时比较。',
              '- 同一位置可能被多次修订，修订次数不是独立样本数。',
              '- 这是小图唯一解任务；结果不能直接推广到自然语言，也不能证明方法首创。']
    if split != 'test':
        lines.append('- 本报告仅为验证集探索结果，不是正式测试结论。')
    if any(r['stop_reason'] != 'max_steps' for r in records):
        lines.append('- 存在未跑完预定步数的版本，需核对预算和中断原因后比较。')
    status = root / 'status.json'
    if status.exists() and json.loads(status.read_text())['state'] != 'completed':
        lines.append('- 整套实验尚未全部完成，本报告仅包含已完成部分。')
    path = root / 'report.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
