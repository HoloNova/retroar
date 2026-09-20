"""Single late-evidence probe, not a streaming/immutable-prefix implementation.

All policies receive the same six partial-graph auxiliary losses and six late
position losses. Targets never enter state construction. Waiting policies throw
away the partial draft and decide only after the full graph becomes visible.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import torch
from torch.nn import functional as F

from .data import load_dataset
from .model import ColorModel, belief
from .run_local import available_mib, rss_mib, terminate
from .train import atomic_json, save_checkpoint, tensors

POLICIES = ('wait_ar', 'restart_hard', 'revise_hard', 'revise_soft')


def partial_graph(full: torch.Tensor, n: int) -> torch.Tensor:
    """Hide EVERY free-to-anchor edge, in both directions, independent of labels."""
    early = full.clone()
    early[:, :n, n:] = 0
    early[:, n:, :n] = 0
    return early


def empty_state(adj: torch.Tensor, n: int):
    state = adj.new_zeros((len(adj), n + 3, 3))
    state[:, n:] = torch.eye(3, device=adj.device)
    present = torch.zeros((len(adj), n + 3), dtype=torch.bool, device=adj.device)
    present[:, n:] = True
    return state, present


def sequence(model, adj, revision=True, sample=False, generator=None):
    """Waiting AR: no partial draft is read, six useful late calls."""
    state, present = empty_state(adj, model.n)
    answers = []
    for pos in range(model.n):
        logits = model(adj, state, present, revision=revision)[:, pos]
        prob = logits.softmax(-1)
        color = (torch.multinomial(prob, 1, generator=generator).squeeze(-1)
                 if sample else logits.argmax(-1))
        state = state.clone()
        state[:, pos] = F.one_hot(color, 3).to(adj.dtype)
        present = present.clone()
        present[:, pos] = True
        answers.append(color)
    return torch.stack(answers, dim=1)


def rollout(model, full, policy, target=None, reveal=True):
    if policy not in POLICIES:
        raise ValueError(policy)
    n = model.n
    early = partial_graph(full, n)
    late = full if reveal else early
    state, present = empty_state(full, n)
    early_losses, late_losses = [], []
    soft = policy == 'revise_soft'
    draft = []
    for pos in range(n):
        logits = model(early, state, present, revision=False)[:, pos]
        if target is not None:
            early_losses.append(F.cross_entropy(logits, target[:, pos]))
        draft.append(logits.argmax(-1))
        state = state.clone()
        state[:, pos] = belief(logits, soft)
        present = present.clone()
        present[:, pos] = True
    initial = torch.stack(draft, dim=1)
    if policy in ('wait_ar', 'restart_hard'):
        state, present = empty_state(full, n)
    predictions = []
    trajectory = []
    for pos in range(n):
        logits = model(late, state, present, revision=True)
        # Same position count and weight in ALL policies, as in balanced stage B.
        if target is not None:
            late_losses.append(F.cross_entropy(logits[:, pos], target[:, pos]))
        state = state.clone()
        present = present.clone()
        if policy == 'wait_ar':
            state[:, pos] = belief(logits[:, pos], False)
            present[:, pos] = True
            predictions.append(logits[:, pos].argmax(-1))
        else:
            state[:, :n] = belief(logits, soft)
            present[:, :n] = True
            trajectory.append(logits.argmax(-1).detach())
    prediction = torch.stack(predictions, dim=1) if policy == 'wait_ar' else trajectory[-1]
    loss = (torch.stack(early_losses + late_losses).mean() if target is not None else None)
    return {'prediction': prediction, 'initial': initial, 'trajectory': trajectory,
            'loss': loss, 'training_calls': 2 * n}


def restart_sequence(model, adj):
    """Discarded early computation is not charged as useful inference work."""
    state, present = empty_state(adj, model.n)
    for _ in range(model.n):
        logits = model(adj, state, present, revision=True)
        state = state.clone()
        state[:, :model.n] = belief(logits, False)
        present = present.clone()
        present[:, :model.n] = True
    return logits.argmax(-1)


def conflicts(adj, answers):
    anchors = torch.arange(3, device=adj.device).expand(len(adj), -1)
    full = torch.cat((answers, anchors), dim=1)
    return ((full[:, :, None] == full[:, None, :]) & adj.bool()).sum((1, 2)) // 2


def choose(adj, candidates):
    scores = torch.stack([conflicts(adj, c) for c in candidates], dim=1)
    # Stable tie-break: candidate order, never use target labels.
    index = scores.argmin(-1)
    return torch.stack(candidates, dim=1)[torch.arange(len(adj)), index]


def permute_repair(adj, draft):
    permutations = [torch.tensor(p, device=adj.device)[draft]
                    for p in itertools.permutations(range(3))]
    return choose(adj, permutations)


@torch.no_grad()
def evaluate(model, adj, target, policy, batch=32, diagnostics=False, seed=0):
    model.eval()
    begin = time.monotonic()
    predictions, initials = [], []
    variants = {'frozen_early': [], 'permutation_repair': [], 'no_reveal': []}
    if policy == 'wait_ar':
        variants['wait_k2'] = []
    ups = downs = changes = 0
    generator = torch.Generator().manual_seed(90000 + seed)
    for start in range(0, len(adj), batch):
        a, y = adj[start:start + batch], target[start:start + batch]
        out = rollout(model, a, policy)
        p, initial = out['prediction'], out['initial']
        predictions.append(p)
        initials.append(initial)
        old = initial
        for new in (out['trajectory'] if policy.startswith('revise_') else []):
            was, now = old == y, new == y
            ups += int((~was & now).sum())
            downs += int((was & ~now).sum())
            changes += int((old != new).sum())
            old = new
        if diagnostics:
            variants['frozen_early'].append(initial)
            variants['permutation_repair'].append(permute_repair(a, initial))
            variants['no_reveal'].append(rollout(model, a, policy, reveal=False)['prediction'])
            if policy == 'wait_ar':
                sample = sequence(model, a, sample=True, generator=generator)
                variants['wait_k2'].append(choose(a, [p, sample]))
    prediction = torch.cat(predictions)
    initial = torch.cat(initials)

    def metrics(p):
        right = p == target
        return {'sequence_accuracy': float(right.all(-1).float().mean()),
                'token_accuracy': float(right.float().mean()),
                'constraint_success': float((conflicts(adj, p) == 0).float().mean()),
                'correct_per_example': right.all(-1).tolist()}

    result = {**metrics(prediction), 'examples': len(adj),
              'effective_forward_calls': model.n * (2 if policy.startswith('revise_') else 1),
              'late_forward_calls': model.n,
              'wrong_to_right_events': ups, 'right_to_wrong_events': downs,
              'changed_events': changes,
              'initial_sequence_accuracy': float((initial == target).all(-1).float().mean()),
              'wall_seconds_including_diagnostics': time.monotonic() - begin}
    if diagnostics:
        result['controls'] = {k: metrics(torch.cat(v)) for k, v in variants.items()}
        result['examples_trace'] = [
            {'initial': initial[i].tolist(), 'final': prediction[i].tolist(),
             'target': target[i].tolist()} for i in range(min(3, len(adj)))]
    return result


def fingerprint(config):
    _, digest = load_dataset(config['dataset'])
    files = ('delayed.py', 'model.py', 'train.py', 'data.py', 'run_local.py')
    source = hashlib.sha256(b''.join(
        name.encode() + Path(__file__).with_name(name).read_bytes() for name in files)).hexdigest()
    return {'config': config, 'dataset_sha256': digest, 'source_sha256': source,
            'torch': str(torch.__version__)}


def worker(config, policy, seed, output, test_only=False, resume=False):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = random.Random(seed + 10000)
    data, _ = load_dataset(config['dataset'])
    model = ColorModel(n=data['n'], **config['model'])
    metadata = {**fingerprint(config), 'policy': policy, 'seed': seed}
    output.mkdir(parents=True, exist_ok=True)
    if test_only:
        if json.loads((output / 'metadata.json').read_text()) != metadata:
            raise RuntimeError('Evaluation fingerprint mismatch')
        checkpoint = torch.load(output / 'best.pt', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        a, y = tensors(data[config.get('final_split', 'test')])
        result = evaluate(model, a, y, policy, diagnostics=True, seed=seed)
        # Actual batch-one inference timing, separate from diagnostics, no rival worker.
        times = []
        policies = ['main'] + (['wait_k2'] if policy == 'wait_ar' else [])
        with torch.no_grad():
            for measure in policies:
                samples = []
                gen = torch.Generator().manual_seed(seed)
                for i in range(min(32, len(a))):
                    tick = time.monotonic()
                    if policy == 'wait_ar':
                        p = sequence(model, a[i:i+1])
                        if measure == 'wait_k2':
                            p = choose(a[i:i+1], [p, sequence(model, a[i:i+1], sample=True, generator=gen)])
                    elif policy == 'restart_hard':
                        p = restart_sequence(model, a[i:i+1])
                    else:
                        p = rollout(model, a[i:i+1], policy)['prediction']
                    samples.append(time.monotonic() - tick)
                times.append({'policy': measure, 'median_seconds_per_question': statistics.median(samples)})
        result['serial_timing'] = times
        atomic_json(output / 'serial_test.json', {'metadata': metadata,
                    'checkpoint_step': checkpoint['step'], 'split': config.get('final_split', 'test'),
                    'metrics': result})
        return
    if (output / 'train_result.json').exists():
        raise RuntimeError('Training already complete')
    if (output / 'metadata.json').exists() and not resume:
        raise RuntimeError('Use --resume for partial output')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    a, y = tensors(data['train'])
    va, vy = tensors(data['val'])
    best = -1.0
    start_step = 0
    elapsed_before = 0.0
    if resume:
        saved = torch.load(output / 'last.pt', weights_only=True)
        if saved['metadata'] != metadata:
            raise RuntimeError('Resume fingerprint mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng'])
        torch.set_rng_state(saved['torch_rng'])
        best, start_step = saved['best'], saved['step']
        elapsed_before = saved['elapsed_seconds']
    atomic_json(output / 'metadata.json', metadata)
    begin = time.monotonic()
    step = start_step

    def checkpoint():
        save_checkpoint(output / 'last.pt', {
            'metadata': metadata, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'step': step, 'rng': rng.getstate(), 'torch_rng': torch.get_rng_state(),
            'best': best, 'elapsed_seconds': elapsed_before + time.monotonic() - begin})

    try:
        for step in range(start_step + 1, config['steps'] + 1):
            if time.monotonic() - begin + elapsed_before > config['run_seconds']:
                step -= 1
                raise RuntimeError('Training time limit; partial checkpoint retained')
            if step % 25 == 1 and (available_mib() < 768 or rss_mib(os.getpid()) > 1024):
                step -= 1
                raise RuntimeError('Resource protection; partial checkpoint retained')
            indices = [rng.randrange(len(a)) for _ in range(config['batch_size'])]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = rollout(model, a[indices], policy, y[indices])['loss']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % config['eval_every'] == 0 or step == config['steps']:
                metrics = evaluate(model, va, vy, policy)
                score = metrics['sequence_accuracy']
                if score > best:
                    best = score
                    save_checkpoint(output / 'best.pt', {'model': model.state_dict(), 'step': step})
                checkpoint()
                row = {'step': step, 'val_accuracy': score, 'loss': float(loss.detach()),
                       'elapsed_seconds': elapsed_before + time.monotonic() - begin}
                with (output / 'metrics.jsonl').open('a') as log:
                    log.write(json.dumps(row) + '\n')
                atomic_json(output / 'status.json', {'state': 'training', **row})
                print(policy, seed, row, flush=True)
        checkpoint()
        atomic_json(output / 'train_result.json', {'metadata': metadata, 'completed_steps': step,
                    'best_validation': best, 'elapsed_seconds': elapsed_before + time.monotonic() - begin,
                    'forward_example_calls': step * config['batch_size'] * 2 * data['n']})
        atomic_json(output / 'status.json', {'state': 'trained', 'step': step})
    except BaseException:
        checkpoint()
        raise


def summarize(output, config):
    rows = {}
    for seed in config['seeds']:
        for policy in config['policies']:
            path = output / f'{policy}_seed{seed}' / 'serial_test.json'
            row = json.loads(path.read_text())['metrics']
            rows.setdefault(policy, []).append(row['sequence_accuracy'])
            for name, control in row['controls'].items():
                rows.setdefault(f'{policy}/{name}', []).append(control['sequence_accuracy'])
    summary = {k: {'mean': statistics.mean(v), 'sd': statistics.stdev(v) if len(v)>1 else 0,
                   'by_seed': v} for k, v in rows.items()}
    contrasts = {}
    for a, b in [('revise_hard', 'wait_ar'), ('revise_hard', 'restart_hard'),
                 ('revise_hard', 'wait_ar/wait_k2'), ('revise_soft', 'revise_hard'),
                 ('revise_hard', 'revise_hard/permutation_repair')]:
        if a not in rows or b not in rows:
            continue
        diffs = [x-y for x,y in zip(rows[a], rows[b])]
        contrasts[f'{a} - {b}'] = {'mean_pp': 100*statistics.mean(diffs),
                                  'positive_seeds': sum(d>0 for d in diffs), 'by_seed_pp': [100*d for d in diffs]}
    atomic_json(output / 'summary.json', {'results': summary, 'contrasts': contrasts})
    text = '# 单次延迟证据实验：自动汇总\n\n仅描述性结果，正式结论见后续综合分析。\n\n| 方法 | 整题正确率 | 种子间标准差 |\n|---|---:|---:|\n'
    for key, value in summary.items():
        text += f'| {key} | {100*value["mean"]:.2f}% | {100*value["sd"]:.2f} pp |\n'
    text += '\n## 配对差值\n\n' + json.dumps(contrasts, ensure_ascii=False, indent=2)
    (output / 'report.md').write_text(text)


def suite(config_path, output, resume=False):
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest = fingerprint(config)
    manifest_path = output / 'suite.json'
    if manifest_path.exists():
        if not resume or json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Existing suite requires matching --resume')
    atomic_json(manifest_path, manifest)
    jobs = [(p, s) for s in config['seeds'] for p in config['policies']]
    active = []
    completed = []
    begin = time.monotonic()

    def status(state, **extra):
        atomic_json(output / 'status.json', {'state': state, 'completed_training': completed,
                    'active': [j['name'] for j in active], 'total_training': len(jobs),
                    'elapsed_seconds': time.monotonic()-begin, **extra})

    def launch(policy, seed, test=False):
        name = f'{policy}_seed{seed}'
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, '-m', 'retroar_min.delayed', '--config', str(config_path),
                '--output', str(directory), '--policy', policy, '--seed', str(seed), '--worker']
        if test:
            args.append('--test-only')
        elif resume and (directory / 'last.pt').exists():
            args.append('--resume')
        log = (directory / ('test.log' if test else 'console.log')).open('a')
        child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env={**os.environ,
                                 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        return {'child': child, 'log': log, 'name': name, 'directory': directory}

    def guard():
        if time.monotonic()-begin > config['suite_seconds']:
            raise RuntimeError('Suite wall time limit')
        if available_mib() < 768:
            raise RuntimeError('Host memory protection')
        if any(rss_mib(j['child'].pid) > 1024 for j in active):
            raise RuntimeError('Worker RSS protection')

    try:
        pending = list(jobs)
        while pending or active:
            guard()
            for job in active[:]:
                code = job['child'].poll()
                if code is not None:
                    job['log'].close()
                    active.remove(job)
                    if code != 0:
                        raise RuntimeError(f'{job["name"]} exited {code}; inspect console.log')
                    row = json.loads((job['directory'] / 'train_result.json').read_text())
                    if row['completed_steps'] != config['steps'] or row['metadata']['source_sha256'] != manifest['source_sha256']:
                        raise RuntimeError('Incomplete training or source mismatch')
                    completed.append(job['name'])
            # Admission includes room for one ~300 MiB child above the 768 MiB
            # host floor. Additional concurrent children retain the 1536 MiB gate.
            while (pending and len(active) < config['workers'] and
                   available_mib() >= (1536 if active else 1152)):
                policy, seed = pending.pop(0)
                path = output / f'{policy}_seed{seed}' / 'train_result.json'
                if resume and path.exists():
                    row = json.loads(path.read_text())
                    if row['metadata'] != {**manifest, 'policy': policy, 'seed': seed} or row['completed_steps'] != config['steps']:
                        raise RuntimeError('Resume completed run mismatch')
                    completed.append(f'{policy}_seed{seed}')
                else:
                    active.append(launch(policy, seed))
            status('training' if active else ('waiting_memory' if pending else 'trained'))
            time.sleep(1)
        # No test-set access until ALL training completes. Evaluation is serial.
        for index, (policy, seed) in enumerate(jobs):
            guard()
            active.append(launch(policy, seed, test=True))
            status('serial_evaluation', completed_evaluations=index)
            while active[0]['child'].poll() is None:
                guard()
                time.sleep(1)
            job = active.pop()
            job['log'].close()
            if job['child'].returncode != 0:
                raise RuntimeError(f'{job["name"]} evaluation failed; inspect test.log')
            atomic_json(job['directory'] / 'status.json', {'state': 'completed'})
        if fingerprint(config) != manifest:
            raise RuntimeError('Source/data/config changed during suite')
        summarize(output, config)
        status('completed', completed_evaluations=len(jobs))
    except BaseException as exc:
        for job in active:
            terminate(job['child'])
            job['log'].close()
        status('blocked', error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--policy', choices=POLICIES)
    parser.add_argument('--seed', type=int, default=30)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.config.read_text()), args.policy, args.seed, args.output,
               args.test_only, args.resume)
    else:
        def interrupted(signum, frame):
            raise RuntimeError(f'Supervisor interrupted by signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        suite(args.config.resolve(), args.output.resolve(), args.resume)


if __name__ == '__main__':
    main()
