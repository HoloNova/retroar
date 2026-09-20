"""Bounded CPU training with validation selection, resumable state, and JSON logs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import time
from pathlib import Path

# Set before importing torch, so BLAS does not silently occupy all host cores.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import torch
from torch.nn import functional as F

from .data import load_dataset
from .model import ColorModel, MODES, POSTHOC, REVISING, rollout


def forward_calls_per_example(n: int, mode: str) -> int:
    return n * (2 if mode in REVISING or mode in POSTHOC or mode == 'ar_matched' else 1)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def tensors(records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor([r['adj'] for r in records], dtype=torch.float32),
            torch.tensor([r['target'] for r in records], dtype=torch.long))


@torch.no_grad()
def evaluate(model: ColorModel, adj: torch.Tensor, target: torch.Tensor,
             mode: str, batch_size: int, soft_temperature: float = 1.0) -> dict:
    model.eval()
    begin = time.monotonic()
    right = exact = valid = total = 0
    ce = brier = 0.0
    entropy_sum = 0.0
    up = down = opportunities = 0
    n = model.n
    for start in range(0, len(adj), batch_size):
        a, y = adj[start:start + batch_size], target[start:start + batch_size]
        result = rollout(model, a, mode, target=y, soft_temperature=soft_temperature)
        q = result['probabilities']
        prediction = q.argmax(-1)
        correct = prediction == y
        right += int(correct.sum())
        exact += int(correct.all(-1).sum())
        anchors = torch.arange(3).expand(len(a), -1)
        full = torch.cat((prediction, anchors), dim=1)
        conflict = (full[:, :, None] == full[:, None, :]) & a.bool()
        valid += int((~conflict.flatten(1).any(-1)).sum())
        ce += float(-q.gather(-1, y.unsqueeze(-1)).clamp_min(1e-9).log().sum())
        brier += float(((q - F.one_hot(y, 3)) ** 2).sum())
        entropy_sum += float(-(q.clamp_min(1e-9).log() * q).sum(-1).sum())
        total += len(a)
        up += result['wrong_to_right']
        down += result['right_to_wrong']
        opportunities += result['revision_opportunities']
    return {'examples': total, 'token_accuracy': right / (total * n),
            'sequence_accuracy': exact / total, 'constraint_success': valid / total,
            'final_cross_entropy': ce / (total * n), 'brier': brier / (total * n),
            'wrong_to_right': up, 'right_to_wrong': down,
            'mean_final_entropy': entropy_sum / (total * n),
            'revision_opportunities': opportunities,
            'net_revision_rate': (up - down) / opportunities if opportunities else None,
            'inference_seconds': time.monotonic() - begin,
            'forward_calls_per_example': forward_calls_per_example(n, mode)}


def run(config: dict, mode: str, seed: int, output: Path, resume: bool = False) -> dict:
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = random.Random(seed + 10000)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise RuntimeError('Completed output exists; use the suite resume logic or a new directory')
    if (output / 'metrics.jsonl').exists() and not resume:
        raise RuntimeError('Partial output exists; explicitly resume or use a new directory')
    data, fingerprint = load_dataset(config['dataset'])
    model = ColorModel(n=data['n'], **config['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    initial_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes()
                                          for p in model.parameters())).hexdigest()
    train_adj, train_target = tensors(data['train'])
    val_adj, val_target = tensors(data['val'])
    test_adj, test_target = tensors(data['test'])
    soft_temperature = config.get('soft_temperature', 1.0)
    best_score = (-1.0, -math.inf)
    best_step = 0
    start_step = 0
    elapsed_before = 0.0
    training_seconds = 0.0
    last = output / 'last.pt'
    metadata = {'config': config, 'mode': mode, 'seed': seed,
                'dataset_sha256': fingerprint, 'torch': str(torch.__version__),
                'parameters': sum(p.numel() for p in model.parameters()),
                'initial_weights_sha256': initial_hash,
                'feedback_gradient': 'detached in every mode',
                'selection': 'validation sequence accuracy, then final cross entropy',
                'source_sha256': hashlib.sha256(b''.join(
                    p.name.encode() + p.read_bytes()
                    for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()}
    if resume and last.exists():
        checkpoint = torch.load(last, map_location='cpu', weights_only=True)
        if checkpoint['metadata'] != metadata:
            raise RuntimeError('Resume configuration/dataset/environment does not match checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        rng.setstate(checkpoint['random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        best_score = tuple(checkpoint['best_score'])
        best_step = checkpoint['best_step']
        start_step = checkpoint['step']
        elapsed_before = checkpoint['elapsed_seconds']
        training_seconds = checkpoint['training_seconds']
    elif resume and (output / 'metrics.jsonl').exists():
        raise RuntimeError('No checkpoint to resume; refusing to overwrite partial logs')
    atomic_json(output / 'metadata.json', metadata)
    begin = time.monotonic()
    stop_reason = 'max_steps'
    completed = start_step

    def checkpoint_at(step: int) -> None:
        save_checkpoint(last, {
            'metadata': metadata, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'step': step,
            'random_state': rng.getstate(), 'torch_random_state': torch.get_rng_state(),
            'best_score': best_score, 'best_step': best_step,
            'elapsed_seconds': elapsed_before + time.monotonic() - begin,
            'training_seconds': training_seconds,
        })

    with (output / 'metrics.jsonl').open('a', encoding='utf-8', buffering=1) as log:
        if start_step == 0:
            baseline = evaluate(model, val_adj, val_target, mode, config['eval_batch'],
                                soft_temperature=soft_temperature)
            best_score = (baseline['sequence_accuracy'], -baseline['final_cross_entropy'])
            save_checkpoint(output / 'best.pt', {'model': model.state_dict(), 'step': 0})
            log.write(json.dumps({'step': 0, 'validation': baseline}) + '\n')
            checkpoint_at(0)
        loss_sum = 0.0
        loss_count = 0
        for step in range(start_step + 1, config['steps'] + 1):
            if elapsed_before + time.monotonic() - begin >= config['max_seconds']:
                stop_reason = 'time_budget'
                break
            tick = time.monotonic()
            indices = [rng.randrange(len(train_adj)) for _ in range(config['batch_size'])]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result = rollout(model, train_adj[indices], mode, train_target[indices],
                             supervision=config.get('supervision', 'legacy'),
                             soft_temperature=soft_temperature)
            loss = result['loss']
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach())
            loss_count += 1
            training_seconds += time.monotonic() - tick
            completed = step
            if step % config['eval_every'] == 0 or step == config['steps']:
                metrics = evaluate(model, val_adj, val_target, mode, config['eval_batch'],
                                   soft_temperature=soft_temperature)
                score = (metrics['sequence_accuracy'], -metrics['final_cross_entropy'])
                if score > best_score:
                    best_score, best_step = score, step
                    save_checkpoint(output / 'best.pt', {'model': model.state_dict(), 'step': step})
                row = {'step': step, 'mean_loss': loss_sum / loss_count,
                       'validation': metrics, 'training_seconds': training_seconds,
                       'elapsed_seconds': elapsed_before + time.monotonic() - begin,
                       'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                       'training_forward_example_calls': step * config['batch_size'] *
                       forward_calls_per_example(data['n'], mode)}
                log.write(json.dumps(row) + '\n')
                checkpoint_at(step)
                atomic_json(output / 'status.json', {'state': 'training', 'mode': mode,
                            'seed': seed, 'step': step, 'best_step': best_step,
                            'validation_sequence_accuracy': metrics['sequence_accuracy']})
                print(f'{mode} seed={seed} step={step} '
                      f'val={metrics["sequence_accuracy"]:.3f} '
                      f'loss={row["mean_loss"]:.3f} rss={row["peak_rss_mib"]:.0f}MiB', flush=True)
                loss_sum = 0.0
                loss_count = 0
        # Always evaluate the final completed update, even after a time limit.
        if completed > start_step and completed % config['eval_every'] and completed != config['steps']:
            metrics = evaluate(model, val_adj, val_target, mode, config['eval_batch'],
                               soft_temperature=soft_temperature)
            score = (metrics['sequence_accuracy'], -metrics['final_cross_entropy'])
            if score > best_score:
                best_score, best_step = score, completed
                save_checkpoint(output / 'best.pt', {'model': model.state_dict(), 'step': completed})
            log.write(json.dumps({'step': completed, 'validation': metrics,
                                  'reason': 'final_partial_interval'}) + '\n')
        checkpoint_at(completed)

    best = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['model'])
    evaluation_split = 'test' if config.get('evaluate_test', True) else 'val'
    eval_adj, eval_target = (test_adj, test_target) if evaluation_split == 'test' else (val_adj, val_target)
    evaluation = evaluate(model, eval_adj, eval_target, mode, config['eval_batch'],
                          soft_temperature=soft_temperature)
    examples = []
    with torch.no_grad():
        for i in range(min(3, len(eval_adj))):
            sample = rollout(model, eval_adj[i:i + 1], mode, trace=True,
                             soft_temperature=soft_temperature)
            examples.append({'adj': data[evaluation_split][i]['adj'], 'target': data[evaluation_split][i]['target'],
                             'prediction': sample['probabilities'][0].argmax(-1).tolist(),
                             'trace': sample['trace']})
    atomic_json(output / 'traces.json', {'examples': examples})
    summary = {**metadata, 'completed_steps': completed, 'best_step': best_step,
               'stop_reason': stop_reason, 'training_seconds': training_seconds,
               'elapsed_seconds': elapsed_before + time.monotonic() - begin,
               'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               'evaluation_split': evaluation_split, 'evaluation': evaluation}
    atomic_json(output / 'result.json', summary)
    atomic_json(output / 'status.json', {'state': 'completed', 'mode': mode, 'seed': seed,
                                       'step': completed, 'stop_reason': stop_reason})
    print(json.dumps({'completed': str(output), evaluation_split: evaluation}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.mode, args.seed,
        Path(args.output), args.resume)


if __name__ == '__main__':
    main()
