"""Serial held-out evaluation; constraint selection never receives target answers."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import load_dataset
from .model import ColorModel, rollout
from .train import atomic_json, tensors


def conflicts(adj: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    anchors = torch.arange(3).expand(len(adj), -1)
    full = torch.cat((predictions, anchors), dim=1)
    edges = adj.bool().triu(1)
    return ((full[:, :, None] == full[:, None, :]) & edges).sum((1, 2))


@torch.no_grad()
def candidates(model: ColorModel, adj: torch.Tensor, count: int, seed: int) -> dict:
    """Candidate 0 greedy; others categorical samples at temperature 1.

    Fixed count, no early exit, no target access; ties keep earliest candidate.
    All candidates run serially so count*n measures actual sample-forward budget.
    """
    if count < 1:
        raise ValueError('count must be positive')
    generator = torch.Generator().manual_seed(seed)
    b, size, _ = adj.shape
    predictions = []
    for k in range(count):
        state = adj.new_zeros((b, size, 3))
        state[:, model.n:] = torch.eye(3)
        present = torch.zeros((b, size), dtype=torch.bool)
        present[:, model.n:] = True
        answer = []
        for t in range(model.n):
            q = model(adj, state, present)[:, t].softmax(-1)
            token = q.argmax(-1) if k == 0 else torch.multinomial(q, 1, generator=generator).squeeze(-1)
            answer.append(token)
            state[:, t] = F.one_hot(token, 3).float()
            present[:, t] = True
        predictions.append(torch.stack(answer, dim=1))
    scores = torch.stack([conflicts(adj, p) for p in predictions], dim=1)
    choices = scores.argmin(1)
    pool = torch.stack(predictions, dim=1)
    return {'prediction': pool[torch.arange(b), choices], 'candidate_conflicts': scores,
            'chosen': choices, 'forward_calls': count * model.n}


def trace_metrics(history: list[dict], target: torch.Tensor, n: int) -> dict:
    first = torch.tensor([history[t]['before'][t] for t in range(n)])
    up = down = changes = returns = opportunities = 0
    previous = first.argmax(-1).tolist()
    penultimate = [None] * n
    for step, entry in enumerate(history):
        before, after = torch.tensor(entry['before']), torch.tensor(entry['after'])
        count = 0 if len(history) > n and step < n else min(step, n)
        # Only revision events; exclude newest position for online policies.
        for i in range(count):
            a, b = int(before[i].argmax()), int(after[i].argmax())
            opportunities += 1
            if a != b:
                changes += 1
                up += int(a != int(target[i]) and b == int(target[i]))
                down += int(a == int(target[i]) and b != int(target[i]))
            if b != previous[i]:
                returns += int(penultimate[i] == b)
                penultimate[i], previous[i] = previous[i], b
        if step < n:
            previous[step] = int(after[step].argmax())
    final = torch.tensor(history[-1]['after'])
    return {'first_proposal_correct_positions': int((first.argmax(-1) == target).sum()),
            'first_proposal_entropy': float(-(first * first.clamp_min(1e-9).log()).sum(-1).mean()),
            'final_entropy': float(-(final * final.clamp_min(1e-9).log()).sum(-1).mean()),
            'wrong_to_right': up, 'right_to_wrong': down, 'old_position_changes': changes,
            'old_position_opportunities': opportunities, 'return_to_previous_choice': returns}


def evaluate_job(directory: Path, split: str = 'test') -> dict:
    result = json.loads((directory / 'result.json').read_text())
    config = result['config']
    data, fingerprint = load_dataset(config['dataset'])
    if fingerprint != result['dataset_sha256']:
        raise RuntimeError('Dataset changed')
    model = ColorModel(n=data['n'], **config['model'])
    best = torch.load(directory / 'best.pt', weights_only=True, map_location='cpu')
    model.load_state_dict(best['model'])
    model.eval()
    adj, target = tensors(data[split])
    mode = result['mode']
    temperature = config.get('soft_temperature', 1.0)
    policies = [1, 2, 4] if mode == 'ar' else [0]
    rows = {}
    with torch.no_grad():
        # Untimed warm-up before each job, no gradient or training changes.
        rollout(model, adj[:1], mode, soft_temperature=temperature)
        for k in policies:
            begin = time.perf_counter()
            predictions, details = [], []
            for i in range(len(adj)):
                a = adj[i:i+1]
                if k:
                    output = candidates(model, a, k, result['seed'] * 100000 + i)
                    prediction = output['prediction'][0]
                    detail = {'candidate_conflicts': output['candidate_conflicts'][0].tolist(),
                              'chosen': int(output['chosen'][0])}
                else:
                    output = rollout(model, a, mode, trace=True,
                                     soft_temperature=temperature)
                    prediction = output['probabilities'][0].argmax(-1)
                    detail = {'trace': output['trace']}
                predictions.append(prediction)
                details.append({'index': i, 'prediction': prediction.tolist(), **detail})
            seconds = time.perf_counter() - begin
            prediction = torch.stack(predictions)
            correct = prediction == target
            if not k:
                for i, detail in enumerate(details):
                    detail['revision_metrics'] = trace_metrics(detail['trace'], target[i], model.n)
            rows[str(k)] = {'examples': len(adj), 'sequence_accuracy': float(correct.all(1).float().mean()),
                            'token_accuracy': float(correct.float().mean()),
                            'constraint_success': float((conflicts(adj, prediction) == 0).float().mean()),
                            'end_to_end_seconds': seconds, 'batch_size': 1,
                            'forward_calls_per_example': output['forward_calls'],
                            'details': details}
    return {'mode': mode, 'seed': result['seed'], 'split': split,
            'best_step': best['step'], 'dataset_sha256': fingerprint,
            'timing_note': 'serial batch=1; includes selection and trace serialization into lists; not pure kernel latency',
            'policies': rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    directory = Path(args.directory)
    output = directory / f'serial_{args.split}.json'
    if output.exists():
        raise RuntimeError('Refusing to overwrite serial evaluation')
    atomic_json(output, evaluate_job(directory, args.split))


if __name__ == '__main__':
    main()
