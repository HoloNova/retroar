"""Post-hoc inference sensitivity probes; no training or model selection.
Not pre-registered evidence or a trained restart_soft baseline.
Run from repository root: .venv/bin/python -m scripts.analyze_delayed_probe
"""
import json
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F
from retroar_min.delayed import rollout
from retroar_min.model import ColorModel
from retroar_min.data import load_dataset
from retroar_min.train import tensors, atomic_json


def main():
    torch.set_num_threads(1)
    root = Path('results/delayed_v1')
    config = json.loads((root/'suite.json').read_text())['config']
    data, fingerprint = load_dataset(config['dataset'])
    adj, target = tensors(data['test'])
    rows = []
    for seed in config['seeds']:
        model = ColorModel(n=data['n'], **config['model']).eval()
        checkpoint = torch.load(root/f'revise_soft_seed{seed}/best.pt', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        for treatment in ('unchanged', 'uniform', 'harden', 'other_question', 'zeros'):
            counts = 0
            entropies, maxima, distances = [], [], []
            for start in range(0, len(adj), 32):
                seen = [False]
                def intervene(module, args, kwargs):
                    if not kwargs.get('revision', False) or seen[0]:
                        return None
                    seen[0] = True
                    graph, state, present = args
                    n = model.n
                    p = state[:, :n]
                    entropies.append(float(-(p*p.clamp_min(1e-12).log()).sum(-1).mean()))
                    maxima.append(float(p.max(-1).values.mean()))
                    distances.append(float((p-1/3).abs().mean()))
                    replacement = state.clone()
                    if treatment == 'uniform':
                        replacement[:, :n] = 1/3
                    elif treatment == 'harden':
                        replacement[:, :n] = F.one_hot(p.argmax(-1), 3).to(p.dtype)
                    elif treatment == 'other_question':
                        replacement[:, :n] = p.roll(1, dims=0)
                    elif treatment == 'zeros':
                        replacement[:, :n] = 0
                    return (graph, replacement, present), kwargs
                hook = model.register_forward_pre_hook(intervene, with_kwargs=True)
                with torch.no_grad():
                    p = rollout(model, adj[start:start+32], 'revise_soft')['prediction']
                hook.remove()
                counts += int((p == target[start:start+32]).all(-1).sum())
            row = {'seed':seed,'treatment':treatment,'accuracy':counts/len(adj),
                   'early_state_entropy':statistics.mean(entropies),
                   'early_mean_max_probability':statistics.mean(maxima),
                   'early_mean_abs_distance_uniform':statistics.mean(distances)}
            rows.append(row)
            print(row, flush=True)
    result = {'post_hoc':True,'split':'test','dataset_sha256':fingerprint,
              'caveat':'Frozen trained soft models; one state replacement at evidence arrival; present flags retained. Distribution shift; cannot substitute for training restart_soft or independent replication.',
              'rows':rows,
              'means':{t:statistics.mean(r['accuracy'] for r in rows if r['treatment']==t)
                       for t in ('unchanged','uniform','harden','other_question','zeros')}}
    for seed in config['seeds']:
        original=json.loads((root/f'revise_soft_seed{seed}/serial_test.json').read_text())['metrics']['sequence_accuracy']
        assert next(r['accuracy'] for r in rows if r['seed']==seed and r['treatment']=='unchanged') == original
    atomic_json(root/'posthoc_state_probe.json', result)
    print(result['means'])


if __name__ == '__main__':
    main()
