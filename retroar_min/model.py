"""One shared architecture, several decoding policies, no gold-prefix feedback."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

MODES = ('ar', 'ar_matched', 'extra_compute', 'hard', 'soft',
         'soft_feedback', 'posthoc_hard', 'posthoc_soft')
REVISING = ('extra_compute', 'hard', 'soft')
POSTHOC = ('posthoc_hard', 'posthoc_soft')


class ColorModel(nn.Module):
    def __init__(self, n: int = 6, width: int = 32, heads: int = 2,
                 ff: int = 64, layers: int = 1):
        super().__init__()
        self.n = n
        size = n + 3
        self.graph = nn.Linear(size, width)
        self.color = nn.Linear(3, width, bias=False)
        self.known = nn.Embedding(2, width)
        self.position = nn.Embedding(size, width)
        self.role = nn.Embedding(2, width)
        layer = nn.TransformerEncoderLayer(width, heads, ff, dropout=0.0,
                                           batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Linear(width, 3)

    def forward(self, adj: torch.Tensor, state: torch.Tensor,
                present: torch.Tensor, revision: bool = False) -> torch.Tensor:
        positions = torch.arange(self.n + 3, device=adj.device)
        hidden = (self.graph(adj) + self.color(state) + self.known(present.long())
                  + self.position(positions) + self.role.weight[int(revision)])
        return self.output(self.encoder(hidden))[:, :self.n]


def belief(logits: torch.Tensor, soft: bool, temperature: float = 1.0) -> torch.Tensor:
    """Uncommitted draft state. temperature>1 keeps a grade of belief instead of a spike.

    Gradients are stopped in both branches, so the carried state never becomes a
    differentiable recurrent path. The reported probabilities stay at T=1, so the
    eventual answer (argmax) is unaffected by temperature and only the draft
    representation fed back into the model changes.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if not soft:
        return F.one_hot(logits.argmax(-1), 3).to(logits.dtype)
    if temperature == 1.0:
        return feedback(logits.softmax(-1), True)
    return (logits / temperature).softmax(-1).detach()


def feedback(q: torch.Tensor, soft: bool) -> torch.Tensor:
    # Both hard and soft policies stop gradients through carried state.
    # Thus the comparison does not silently add a differentiable recurrent path.
    q = q.detach()
    return q if soft else F.one_hot(q.argmax(-1), 3).to(q.dtype)


def rollout(model: ColorModel, adj: torch.Tensor, mode: str,
            target: torch.Tensor | None = None, trace: bool = False,
            supervision: str = 'legacy', soft_temperature: float = 1.0) -> dict:
    """Targets are used only for loss/statistics, never to build model input.

    The complete input graph is always visible. 'Causal' means the sequence of
    proposed colors grows left-to-right, not that graph conditions are hidden.
    All modes have exactly the same parameterization.
    """
    if supervision not in ('legacy', 'balanced'):
        raise ValueError(f'Unknown supervision: {supervision}')
    if mode not in MODES:
        raise ValueError(f'Unknown mode: {mode}')
    batch, size, _ = adj.shape
    n = model.n
    state = adj.new_zeros((batch, size, 3))
    state[:, n:] = torch.eye(3, device=adj.device)
    present = torch.zeros((batch, size), dtype=torch.bool, device=adj.device)
    present[:, n:] = True
    q = adj.new_zeros((batch, n, 3))
    losses = []
    history = []
    wrong_to_right = right_to_wrong = attempts = 0
    if soft_temperature <= 0:
        raise ValueError('soft_temperature must be positive')
    soft = mode in ('soft', 'soft_feedback', 'posthoc_soft')

    for t in range(n):
        logits = model(adj, state, present)
        proposal = logits[:, t].softmax(-1)
        if target is not None:
            losses.append(F.cross_entropy(logits[:, t], target[:, t]))
        if mode == 'ar_matched':
            # Burn an identical second forward pass without changing the state.
            # This isolates raw extra compute from revision or extra supervision.
            matched_logits = model(adj, state, present)
            if target is not None:
                losses.append(F.cross_entropy(matched_logits[:, t], target[:, t]))
        q = q.clone()
        q[:, t] = proposal.detach()
        state = state.clone()
        state[:, t] = belief(logits[:, t], soft, soft_temperature)
        present = present.clone()
        present[:, t] = True
        before = q[:, :t + 1].clone()

        if mode in REVISING:
            revised = model(adj, state, present, revision=True)[:, :t + 1]
            probabilities = revised.softmax(-1)
            if target is not None:
                # extra_compute has the same auxiliary supervision as hard/soft,
                # but cannot replace earlier proposals with its new predictions.
                if supervision == 'balanced':
                    # Each position receives exactly one revision loss, weight 1/(2*n).
                    losses.append(F.cross_entropy(revised[:, t], target[:, t]))
                else:
                    losses.append(F.cross_entropy(revised.reshape(-1, 3),
                                                   target[:, :t + 1].reshape(-1)))
            if mode == 'extra_compute':
                q[:, t] = probabilities[:, t].detach()
                state = state.clone()
                state[:, t] = belief(revised[:, t], False, soft_temperature)
            else:
                q[:, :t + 1] = probabilities.detach()
                state = state.clone()
                state[:, :t + 1] = belief(revised, soft, soft_temperature)
                if target is not None and t:
                    was_right = before[:, :t].argmax(-1) == target[:, :t]
                    now_right = q[:, :t].argmax(-1) == target[:, :t]
                    wrong_to_right += int((~was_right & now_right).sum())
                    right_to_wrong += int((was_right & ~now_right).sum())
                    attempts += batch * t
        if trace:
            history.append({'position': t, 'before': before[0].tolist(),
                            'after': q[0, :t + 1].tolist()})

    if mode in POSTHOC:
        # The whole sequence is first generated, then refined after generation.
        # Use n full passes so the number of major forward calls equals online
        # revision (n proposal calls + n revision calls), while the order differs.
        present[:, :n] = True
        for pass_index in range(n):
            before = q.clone()
            revised = model(adj, state, present, revision=True)
            probabilities = revised[:, :n].softmax(-1)
            if target is not None:
                if supervision == 'balanced':
                    losses.append(F.cross_entropy(revised[:, pass_index], target[:, pass_index]))
                else:
                    losses.append(F.cross_entropy(revised[:, :n].reshape(-1, 3),
                                                   target.reshape(-1)))
            q = probabilities.detach()
            state = state.clone()
            state[:, :n] = belief(revised[:, :n], soft, soft_temperature)
            if target is not None:
                was_right = before.argmax(-1) == target
                now_right = q.argmax(-1) == target
                wrong_to_right += int((~was_right & now_right).sum())
                right_to_wrong += int((was_right & ~now_right).sum())
                attempts += batch * n
            if trace:
                history.append({'position': n + pass_index,
                                'before': before[0].tolist(),
                                'after': q[0].tolist()})

    return {'probabilities': q, 'loss': torch.stack(losses).mean() if losses else None,
            'wrong_to_right': wrong_to_right, 'right_to_wrong': right_to_wrong,
            'revision_opportunities': attempts, 'trace': history,
            'forward_calls': n * (2 if mode in REVISING or mode in POSTHOC or mode == 'ar_matched' else 1)}
