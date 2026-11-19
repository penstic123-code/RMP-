"""DSPN v0.12 follow-up: two design fixes found by experiment 1.

Fix A (delay): lagged traces. The readout already keeps a queue of the states it
  predicted from. Updating the trace from that queue, D steps behind, lines the
  credit up with the error that arrives now. Extra cost: one stored state (d numbers).
  shared_lag : per-source EMA, D steps behind, x slope at prediction time   (2d)
  snap_lag   : SnAp-1 trace, D steps behind                                 (d*d + d)
Fix B (memory horizon): leaky units with spread time constants
  h <- (1-a) h + a tanh(W h + U x + b),  a_i spread evenly over [a_min, 1].

With D=0 and leak off, shared_lag == shared and snap_lag == snap (same numbers).
"""
import numpy as np
from trace_lab2 import make_task, STEPS, FREEZE

TRACE_NUMBERS = {'fixed': lambda d: 0, 'shared': lambda d: d, 'snap': lambda d: d * d,
                 'shared_lag': lambda d: 2 * d, 'snap_lag': lambda d: d * d + d}


def run(seed, L=3, D=0, d=16, method='fixed', lr=0.0, lam=0.5, a_min=None):
    initial, inputs, x, y = make_task(seed, L, d)
    a = np.ones(d) if a_min is None else np.linspace(a_min, 1.0, d)
    w = initial.copy()
    readout = np.zeros(d)
    state = np.zeros(d)
    pre = np.zeros(d)
    trace = np.zeros((d, d))
    prev_feat = np.zeros(d)          # state one step before the popped prediction (lag methods)
    queue = []
    errors = []
    for t in range(L, STEPS):
        old = state
        state = (1 - a) * old + a * np.tanh(w @ old + inputs @ x[t] + .15)
        error = y[t] - readout @ state
        if t >= FREEZE:
            errors.append(error ** 2)
            continue
        gain = a * (1 - ((state - (1 - a) * old) / a) ** 2)   # d state_i / d (W h)_i
        if method == 'shared':
            pre = lam * pre + (1 - lam) * old
        elif method == 'snap':
            trace = (1 - a)[:, None] * trace + gain[:, None] * (old[None, :] + np.diag(w)[:, None] * trace)
        queue.append((error, state.copy(), gain.copy()))
        if len(queue) <= D:
            continue
        err, feat, g = queue.pop(0)
        readout += .04 * err * feat / (1 + feat @ feat)
        if method == 'fixed':
            prev_feat = feat
            continue
        if method == 'shared':
            update = gain[:, None] * pre
        elif method == 'snap':
            update = trace
        elif method == 'shared_lag':
            pre = lam * pre + (1 - lam) * prev_feat
            update = g[:, None] * pre
        elif method == 'snap_lag':
            trace = (1 - a)[:, None] * trace + g[:, None] * (prev_feat[None, :] + np.diag(w)[:, None] * trace)
            update = trace
        prev_feat = feat
        w += lr * err * readout[:, None] * update
    e = float(np.mean(errors))
    return e if np.isfinite(e) else 1e3
