"""DSPN v0.12 trace experiments: one 16-unit (or d-unit) recurrent block,
online NLMS readout, local plasticity of W with feedback = readout weights.

Task: y(t) = tanh(x0(t-L)) * tanh(x1(t)) + 0.4 * x0(t-L+1)   (L=3 is the v0.11 task)
Feedback delay D: the error of the prediction made at t is only known at t+D.
  - readout learns with the features saved at prediction time (queue of D states)
  - W learns at t+D with the trace as it stands at t+D (the trace must bridge D)
Train 4000 steps, then freeze everything and score the last 512 steps.

Methods and persistent trace numbers per block (d=16):
  fixed   0      only the readout learns
  shared  d      per-source EMA of activity, x current target slope (decay tuned)
  multi   3d     shared trace at three fixed decays 0.5/0.9/0.99, averaged
  full    d*d    per-synapse EMA of (target slope x source activity) (decay tuned)
  snap    d*d    SnAp-1: e_ij = slope_i * (h_j(t-1) + W_ii * e_ij)   (no decay knob)
"""
import numpy as np

STEPS, FREEZE, SCORE = 4512, 4000, 512
TRACE_NUMBERS = {'fixed': lambda d: 0, 'shared': lambda d: d, 'multi': lambda d: 3 * d,
                 'full': lambda d: d * d, 'snap': lambda d: d * d}


def make_task(seed, L, d):
    rng = np.random.default_rng(seed)
    initial = rng.normal(0, .75 / np.sqrt(d), (d, d))
    inputs = rng.normal(0, .35, (d, 2))
    rng.choice([-1., 1.], d)        # v0.11 drew a random feedback vector here; kept so data match
    x = rng.normal(size=(STEPS, 2))
    y = np.tanh(np.roll(x[:, 0], L)) * np.tanh(x[:, 1]) + .4 * np.roll(x[:, 0], L - 1)
    return initial, inputs, x, y


def run(seed, L=3, D=0, d=16, method='fixed', lr=0.0, lam=0.5):
    initial, inputs, x, y = make_task(seed, L, d)
    w = initial.copy()
    readout = np.zeros(d)
    state = np.zeros(d)
    pre = np.zeros(d)
    multi = np.zeros((3, d))
    lams = np.array([.5, .9, .99])[:, None]
    trace = np.zeros((d, d))
    queue = []                      # (error, features) waiting for their observation
    errors = []
    for t in range(L, STEPS):
        old = state
        state = np.tanh(w @ old + inputs @ x[t] + .15)
        error = y[t] - readout @ state
        if t >= FREEZE:
            errors.append(error ** 2)
            continue
        slope = 1 - state ** 2
        # update the traces every step (they only look at activity)
        if method == 'shared':
            pre = lam * pre + (1 - lam) * old
        elif method == 'multi':
            multi = lams * multi + (1 - lams) * old
        elif method == 'full':
            trace = lam * trace + (1 - lam) * slope[:, None] * old
        elif method == 'snap':
            trace = slope[:, None] * (old[None, :] + np.diag(w)[:, None] * trace)
        queue.append((error, state.copy()))
        if len(queue) <= D:
            continue
        err, feat = queue.pop(0)    # the observation for this old prediction arrives now
        readout += .04 * err * feat / (1 + feat @ feat)
        if method == 'fixed':
            continue
        if method == 'shared':
            update = slope[:, None] * pre
        elif method == 'multi':
            update = slope[:, None] * multi.mean(0)
        else:
            update = trace
        w += lr * err * readout[:, None] * update
    e = float(np.mean(errors))
    return e if np.isfinite(e) else 1e3
