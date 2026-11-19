"""DSPN v0.13: hidden-module feedback on a 4-module chain.

Module 0 receives the 2 input channels; only module K-1 has a readout. Modules talk
with 4-dim messages z = P h to their chain neighbours, m_i = sum z_j / 4 (v0.12 rule).
Information must be relayed through the hidden modules 1..K-2.

Task (v0.12 task shifted by H = K-1 hops):
  y(t) = tanh(x0(t-H-3)) * tanh(x1(t-H)) + 0.4 * x0(t-H-2)

Only the internal W of each module learns (plus the NLMS readout). P, U, V fixed.
Methods
  fixed      readout only
  pred_only  SnAp-1 in the readout module only (hidden modules frozen)
  random     every module SnAp-1; hidden modules get err x fixed random +-1 vector
  symmetric  every module SnAp-1; hidden feedback relayed back one hop at a time
             along the message path: delta_j = (U_i P_j / 4)^T (slope_i * delta_i)
  symmetric_lag  as symmetric, but module j's trace (and the downstream slopes used in
             the relay) run k = K-1-j steps behind, matching the k steps its state needs
             to reach the readout. Extra memory: the last K+1 chain states.
  bptt       truncated BPTT over the whole chain (window 16), W blocks only: reference
"""
import numpy as np

STEPS, FREEZE = 4512, 4000
K, d, q = 4, 16, 4
H = K - 1


def make(seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, .75 / np.sqrt(d), (K, d, d))
    P = rng.normal(0, 1 / np.sqrt(d), (K, q, d))
    U = rng.normal(0, 1.0, (K, d, q))
    V = rng.normal(0, .35, (d, 2))
    B = rng.choice([-1., 1.], (K, d))
    x = rng.normal(size=(STEPS, 2))
    y = (np.tanh(np.roll(x[:, 0], H + 3)) * np.tanh(np.roll(x[:, 1], H))
         + .4 * np.roll(x[:, 0], H + 2))
    return W, P, U, V, B, x, y


def coupling(P, U):
    """A[i][j] = U_i P_j / 4 for chain neighbours, as a dict of blocks."""
    return {(i, j): U[i] @ P[j] / 4 for i in range(K) for j in (i - 1, i + 1) if 0 <= j < K}


def big_matrix(W, C):
    A = np.zeros((K * d, K * d))
    for i in range(K):
        A[i*d:(i+1)*d, i*d:(i+1)*d] = W[i]
    for (i, j), blk in C.items():
        A[i*d:(i+1)*d, j*d:(j+1)*d] = blk
    return A


def run(seed, method='fixed', lr=0.0, window=16, lr_h=None):
    lr_h = lr if lr_h is None else lr_h
    W, P, U, V, B, x, y = make(seed)
    C = coupling(P, U)
    Vbig = np.zeros((K * d, 2)); Vbig[:d] = V
    h = np.zeros(K * d)
    readout = np.zeros(d)
    trace = np.zeros((K, d, d))
    hist_h, hist_prev, hist_e, hist_r = [], [], [], []
    errors = []
    hs = []
    learn = [i for i in range(K)] if method in ('random', 'symmetric', 'symmetric_lag') else ([K - 1] if method == 'pred_only' else [])
    for t in range(H + 3, STEPS):
        A = big_matrix(W, C)
        prev = h
        h = np.tanh(A @ prev + Vbig @ x[t] + .15)
        hp = h[-d:]
        err = y[t] - readout @ hp
        if t >= FREEZE:
            errors.append(err ** 2)
            continue
        slope = (1 - h ** 2).reshape(K, d)
        hb, pb = h.reshape(K, d), prev.reshape(K, d)
        hs.append(hb.copy())
        if len(hs) > K + 2:
            hs.pop(0)
        if method == 'symmetric_lag':
            for i in learn:   # lagged SnAp-1: module i looks k_i steps back
                k = K - 1 - i
                if len(hs) >= k + 2:
                    cur, old = hs[-1 - k], hs[-2 - k]
                    trace[i] = (1 - cur[i] ** 2)[:, None] * (old[i][None, :] + np.diag(W[i])[:, None] * trace[i])
        elif learn:
            for i in learn:   # SnAp-1 per module
                trace[i] = slope[i][:, None] * (pb[i][None, :] + np.diag(W[i])[:, None] * trace[i])
        r_used = readout.copy()
        readout += .04 * err * hp / (1 + hp @ hp)
        if method in ('pred_only', 'random', 'symmetric', 'symmetric_lag'):
            delta = np.zeros((K, d))
            delta[K - 1] = err * readout
            for j in range(K - 2, -1, -1):
                if method == 'random':
                    delta[j] = err * B[j]
                elif method == 'symmetric':
                    delta[j] = C[(j + 1, j)].T @ (slope[j + 1] * delta[j + 1])
                elif method == 'symmetric_lag':
                    k1 = K - 1 - (j + 1)
                    s1 = 1 - hs[-1 - k1][j + 1] ** 2 if len(hs) > k1 else slope[j + 1]
                    delta[j] = C[(j + 1, j)].T @ (s1 * delta[j + 1])
            for i in learn:
                W[i] += (lr if i == K - 1 else lr_h) * delta[i][:, None] * trace[i]
        elif method == 'bptt':
            hist_h.append(h); hist_prev.append(prev); hist_e.append(err); hist_r.append(r_used)
            if len(hist_h) == window:
                g = np.zeros(K * d)
                gA = np.zeros((K, d, d))
                for k in range(window - 1, -1, -1):
                    out = np.zeros(K * d); out[-d:] = hist_e[k] * hist_r[k]
                    g = (out + g) * (1 - hist_h[k] ** 2)          # -dL/d(pre-activation)
                    gb, pb_ = g.reshape(K, d), hist_prev[k].reshape(K, d)
                    for i in range(K):
                        gA[i] += np.outer(gb[i], pb_[i])
                    g = A.T @ g
                W += lr * gA / window
                hist_h, hist_prev, hist_e, hist_r = [], [], [], []
    e = float(np.mean(errors))
    return e if np.isfinite(e) else 1e3
