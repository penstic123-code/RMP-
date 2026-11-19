import numpy as np

def run(seed, mode, width=8, period=128):
    rng = np.random.default_rng(seed)
    search = np.random.default_rng(seed + 1000)
    graphs = rng.permutation(width)
    parents = np.stack([graphs, np.roll(graphs, 1)])
    x = rng.normal(size=(8192, width)).astype('f')
    noise = rng.normal(0, .35, x.shape).astype('f')
    dst = np.arange(width, dtype=np.int32)
    src = search.permutation(width).astype(np.int32)
    w = np.zeros(width, dtype='f')
    buf = np.empty((2, period, width), dtype='f')
    errors = []
    for t, row in enumerate(x):
        pred = row[src] * w
        target = .8 * row[parents[t // 4096]] + noise[t]
        err = target - pred
        errors.append(float(np.mean(err**2)))
        if t < 4096 or mode != 'freeze':
            w += .025 * err * row[src]
        buf[:, t % period] = row, target
        if (t+1) % period or mode == 'static' or (mode in ('weight','freeze') and t >= 4096):
            continue
        for k in dst:
            candidates = np.unique([src[k], *search.choice(width, 2, replace=False)])
            scores = [abs(buf[0, :, j] @ buf[1, :, k]) /
                      max(np.linalg.norm(buf[0, :, j]), 1e-8) for j in candidates]
            best = candidates[np.argmax(scores)]
            if best != src[k]:
                src[k], w[k] = best, 0.
    return [np.mean(errors[a:b]) for a,b in [(3072,4096),(4096,4224),(7168,8192)]], np.mean(src==parents[1])

if __name__ == '__main__':
    for mode in ['static','freeze','weight','online']:
        res = [run(s,mode) for s in range(10)]
        print(mode, np.mean([v[0] for v in res],0), np.mean([v[1] for v in res]))
