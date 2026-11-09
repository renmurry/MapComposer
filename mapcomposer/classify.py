"""Numeric classification for graduated colours: natural breaks (Fisher-Jenks), quantile, equal interval."""
import math
import numpy as np

METHODS = {"Natural breaks": "jenks", "Quantile": "quantile", "Equal interval": "equal"}


def _jenks(values, k):
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) > 800:   # sample evenly for speed; breaks on the sample are a good approximation
        x = x[np.linspace(0, len(x) - 1, 800).astype(int)]
    n = len(x)
    k = min(k, len(np.unique(x)))
    if k < 2:
        return [x[0], x[-1]]
    s1 = np.concatenate([[0], np.cumsum(x)]); s2 = np.concatenate([[0], np.cumsum(x * x)])

    def ssd(i, j):  # sum of squared deviations of x[i:j] for arrays of i
        cnt = j - i
        s = s1[j] - s1[i]
        return (s2[j] - s2[i]) - s * s / np.maximum(cnt, 1)

    cost = np.full((k + 1, n + 1), np.inf); back = np.zeros((k + 1, n + 1), dtype=int)
    cost[0, 0] = 0
    for c in range(1, k + 1):
        for j in range(c, n + 1):
            i = np.arange(c - 1, j)
            v = cost[c - 1, i] + ssd(i, j)
            m = int(np.argmin(v)); cost[c, j] = v[m]; back[c, j] = i[m]
    cuts, j = [], n
    for c in range(k, 0, -1):
        i = back[c, j]; cuts.append(j); j = i
    cuts = sorted(cuts)
    edges = [x[0]] + [x[c - 1] for c in cuts]
    return edges


def breaks(values, method="jenks", k=5):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return []
    lo, hi = float(v.min()), float(v.max())
    if lo == hi:
        return [lo, hi]
    if method == "quantile":
        e = list(np.unique(np.quantile(v, np.linspace(0, 1, k + 1))))
    elif method == "equal":
        e = list(np.linspace(lo, hi, k + 1))
    else:
        e = list(np.unique(_jenks(v, k)))
    e[0], e[-1] = lo, hi
    return e


def assign(values, edges):
    """Class index 0..len(edges)-2 for each value (NaN -> -1)."""
    v = np.asarray(values, dtype=float)
    idx = np.searchsorted(np.asarray(edges[1:-1]), v, side="left")
    idx[np.isnan(v)] = -1
    return idx


def fmt(v):
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:,.1f}M"
    if a >= 100 or float(v).is_integer():
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:,.1f}"
    return f"{v:.3g}"


def labels(edges):
    return [f"{fmt(edges[i])} – {fmt(edges[i + 1])}" for i in range(len(edges) - 1)]
