"""城市点上的空间权重、全局 Moran’s I 和局部 LISA。

权重在城市代表点之间计算。这描述的是城市均值之间的空间聚集，
不是街道尺度的连续污染场。
"""

from __future__ import annotations

import numpy as np

from src.spatial import haversine_km


def knn_weights(lat, lon, k: int = 8) -> np.ndarray:
    """k 近邻、行标准化、对角线为 0。k 大于邻居数时自动缩小。"""
    lat = np.asarray(lat, dtype=float).reshape(-1)
    lon = np.asarray(lon, dtype=float).reshape(-1)
    n = len(lat)
    if n < 2:
        raise ValueError("至少需要两个点才能构造空间权重")
    k_use = min(int(k), n - 1)
    dist = haversine_km(lat, lon, lat, lon).astype(float)
    np.fill_diagonal(dist, np.inf)
    neighbors = np.argpartition(dist, k_use - 1, axis=1)[:, :k_use]
    weights = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    weights[rows, neighbors] = 1.0
    np.fill_diagonal(weights, 0.0)
    row_sum = weights.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return weights / row_sum


def inverse_distance_weights(lat, lon, cutoff_km: float = 400.0) -> tuple[np.ndarray, np.ndarray]:
    """反距离权重，超过 cutoff 的城市对记为 0。

    返回 (权重, 保留下来的行号)。完全没有邻居的城市会被拿掉，
    权重在剩余城市上重新做行标准化。
    """
    lat = np.asarray(lat, dtype=float).reshape(-1)
    lon = np.asarray(lon, dtype=float).reshape(-1)
    n = len(lat)
    dist = haversine_km(lat, lon, lat, lon).astype(float)
    np.fill_diagonal(dist, np.inf)
    raw = np.zeros((n, n), dtype=float)
    close = dist <= float(cutoff_km)
    raw[close] = 1.0 / dist[close]
    keep = raw.sum(axis=1) > 0
    idx = np.flatnonzero(keep)
    if len(idx) < 2:
        return np.zeros((0, 0)), idx
    sub = raw[np.ix_(idx, idx)]
    row_sum = sub.sum(axis=1, keepdims=True)
    return sub / row_sum, idx


def spatial_lag_matrix(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """对每一行做空间滞后。缺失的邻居不进入权重，并按剩余邻居重新标准化。

    values 形状 (n,) 或 (n_times, n_cities)。weights[i, j] 是城市 i 使用城市 j 的权重。
    """
    array = np.asarray(values, dtype=float)
    squeezed = array.ndim == 1
    if squeezed:
        array = array.reshape(1, -1)
    valid = np.isfinite(array)
    filled = np.where(valid, array, 0.0)
    numer = filled @ weights.T
    denom = valid.astype(float) @ weights.T
    out = np.full(numer.shape, np.nan, dtype=float)
    np.divide(numer, denom, out=out, where=denom > 0)
    return out.reshape(-1) if squeezed else out


def _as_z(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("空间统计的输入不能有缺失")
    std = float(x.std(ddof=0))
    if std == 0 or len(x) < 3:
        raise ValueError("取值没有变异，或点数少于 3")
    return (x - x.mean()) / std


def global_morans_i(values, weights) -> dict:
    """行标准化或任意非负权重下的全局 Moran’s I，以及正态假设下的期望。"""
    z = _as_z(values)
    w = np.asarray(weights, dtype=float)
    n = len(z)
    s0 = float(w.sum())
    if s0 <= 0:
        raise ValueError("空间权重全为 0")
    numerator = float(z @ w @ z)
    denominator = float(z @ z)
    observed = (n / s0) * (numerator / denominator)
    return {
        "I": observed,
        "expected": -1.0 / (n - 1),
        "n": n,
        "s0": s0,
    }


def permutation_p_value(observed: float, null_values: np.ndarray, expected: float) -> float:
    """双侧置换 p 值：(更极端的次数 + 1) / (置换次数 + 1)。"""
    null_values = np.asarray(null_values, dtype=float)
    extreme = np.abs(null_values - expected) >= abs(observed - expected) - 1e-15
    return float((int(extreme.sum()) + 1) / (len(null_values) + 1))


def _permute_matrix(z: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((n_perm, len(z))), axis=1)
    return z[order]


def global_moran_test(values, weights, n_perm: int = 999, seed: int = 42) -> dict:
    """全局 Moran’s I 加置换检验。"""
    base = global_morans_i(values, weights)
    z = _as_z(values)
    w = np.asarray(weights, dtype=float)
    n = len(z)
    s0 = base["s0"]
    zp = _permute_matrix(z, n_perm, seed)
    denom = np.sum(zp * zp, axis=1)
    numer = np.sum(zp * (zp @ w.T), axis=1)
    null = (n / s0) * (numer / denom)
    p_value = permutation_p_value(base["I"], null, base["expected"])
    return {**base, "p_value": p_value, "n_permutations": int(n_perm), "seed": int(seed)}


def local_moran(values, weights, n_perm: int = 999, seed: int = 42, alpha: float = 0.05) -> dict:
    """局部 Moran’s I（LISA）。显著类型只在未校正的置换 p 值低于 alpha 时标注。

    类型：高-高、低-低、高-低、低-高、不显著。p 值没有做多重比较校正。
    """
    z = _as_z(values)
    w = np.asarray(weights, dtype=float)
    lag = w @ z
    local = z * lag
    zp = _permute_matrix(z, n_perm, seed)
    local_null = zp * (zp @ w.T)
    exceed = np.abs(local_null) >= np.abs(local)[None, :]
    p_value = (exceed.sum(axis=0) + 1) / (n_perm + 1)
    labels = np.array(["不显著"] * len(z), dtype=object)
    significant = p_value < alpha
    hh = significant & (z > 0) & (lag > 0)
    ll = significant & (z < 0) & (lag < 0)
    hl = significant & (z > 0) & (lag < 0)
    lh = significant & (z < 0) & (lag > 0)
    labels[hh] = "高-高"
    labels[ll] = "低-低"
    labels[hl] = "高-低"
    labels[lh] = "低-高"
    return {
        "local_i": local,
        "z": z,
        "spatial_lag": lag,
        "p_value": p_value.astype(float),
        "cluster": labels,
        "n_permutations": int(n_perm),
        "alpha": float(alpha),
        "seed": int(seed),
    }
