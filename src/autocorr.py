"""城市点上的空间自相关。

用 k 个最近城市做行标准化权重，不估计县和乡镇。
全局 Moran’s I 的 p 值来自随机打乱浓度、权重不动的置换，不是正态近似。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.china_settings import KNN_K, N_PERM, RANDOM_SEED, SEASON_MIN_DAYS, YEAR_MIN_DAYS
from src.spatial import haversine_km


def knn_weights(lat: np.ndarray, lon: np.ndarray, k: int = KNN_K) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = len(lat)
    if n < 3:
        raise ValueError("至少需要 3 个点才能做空间权重")
    k = min(k, n - 1)
    distance = haversine_km(lat, lon, lat, lon)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
    weights = np.zeros((n, n), dtype=float)
    weights[np.arange(n)[:, None], neighbors] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def morans_i(values: np.ndarray, weights: np.ndarray) -> float | None:
    z = np.asarray(values, dtype=float)
    z = z - z.mean()
    denominator = float(np.dot(z, z))
    if denominator <= 0:
        return None
    numerator = float(np.dot(z, weights @ z))
    n = len(z)
    s0 = float(weights.sum())
    return (n / s0) * (numerator / denominator)


def moran_test(values: np.ndarray, weights: np.ndarray, n_perm: int = N_PERM, seed: int = RANDOM_SEED) -> dict:
    observed = morans_i(values, weights)
    if observed is None:
        return {"I": None, "p": None, "n": int(len(values))}
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        shuffled = morans_i(rng.permutation(values), weights)
        if shuffled is not None and abs(shuffled) >= abs(observed):
            extreme += 1
    return {"I": float(observed), "p": (extreme + 1) / (n_perm + 1), "n": int(len(values))}


def _neighbor_count(weights: np.ndarray) -> int:
    return int((weights[0] > 0).sum())


def local_quadrants(values: np.ndarray, weights: np.ndarray, n_perm: int = N_PERM, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """局部 Moran 的象限。未过置换检验的点记为不显著，不做多重比较校正。"""
    x = np.asarray(values, dtype=float)
    z = x - x.mean()
    lag = weights @ z
    observed = z * lag
    rng = np.random.default_rng(seed)
    extreme = np.ones(len(x), dtype=int)
    for _ in range(n_perm):
        zs = rng.permutation(x)
        zs = zs - zs.mean()
        local = zs * (weights @ zs)
        extreme += (np.abs(local) >= np.abs(observed)).astype(int)
    p_values = extreme / (n_perm + 1)
    labels = []
    for zi, li, p in zip(z, lag, p_values):
        if p >= 0.05:
            labels.append("不显著")
        elif zi > 0 and li > 0:
            labels.append("高高")
        elif zi < 0 and li < 0:
            labels.append("低低")
        elif zi > 0 and li < 0:
            labels.append("高低")
        else:
            labels.append("低高")
    return pd.DataFrame({"local_i": observed, "p": p_values, "quadrant": labels, "value": x})


def _city_means(daily: pd.DataFrame, mask: pd.Series, value_col: str, min_days: int) -> pd.DataFrame:
    part = daily.loc[mask, ["city", value_col]].dropna()
    grouped = part.groupby("city")[value_col].agg(value="mean", n_days="size").reset_index()
    return grouped.loc[grouped["n_days"] >= min_days].copy()


def spatial_frame(means: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    merged = means.merge(meta[["city", "lat", "lon", "region", "province"]], on="city", how="inner")
    return merged.dropna(subset=["lat", "lon", "value"]).reset_index(drop=True)


def annual_global_moran(daily: pd.DataFrame, meta: pd.DataFrame, value_col: str = "pm25") -> pd.DataFrame:
    frame = daily[["date", "city", value_col]].dropna().copy()
    frame["year"] = frame["date"].dt.year
    rows = []
    for year, part in frame.groupby("year"):
        means = part.groupby("city")[value_col].agg(value="mean", n_days="size").reset_index()
        means = means.loc[means["n_days"] >= YEAR_MIN_DAYS]
        located = spatial_frame(means, meta)
        if len(located) < 30:
            rows.append({"year": int(year), "I": None, "p": None, "n": int(len(located)), "min_days": YEAR_MIN_DAYS})
            continue
        weights = knn_weights(located["lat"].to_numpy(), located["lon"].to_numpy())
        result = moran_test(located["value"].to_numpy(), weights)
        rows.append({
            "year": int(year),
            "I": result["I"],
            "p": result["p"],
            "n": result["n"],
            "k": _neighbor_count(weights),
            "min_days": YEAR_MIN_DAYS,
        })
    return pd.DataFrame(rows)


def annual_local_moran(daily: pd.DataFrame, meta: pd.DataFrame, value_col: str = "pm25") -> pd.DataFrame:
    frame = daily[["date", "city", value_col]].dropna().copy()
    frame["year"] = frame["date"].dt.year
    pieces = []
    for year, part in frame.groupby("year"):
        means = part.groupby("city")[value_col].agg(value="mean", n_days="size").reset_index()
        means = means.loc[means["n_days"] >= YEAR_MIN_DAYS]
        located = spatial_frame(means, meta)
        if len(located) < 30:
            continue
        weights = knn_weights(located["lat"].to_numpy(), located["lon"].to_numpy())
        local = local_quadrants(located["value"].to_numpy(), weights, seed=RANDOM_SEED + int(year))
        local.insert(0, "year", int(year))
        local.insert(1, "city", located["city"].to_numpy())
        local["n_days"] = located["n_days"].to_numpy()
        pieces.append(local)
    if not pieces:
        return pd.DataFrame(columns=["year", "city", "local_i", "p", "quadrant", "value", "n_days"])
    return pd.concat(pieces, ignore_index=True)


def season_moran(daily: pd.DataFrame, meta: pd.DataFrame, months: set[int], value_col: str = "pm25") -> dict:
    mask = daily["date"].dt.month.isin(months)
    means = _city_means(daily, mask, value_col, SEASON_MIN_DAYS)
    located = spatial_frame(means, meta)
    if len(located) < 30:
        return {"I": None, "p": None, "n": int(len(located)), "min_days": SEASON_MIN_DAYS}
    weights = knn_weights(located["lat"].to_numpy(), located["lon"].to_numpy())
    result = moran_test(located["value"].to_numpy(), weights, seed=RANDOM_SEED + sum(months))
    result["min_days"] = SEASON_MIN_DAYS
    result["k"] = _neighbor_count(weights)
    return result
