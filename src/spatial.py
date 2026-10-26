"""站点层次聚类与逆距离加权插值。只使用站点均值，不做高分辨率宣称。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.settings import IDW_POWER, POLLUTANTS, STATIONS, station_zh

EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """返回形状 (n_target, n_source) 的球面距离，单位公里。"""
    lat1 = np.asarray(lat1, dtype=float).reshape(-1)
    lon1 = np.asarray(lon1, dtype=float).reshape(-1)
    lat2 = np.asarray(lat2, dtype=float).reshape(-1)
    lon2 = np.asarray(lon2, dtype=float).reshape(-1)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)[:, None]
    dphi = phi2 - phi1
    dlmb = np.radians(lon2)[:, None] - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi2) * np.cos(phi1) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def idw_predict(src_lat, src_lon, values, tgt_lat, tgt_lon, power: float = IDW_POWER) -> np.ndarray:
    """逆距离加权。目标点与某个站点重合时直接返回该站的值。"""
    values = np.asarray(values, dtype=float).reshape(-1)
    dist = haversine_km(src_lat, src_lon, tgt_lat, tgt_lon)
    nearest = dist.argmin(axis=1)
    exact = dist.min(axis=1) < 0.001
    dist = np.maximum(dist, 1e-6)
    weights = 1.0 / dist ** power
    pred = (weights * values).sum(axis=1) / weights.sum(axis=1)
    pred[exact] = values[nearest[exact]]
    return pred


def idw_grid(
    src_lat,
    src_lon,
    values,
    power: float = IDW_POWER,
    resolution: int = 90,
    pad: float = 0.12,
    max_km: float = 40.0,
) -> dict:
    south = float(np.min(src_lat) - pad)
    north = float(np.max(src_lat) + pad)
    west = float(np.min(src_lon) - pad)
    east = float(np.max(src_lon) + pad)
    lat_axis = np.linspace(south, north, resolution)
    lon_axis = np.linspace(west, east, resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_axis, lat_axis)
    pred = idw_predict(src_lat, src_lon, values, lat_mesh.ravel(), lon_mesh.ravel(), power=power)
    dist = haversine_km(src_lat, src_lon, lat_mesh.ravel(), lon_mesh.ravel())
    nearest_km = dist.min(axis=1)
    grid = pred.reshape(resolution, resolution)
    nearest_grid = nearest_km.reshape(resolution, resolution)
    grid[nearest_grid > max_km] = np.nan
    return {
        "grid": grid,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "lat_axis": lat_axis,
        "lon_axis": lon_axis,
    }


def loocv_idw_nn(lat, lon, values, power: float = IDW_POWER) -> dict:
    """留一法：每次拿掉一个站点，用其余站点做 IDW 和最近邻预测。"""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 4 or np.isnan(values).any():
        return {"n": int(np.sum(~np.isnan(values))), "ok": False}
    idw_pred = np.empty(n)
    nn_pred = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        idw_pred[i] = idw_predict(lat[mask], lon[mask], values[mask], [lat[i]], [lon[i]], power=power)[0]
        dist = haversine_km(lat[mask], lon[mask], [lat[i]], [lon[i]])[0]
        nn_pred[i] = values[mask][int(np.argmin(dist))]
    def _scores(pred):
        err = pred - values
        return {
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
        }
    idw_scores = _scores(idw_pred)
    nn_scores = _scores(nn_pred)
    return {
        "ok": True,
        "n": n,
        "power": power,
        "idw_rmse": idw_scores["rmse"],
        "idw_mae": idw_scores["mae"],
        "nn_rmse": nn_scores["rmse"],
        "nn_mae": nn_scores["mae"],
    }


def _cluster_names(k: int) -> list[str]:
    if k == 2:
        return ["较高组", "较低组"]
    if k == 3:
        return ["高值组", "中值组", "低值组"]
    return [f"组 {i + 1}" for i in range(k)]


def cluster_stations(daily: pd.DataFrame) -> dict | None:
    """按污染物日均和冬夏 PM2.5 比值做 ward 层次聚类。

    分组个数在 2 到 5 之间用轮廓系数选取。站点太少时返回 None。
    """
    if daily.empty:
        return None
    means = daily.groupby("station")[POLLUTANTS + ["WSPM"]].mean()
    winter = daily.loc[daily["season"] == "冬"].groupby("station")["PM2.5"].mean()
    summer = daily.loc[daily["season"] == "夏"].groupby("station")["PM2.5"].mean()
    ratio = (winter / summer).rename("pm25_winter_summer_ratio")
    features = means.join(ratio).replace([np.inf, -np.inf], np.nan)
    used_ratio = bool(features["pm25_winter_summer_ratio"].notna().all())
    if not used_ratio:
        features = features.drop(columns=["pm25_winter_summer_ratio"])
    features = features.dropna()
    if len(features) < 4:
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    max_k = min(5, len(features) - 1)
    scores = {}
    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        labels_try = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled)
        score = float(silhouette_score(scaled, labels_try))
        scores[str(k)] = score
        if score > best_score:
            best_k, best_score = k, score
    labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(scaled)
    link = linkage(scaled, method="ward")

    labeled = features.copy()
    labeled["cluster_id"] = labels
    order = (
        labeled.groupby("cluster_id")["PM2.5"].mean().sort_values(ascending=False).index.tolist()
    )
    names = _cluster_names(best_k)
    id_to_name = {cid: names[rank] for rank, cid in enumerate(order)}
    labeled["cluster"] = labeled["cluster_id"].map(id_to_name)
    labeled["zh"] = [station_zh(sid) for sid in labeled.index]
    labeled["zone"] = [STATIONS[sid]["zone"] for sid in labeled.index]
    labeled["lat"] = [STATIONS[sid]["lat"] for sid in labeled.index]
    labeled["lon"] = [STATIONS[sid]["lon"] for sid in labeled.index]

    members = {
        name: [station_zh(sid) for sid in labeled.index[labeled["cluster"] == name]]
        for name in names
    }
    return {
        "k": best_k,
        "silhouette": best_score,
        "silhouette_by_k": scores,
        "used_winter_summer_ratio": used_ratio,
        "feature_columns": list(features.columns),
        "members": members,
        "table": labeled.reset_index(),
        "linkage": link,
        "scaled": scaled,
        "leaf_labels": [station_zh(sid) for sid in features.index],
    }


def draw_dendrogram(cluster_result: dict, ax) -> None:
    dendrogram(
        cluster_result["linkage"],
        labels=cluster_result["leaf_labels"],
        leaf_rotation=45,
        ax=ax,
        color_threshold=0,
        above_threshold_color="#0B6E4F",
    )
