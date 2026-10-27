"""全国城市代表点的 Folium 地图。一个点是一座城市，不是连续浓度场。"""

from __future__ import annotations

import folium
import numpy as np
import pandas as pd

LISA_COLORS = {
    "高-高": "#B42318",
    "低-低": "#1D4E89",
    "高-低": "#E07A2F",
    "低-高": "#5B8FA8",
    "不显著": "#C5C0B6",
}

METRIC_BREAKS = {
    "PM2.5": [35, 75, 115, 150, 250],
    "PM10": [50, 150, 250, 350, 420],
    "SO2": [20, 60, 150, 475, 800],
    "NO2": [40, 80, 180, 280, 565],
    "CO": [2, 4, 14, 24, 36],
    "O3": [40, 80, 120, 160, 200],
    "AQI": [50, 100, 150, 200, 300],
}
SCALE_COLORS = ["#1F7A4D", "#7D9B3A", "#C48A12", "#E07A2F", "#D9482B", "#8C1D18"]


def metric_color(metric: str, value: float) -> str:
    if value is None or not np.isfinite(value):
        return "#B7B1A6"
    breaks = METRIC_BREAKS.get(metric, METRIC_BREAKS["PM2.5"])
    for color, upper in zip(SCALE_COLORS, breaks):
        if value <= upper:
            return color
    return SCALE_COLORS[-1]


def _base_map() -> folium.Map:
    return folium.Map(location=[35.2, 103.8], zoom_start=4, control_scale=True, tiles="OpenStreetMap")


def build_value_map(points: pd.DataFrame, metric: str, unit: str) -> folium.Map | None:
    usable = points.dropna(subset=["lat", "lon", "value"])
    if usable.empty:
        return None
    fmap = _base_map()
    for row in usable.itertuples(index=False):
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6.5,
            color=metric_color(metric, row.value),
            fill=True,
            fill_opacity=0.85,
            weight=1,
            tooltip=f"{row.city} · {row.province}<br>{metric} {row.value:.1f} {unit}<br>有效日 {int(row.n_days)}",
        ).add_to(fmap)
    return fmap


def build_lisa_map(points: pd.DataFrame) -> folium.Map | None:
    usable = points.dropna(subset=["lat", "lon", "cluster"])
    if usable.empty:
        return None
    fmap = _base_map()
    for row in usable.itertuples(index=False):
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6.5,
            color=LISA_COLORS.get(row.cluster, "#C5C0B6"),
            fill=True,
            fill_opacity=0.88,
            weight=1,
            tooltip=(
                f"{row.city} · {row.cluster}<br>"
                f"时段平均 PM2.5 {row.pm25_mean:.1f} μg/m³<br>"
                f"局部 I {row.local_i:.2f}，p={row.p_value:.3f}"
            ),
        ).add_to(fmap)
    return fmap
