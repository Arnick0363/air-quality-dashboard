"""Folium 地图：站点标记、热力层和逆距离加权色斑。"""

from __future__ import annotations

import base64
import io

import numpy as np

from src.settings import IDW_MAX_KM, IDW_POWER, STATIONS
from src.spatial import idw_grid


def _png_data_uri(grid: np.ndarray, vmin: float, vmax: float) -> str:
    import matplotlib.pyplot as plt

    if vmax <= vmin:
        vmax = vmin + 1
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad((0, 0, 0, 0))
    ax.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="auto",
    )
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _hex_color(value: float, vmin: float, vmax: float) -> str:
    import matplotlib.pyplot as plt

    if vmax <= vmin:
        tone = 0.5
    else:
        tone = min(1.0, max(0.0, (value - vmin) / (vmax - vmin)))
    red, green, blue, _ = plt.get_cmap("YlOrRd")(tone)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def build_map(station_stats, pollutant: str, unit: str, show_heat: bool, show_idw: bool):
    """station_stats 需要 station, zh, value, n_days, lat, lon, zone。"""
    import folium
    from branca.colormap import linear
    from folium.plugins import HeatMap

    frame = station_stats.dropna(subset=["value", "lat", "lon"]).copy()
    if frame.empty:
        return None
    vmin = float(frame["value"].min())
    vmax = float(frame["value"].max())
    center = [float(frame["lat"].mean()), float(frame["lon"].mean())]
    fmap = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(fmap)

    if show_idw and len(frame) >= 3:
        grid = idw_grid(
            frame["lat"].to_numpy(),
            frame["lon"].to_numpy(),
            frame["value"].to_numpy(),
            power=IDW_POWER,
            max_km=IDW_MAX_KM,
        )
        folium.raster_layers.ImageOverlay(
            image=_png_data_uri(grid["grid"], vmin, vmax),
            bounds=[[grid["south"], grid["west"]], [grid["north"], grid["east"]]],
            opacity=0.62,
            name="逆距离加权",
            interactive=False,
        ).add_to(fmap)

    if show_heat:
        span = vmax - vmin
        heat = []
        for row in frame.itertuples(index=False):
            weight = 0.35 if span <= 0 else 0.2 + 0.8 * ((row.value - vmin) / span)
            heat.append([row.lat, row.lon, weight])
        HeatMap(heat, name="热力", radius=28, blur=22, min_opacity=0.35).add_to(fmap)

    markers = folium.FeatureGroup(name="站点", show=True)
    for row in frame.itertuples(index=False):
        popup = (
            f"<b>{row.zh}</b><br>{row.zone}<br>"
            f"{pollutant}：{row.value:.1f} {unit}<br>有效天数：{int(row.n_days)}"
        )
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=9,
            color="#1C2830",
            weight=1,
            fill=True,
            fill_color=_hex_color(row.value, vmin, vmax),
            fill_opacity=0.95,
            tooltip=f"{row.zh} {row.value:.1f}",
            popup=folium.Popup(popup, max_width=260),
        ).add_to(markers)
    markers.add_to(fmap)

    if vmax > vmin:
        legend = linear.YlOrRd_09.scale(vmin, vmax)
        legend.caption = f"{pollutant}（{unit}）"
        legend.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def attach_coordinates(stats):
    out = stats.copy()
    out["zh"] = out["station"].map(lambda sid: STATIONS[sid]["zh"])
    out["lat"] = out["station"].map(lambda sid: STATIONS[sid]["lat"])
    out["lon"] = out["station"].map(lambda sid: STATIONS[sid]["lon"])
    out["zone"] = out["station"].map(lambda sid: STATIONS[sid]["zone"])
    return out
