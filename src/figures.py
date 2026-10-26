"""把全周期结果画成可放进说明文档的静态图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.metrics import diurnal_profile
from src.settings import SEASON_ORDER
from src.spatial import draw_dendrogram

ACCENT = "#0B6E4F"
INK = "#1C2830"
MUTED = "#8A8175"
ALERT = "#B42318"


def setup_font() -> str | None:
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.family"] = "sans-serif"
            return name
    return None


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=13, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color="#E6E1D6", zorder=0)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_all(summary: dict, extras: dict, fig_dir: Path) -> list[str]:
    setup_font()
    sns.set_theme(style="whitegrid", font=plt.rcParams.get("font.sans-serif", ["sans-serif"])[0])
    city = extras["city"]
    hourly = extras["hourly"]
    written = []

    monthly = city.assign(ym=city["date"].dt.to_period("M").dt.to_timestamp())
    monthly = monthly.groupby("ym", as_index=False)["PM2.5"].mean()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(monthly["ym"], monthly["PM2.5"], color=ACCENT, lw=1.8)
    ax.axhline(75, color=ALERT, ls="--", lw=1, label="75 μg/m³")
    _style(ax, "城市日均 PM2.5 的月均序列", "月份", "μg/m³")
    ax.legend(frameon=False)
    path = fig_dir / "monthly_pm25.png"
    _save(fig, path)
    written.append(path.name)

    season = city.groupby("season")["PM2.5"].mean().reindex(SEASON_ORDER)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.bar(season.index, season.values, color=["#7BA38A", "#E2B15A", "#D08A3A", "#3E6B89"])
    _style(ax, "城市日均 PM2.5 的季节平均", "季节", "μg/m³")
    path = fig_dir / "seasonal_pm25.png"
    _save(fig, path)
    written.append(path.name)

    stations = pd.DataFrame(summary["pm25"]["stations"]).sort_values("value")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [{"城区": "#0B6E4F", "郊区": "#C48A12", "背景点": "#3E6B89"}.get(z, ACCENT) for z in stations["zone"]]
    ax.barh(stations["zh"], stations["value"], color=colors)
    _style(ax, "各站 PM2.5 日均（有效日）", "μg/m³")
    path = fig_dir / "station_rank_pm25.png"
    _save(fig, path)
    written.append(path.name)

    diurnal = diurnal_profile(hourly, "PM2.5")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(diurnal["hour"], diurnal["value"], marker="o", color=ACCENT)
    _style(ax, "PM2.5 平均日变化", "小时", "μg/m³")
    ax.set_xticks(range(0, 24, 2))
    path = fig_dir / "diurnal_pm25.png"
    _save(fig, path)
    written.append(path.name)

    cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    corr = city[cols].corr()
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.8}, linewidths=0.4, linecolor="white")
    ax.set_title("城市日尺度相关系数", loc="left", color=INK)
    path = fig_dir / "correlation_heatmap.png"
    _save(fig, path)
    written.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, xcol, label in zip(axes, ["WSPM", "DEWP"], ["风速 m/s", "露点 °C"]):
        ax.scatter(city[xcol], city["PM2.5"], s=12, alpha=0.35, color=ACCENT, linewidths=0)
        _style(ax, f"PM2.5 与{label.split()[0]}", label, "PM2.5 μg/m³")
    path = fig_dir / "meteo_scatter.png"
    _save(fig, path)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(city["PM2.5"], bins=40, color=ACCENT, alpha=0.9)
    ax.axvline(75, color=MUTED, ls="--", label="75")
    ax.axvline(150, color=ALERT, ls="--", label="150")
    _style(ax, "城市日均 PM2.5 分布", "μg/m³", "天数")
    ax.legend(frameon=False)
    path = fig_dir / "pm25_distribution.png"
    _save(fig, path)
    written.append(path.name)

    before = cleaning_pairs(summary, "missing_rate_before_interp")
    after = cleaning_pairs(summary, "missing_rate_after_interp")
    labels = list(before.keys())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, [before[k] * 100 for k in labels], width=0.36, color=MUTED, label="插值前")
    ax.bar(x + 0.18, [after[k] * 100 for k in labels], width=0.36, color=ACCENT, label="插值后")
    ax.set_xticks(x, labels, rotation=30)
    _style(ax, "缺失率（插值最多 6 小时）", ylabel="%")
    ax.legend(frameon=False)
    path = fig_dir / "missingness.png"
    _save(fig, path)
    written.append(path.name)

    cluster = extras.get("cluster")
    if cluster is not None:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        draw_dendrogram(cluster, ax)
        _style(ax, "站点层次聚类树状图（展示距离，分组以轮廓系数选定的 k 为准）")
        path = fig_dir / "cluster_dendrogram.png"
        _save(fig, path)
        written.append(path.name)

    loocv = summary["loocv"]
    if loocv.get("ok"):
        fig, ax = plt.subplots(figsize=(5.6, 4))
        ax.bar(["逆距离加权", "最近邻"], [loocv["idw_rmse"], loocv["nn_rmse"]], color=[ACCENT, "#C48A12"])
        _style(ax, "PM2.5 站均浓度留一法 RMSE", ylabel="μg/m³")
        path = fig_dir / "spatial_loocv.png"
        _save(fig, path)
        written.append(path.name)

    return written


def cleaning_pairs(summary: dict, key: str) -> dict:
    data = summary["cleaning"][key]
    order = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "WSPM", "DEWP", "RAIN"]
    return {name: data[name] for name in order if name in data}
