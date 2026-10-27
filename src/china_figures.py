"""全国层的三张静态图，给 README 和面试材料用。"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.china_settings import REGION_ORDER
from src.figures import setup_font
from src.paths import FIG_DIR


def save_china_figures(summary: dict) -> None:
    setup_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _regions(summary)
    _moran(summary)
    _errors(summary)
    plt.close("all")


def _regions(summary: dict) -> None:
    rows = [row for row in summary["levels"]["regions"] if row.get("mean") is not None]
    rows.sort(key=lambda row: REGION_ORDER.index(row["region"]) if row["region"] in REGION_ORDER else 99)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    names = [row["region"] for row in rows]
    values = [row["mean"] for row in rows]
    colors = ["#0B6E4F" if name != "其他" else "#8A8175" for name in names]
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("PM2.5（μg/m³）")
    ax.set_title("长期城市等权平均，不是人口暴露", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "china_region_pm25.png", dpi=140)
    plt.close(fig)


def _moran(summary: dict) -> None:
    rows = [row for row in summary["moran_annual"] if row.get("I") is not None]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot([row["year"] for row in rows], [row["I"] for row in rows], color="#0B6E4F", marker="o")
    ax.axhline(0, color="#C8C2B6", linewidth=1)
    ax.set_ylabel("Moran’s I")
    ax.set_title("城市年均 PM2.5 的全局空间自相关", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "china_moran_year.png", dpi=140)
    plt.close(fig)


def _errors(summary: dict) -> None:
    test = summary["forecast"]["splits"].get("test") or {}
    if not test.get("persistence"):
        return
    labels = ["明天等于今天", "岭回归", "梯度提升"]
    values = [test["persistence"]["mae"], test["ridge"]["mae"], test["boosting"]["mae"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, values, color=["#8A8175", "#2F6F9F", "#0B6E4F"])
    ax.set_ylabel("测试集平均绝对误差（μg/m³）")
    ax.set_title("2025 年目标日，同一批样本", loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "china_forecast_mae.png", dpi=140)
    plt.close(fig)
