"""全样本的 Moran’s I、LISA，以及当前筛选下的城市均值地图。"""

from __future__ import annotations

import streamlit as st

from src.national_map import LISA_COLORS, build_lisa_map, build_value_map
from src.national_ui import (
    METRICS,
    apply_filter,
    cached_national,
    missing_national_page,
    national_empty,
    render_national_sidebar,
    show_map,
)
from src.ui import cards, hero

hero(
    "空间关联",
    "Moran’s I 和 LISA 用的是整段时间里各城市的 PM2.5 平均，不随侧栏改一天就重算。",
    kicker="全国城市 · k 近邻 · 不是连续污染场",
)

bundle = cached_national()
if bundle is None:
    missing_national_page()
    st.stop()
daily, summary, lisa, _pred = bundle
moran = summary.get("moran_period_mean") or {}
if not moran.get("ok"):
    st.markdown(
        f'<div class="empty-box"><h3>空间统计没有做成</h3><p>{moran.get("reason", "样本不足")}</p></div>',
        unsafe_allow_html=True,
    )
else:
    counts = moran["lisa_counts"]
    cards([
        {"label": "全局 Moran’s I", "value": f"{moran['I']}", "note": f"期望 {moran['expected']} · n={moran['n']}", "accent": "#0B6E4F"},
        {"label": "置换 p 值", "value": f"{moran['p_value']}", "note": f"{moran['n_permutations']} 次，双侧", "accent": "#0B6E4F"},
        {"label": "高-高", "value": str(counts["高-高"]), "note": "和近邻一起偏高", "accent": "#B42318"},
        {"label": "低-低", "value": str(counts["低-低"]), "note": "和近邻一起偏低", "accent": "#1D4E89"},
    ])
    st.markdown(
        f"权重是 **{moran['weight']}**。I 高于期望，表示邻近城市的时段平均更相像。"
        if moran["I"] > moran["expected"]
        else f"权重是 **{moran['weight']}**。这次 I 没有高于期望。"
    )
    inv = moran.get("inverse_distance_400km") or {}
    if inv.get("ok"):
        st.caption(
            f"敏感性：400 公里反距离权重的 I 是 {inv['I']}（n={inv['n']}，p={inv['p_value']}）。"
            f"有 {inv['dropped_isolated']} 个城市因为 400 公里内没有邻居被拿掉。"
        )
    st.caption(moran.get("lisa_note", "局部 p 值未做多重比较校正。") + " 高-高不是排放源定位。")
    winter = summary.get("moran_winter_mean") or {}
    summer = summary.get("moran_summer_mean") or {}
    if winter.get("ok") and summer.get("ok"):
        st.caption(
            f"冬季城市平均的 I 是 {winter['I']}（p={winter['p_value']}，n={winter['n']}），"
            f"夏季是 {summer['I']}（p={summer['p_value']}，n={summer['n']}）。"
        )

st.subheader("LISA 类型")
legend = " ".join(
    f'<span style="color:{color}">● {name}</span>' for name, color in LISA_COLORS.items()
)
st.markdown(legend, unsafe_allow_html=True)
if lisa is None or lisa.empty:
    st.markdown(
        '<div class="empty-box"><h3>没有 LISA 结果文件</h3><p>重新运行 python scripts/prepare_national.py 后，这里会标出高-高、低-低、高-低和低-高。</p></div>',
        unsafe_allow_html=True,
    )
else:
    try:
        lisa_map = build_lisa_map(lisa)
    except Exception as exc:
        lisa_map = None
        st.error(f"LISA 地图没有渲染成功：{exc}")
    if lisa_map is not None:
        show_map(lisa_map, "lisa-full-sample")
    cluster_counts = (
        lisa.groupby(["region", "cluster"]).size().rename("城市数").reset_index()
        .rename(columns={"region": "区域", "cluster": "类型"})
    )
    st.dataframe(cluster_counts, hide_index=True, width="stretch")

st.subheader("当前筛选下的城市平均")
st.caption("下面这张图会随省份、城市、指标和日期变化。它仍然是城市点，不是插值表面。")
state = render_national_sidebar(daily)
if not state.ok:
    national_empty(state.reason)
    st.stop()
part = apply_filter(daily, state).dropna(subset=[state.metric])
if part.empty:
    national_empty("metric")
    st.stop()
points = (
    part.groupby(["city", "province", "lat", "lon", "matched"], dropna=False)[state.metric]
    .agg(value="mean", n_days="size")
    .reset_index()
)
points = points.loc[points["matched"].astype(bool)].dropna(subset=["lat", "lon", "value"])
if points.empty:
    national_empty("rows")
else:
    unit = METRICS[state.metric]["unit"]
    try:
        value_map = build_value_map(points, state.metric, unit)
    except Exception as exc:
        value_map = None
        st.error(f"地图没有渲染成功：{exc}")
    if value_map is not None:
        signature = f"spatial-filter-{state.metric}-{state.season}-{len(points)}-{points['value'].sum():.1f}"
        show_map(value_map, signature)

st.subheader("为什么全国不做克里金")
st.write(summary["interpolation"]["reason"])
st.caption("北京 12 站的逆距离加权留在「北京站点 · 空间分布」。那一层距最近站点超过 40 公里的格点不着色。")
