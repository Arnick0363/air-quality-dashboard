"""站点地图、分区对比和插值留一法。"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.mapping import attach_coordinates, build_map
from src.metrics import station_means
from src.settings import IDW_POWER, STANDARDS, value_column
from src.spatial import loocv_idw_nn
from src.ui import empty_state, filtered, get_filters, hero

state = get_filters()
hero("空间分布", "站点位置、分区差异，以及用站点均值做的逆距离加权和最近邻对比。")

if not state.ok:
    empty_state(state.reason)
    st.stop()

_hourly, daily = filtered(state)
pollutant = state.pollutant
column = value_column(pollutant)
unit = STANDARDS[pollutant]["unit"]
if daily.empty or column not in daily.columns or daily[column].notna().sum() == 0:
    empty_state("days")
    st.stop()

stats = attach_coordinates(station_means(daily, pollutant))
if stats.empty:
    empty_state("days")
    st.stop()

st.caption("坐标来自公开文献中的国控点位置（三位小数）。色斑是描述性插值，不是监管用浓度场。距最近站点超过 40 公里的格点不着色。")
show_heat = st.toggle("显示热力层", value=True)
show_idw = st.toggle("显示逆距离加权", value=True)
if len(stats) < 3:
    st.warning("逆距离加权至少需要 3 个有有效日值的站点。当前只画站点标记。")
    show_idw = False

try:
    fmap = build_map(stats, pollutant, unit, show_heat=show_heat, show_idw=show_idw)
except Exception as exc:
    fmap = None
    st.error(f"地图没有渲染成功：{exc}")

if fmap is None:
    empty_state("days")
else:
    from streamlit_folium import st_folium

    signature = (
        f"{pollutant}-{tuple(stats['station'])}-{show_heat}-{show_idw}-"
        f"{stats['value'].round(2).sum():.2f}"
    )
    try:
        st_folium(fmap, height=560, use_container_width=True, returned_objects=[], key=signature)
    except TypeError:
        st_folium(fmap, height=560, width=900, returned_objects=[], key=signature)

zone = stats.groupby("zone", as_index=False).agg(平均=("value", "mean"), 站点数=("station", "count"))
st.subheader("功能分区")
st.caption("城区、郊区、背景点是监测网络的既有分类，不是聚类结果。每个点先取本站平均，再在区内等权。")
zone_chart = (
    alt.Chart(zone)
    .mark_bar(color="#0B6E4F")
    .encode(
        x=alt.X("zone:N", title=None),
        y=alt.Y("平均:Q", title=f"{pollutant}（{unit}）"),
        tooltip=["zone:N", alt.Tooltip("平均:Q", format=".1f"), "站点数:Q"],
    )
    .properties(height=280)
)
st.altair_chart(zone_chart, width="stretch")

st.subheader("插值留一法")
if len(stats) < 4:
    st.markdown(
        '<div class="empty-box"><h3>站点不够做留一法</h3>'
        "<p>至少留下 4 个有数值的站点，才能比较逆距离加权和最近邻。</p></div>",
        unsafe_allow_html=True,
    )
else:
    scores = loocv_idw_nn(stats["lat"], stats["lon"], stats["value"], power=IDW_POWER)
    if not scores.get("ok"):
        empty_state("days")
    else:
        better = "逆距离加权" if scores["idw_rmse"] <= scores["nn_rmse"] else "最近邻"
        c1, c2 = st.columns(2)
        c1.metric("逆距离加权 RMSE", f"{scores['idw_rmse']:.2f}")
        c2.metric("最近邻 RMSE", f"{scores['nn_rmse']:.2f}")
        st.caption(
            f"幂次 {IDW_POWER}，每次拿掉一个站点再用其余站点预测它。"
            f"这次误差更小的是{better}。MAE：插值 {scores['idw_mae']:.2f}，最近邻 {scores['nn_mae']:.2f}。"
        )

st.subheader("站点数值")
table = stats.rename(columns={"zh": "站点", "zone": "分区", "value": "平均", "n_days": "有效天数"})
st.dataframe(table[["站点", "分区", "平均", "有效天数", "lat", "lon"]], hide_index=True, width="stretch")
