"""区域和季节对比。区域边界是事先写好的，不是模型分出来的。"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.national_ui import (
    METRICS,
    apply_filter,
    cached_national,
    format_metric,
    missing_national_page,
    national_empty,
    render_national_sidebar,
)
from src.settings import SEASON_ORDER
from src.ui import hero

FOCUS = ["华北", "长三角", "珠三角", "成渝", "西北"]

hero(
    "区域与季节",
    "先看事先划定的区域，再看春夏秋冬。换侧栏之后，这页的柱子会跟着变。",
    kicker="全国城市 · 日尺度 · 区域是定义，不是聚类",
)
st.caption(
    "华北是京津冀晋和内蒙古，长三角是沪苏浙皖，珠三角只含广州、深圳、佛山、东莞、中山、珠海、惠州、江门、肇庆，"
    "成渝是四川和重庆，西北是陕甘青宁新。广东其他城市不进珠三角。"
)

bundle = cached_national()
if bundle is None:
    missing_national_page()
    st.stop()
daily, summary, _lisa, _pred = bundle
state = render_national_sidebar(daily)
if not state.ok:
    national_empty(state.reason)
    st.stop()
part = apply_filter(daily, state)
metric = state.metric
unit = METRICS[metric]["unit"]
valid = part.dropna(subset=[metric])
if valid.empty:
    national_empty("metric")
    st.stop()

region_order = [name for name in FOCUS if name in set(valid["region"])]
extras = sorted(set(valid["region"]) - set(region_order))
region_order.extend(extras)
region_mean = (
    valid.groupby("region", as_index=False)[metric]
    .mean()
    .rename(columns={metric: "均值", "region": "区域"})
)
region_mean["区域"] = pd.Categorical(region_mean["区域"], categories=region_order, ordered=True)
st.subheader("区域平均")
st.altair_chart(
    alt.Chart(region_mean.dropna(subset=["区域"]))
    .mark_bar(color="#0B6E4F")
    .encode(
        x=alt.X("区域:N", sort=region_order, title=None),
        y=alt.Y("均值:Q", title=f"{metric}（{unit}）"),
        tooltip=["区域", alt.Tooltip("均值:Q", format=".1f")],
    )
    .properties(height=320),
    width="stretch",
)

season_rows = (
    valid.groupby(["region", "season"], as_index=False)[metric]
    .mean()
    .rename(columns={metric: "均值", "region": "区域", "season": "季节"})
)
season_rows = season_rows.loc[season_rows["区域"].isin(region_order)]
st.subheader("同一区域内的季节")
if season_rows.empty:
    national_empty("rows")
else:
    st.altair_chart(
        alt.Chart(season_rows)
        .mark_bar()
        .encode(
            x=alt.X("季节:N", sort=SEASON_ORDER, title=None),
            y=alt.Y("均值:Q", title=f"{metric}（{unit}）"),
            color=alt.Color("区域:N", title=None),
            xOffset="区域:N",
            tooltip=["区域", "季节", alt.Tooltip("均值:Q", format=".1f")],
        )
        .properties(height=340),
        width="stretch",
    )

table = []
for region in region_order:
    block = valid.loc[valid["region"] == region, metric]
    row = {
        "区域": region,
        "城市日平均": format_metric(float(block.mean()) if len(block) else None, metric),
        "城市数": int(valid.loc[valid["region"] == region, "city"].nunique()),
        "城市日": int(len(block)),
    }
    for season in SEASON_ORDER:
        season_part = valid.loc[(valid["region"] == region) & (valid["season"] == season), metric]
        row[season] = format_metric(float(season_part.mean()) if len(season_part) else None, metric)
    table.append(row)
st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
st.caption("格子里的数是当前筛选下的城市日平均。冬天的样本量和夏天不一定一样，所以冬夏比只在两边都有数时才有意义。")

with st.expander("全样本结论里的区域数字（不随侧栏变化）"):
    rows = summary.get("regions") or []
    if not rows:
        st.write("还没有全样本区域表。")
    else:
        show = pd.DataFrame(rows).rename(columns={
            "region": "区域",
            "mean": "PM2.5",
            "winter_mean": "冬",
            "summer_mean": "夏",
            "winter_summer_ratio": "冬夏比",
            "n_cities": "城市数",
            "n_city_days": "城市日",
        })
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("这张表固定是 PM2.5，方便和文档里的句子对照。上面的图才跟着侧栏的指标走。")
