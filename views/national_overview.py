"""全国监测城市总览：筛选、指标和城市代表点地图。"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.national_map import build_value_map
from src.national_ui import (
    METRICS,
    apply_filter,
    cached_national,
    format_metric,
    missing_national_page,
    national_empty,
    render_national_sidebar,
    show_map,
)
from src.ui import cards, hero

hero(
    "全国监测城市",
    "一座城市一个点。这里是日尺度的城市值，不是北京那 12 个国控站的小时过程。",
    kicker="全国城市 · 日尺度 · 2023–2024",
)
st.markdown(
    """
    <div class="story-grid">
      <div class="stat-card"><div class="stat-label">问题</div><div class="story-body">城市之间的颗粒物有没有空间上的聚集，还是只是各自波动。</div></div>
      <div class="stat-card"><div class="stat-label">数据</div><div class="story-body">监测城市的日均。一天至少 18 个有效小时才算数。</div></div>
      <div class="stat-card"><div class="stat-label">方法</div><div class="story-body">地图、区域对比、Moran’s I，以及按时间切开的梯度提升树。</div></div>
      <div class="stat-card"><div class="stat-label">不能说的</div><div class="story-body">监测表不是图像，这里不做目标检测。城市代表点也不能插成街道污染场。</div></div>
    </div>
    """,
    unsafe_allow_html=True,
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
meta = METRICS[metric]
valid = part.dropna(subset=[metric])
if valid.empty:
    national_empty("metric")
    st.stop()

city_stats = (
    valid.groupby(["city", "province", "region", "lat", "lon", "matched"], dropna=False)[metric]
    .agg(value="mean", n_days="size")
    .reset_index()
)
mean_value = float(valid[metric].mean())
median_value = float(city_stats["value"].median())
limit = meta["limit"]
over_note = "这个指标不对着日标准算超标。"
accent = "#0B6E4F"
if limit is not None:
    over = int((valid[metric] > limit).sum())
    over_note = f"城市日超过 {limit:g} {meta['unit']} 的有 {over} 天，占 {over / len(valid) * 100:.1f}%。这是参照，不是评价报告。"
    accent = "#1F7A4D" if mean_value <= limit else "#B42318"
if metric == "O3":
    over_note = "这是小时臭氧的日均，不是日最大 8 小时，所以不拿 160 μg/m³ 来数超标。"
if metric == "AQI":
    over_note = "这是小时 AQI 的日均，不是生态环境部发布的日 AQI。"
if metric == "CO":
    over_note = "全国这份 CO 是毫克/立方米。北京小时表里的 CO 是微克/立方米，两边不能直接比。"

cards([
    {"label": f"{meta['label']} · 城市日平均", "value": format_metric(mean_value, metric), "note": meta["unit"], "accent": accent},
    {"label": "城市等权中位数", "value": format_metric(median_value, metric), "note": "每座城市先求平均，再取中位数", "accent": "#0B6E4F"},
    {"label": "城市数", "value": f"{valid['city'].nunique()}", "note": f"有效城市日 {len(valid):,}", "accent": "#0B6E4F"},
    {"label": "时间", "value": f"{valid['date'].nunique()} 天", "note": f"{state.start.isoformat()} 至 {state.end.isoformat()}", "accent": "#0B6E4F"},
])
st.caption(over_note)

st.subheader("城市代表点")
st.caption(
    "颜色按这个指标自己的固定色阶，方便看相对高低。点与点之间没有插值。"
    + (" PM2.5 的断点对应 HJ 633-2012 的 24 小时分段。" if metric == "PM2.5" else "")
)
mapped = city_stats.loc[city_stats["matched"].astype(bool)].dropna(subset=["lat", "lon", "value"])
unmatched_n = int((~city_stats["matched"].astype(bool)).sum())
if unmatched_n:
    st.caption(f"当前范围里有 {unmatched_n} 个城市名没有对上坐标，表里还在，地图上不画。")
if mapped.empty:
    national_empty("rows")
else:
    try:
        fmap = build_value_map(mapped, metric, meta["unit"])
    except Exception as exc:
        fmap = None
        st.error(f"地图没有渲染成功：{exc}")
    if fmap is None:
        national_empty("rows")
    else:
        signature = f"nat-{metric}-{state.season}-{state.start}-{state.end}-{len(mapped)}-{mapped['value'].sum():.1f}"
        show_map(fmap, signature)

st.subheader("时间趋势")
focus_cities = state.cities[:6]
if focus_cities:
    trend = valid.loc[valid["city"].isin(focus_cities), ["date", "city", metric]].dropna()
    trend = trend.rename(columns={metric: "值", "city": "城市", "date": "日期"})
    chart = (
        alt.Chart(trend)
        .mark_line()
        .encode(
            x=alt.X("日期:T", title=None),
            y=alt.Y("值:Q", title=f"{metric}（{meta['unit']}）"),
            color=alt.Color("城市:N", title=None),
            tooltip=["日期", "城市", "值"],
        )
        .properties(height=320)
    )
    st.caption("侧栏选了城市，所以这里画这几座城市自己的日序列，最多 6 条。")
else:
    trend = valid.groupby("date", as_index=False)[metric].median().rename(columns={metric: "值", "date": "日期"})
    chart = (
        alt.Chart(trend)
        .mark_line(color="#0B6E4F")
        .encode(
            x=alt.X("日期:T", title=None),
            y=alt.Y("值:Q", title=f"{metric} 城市中位数（{meta['unit']}）"),
            tooltip=["日期", "值"],
        )
        .properties(height=320)
    )
    st.caption("没有指定城市时，画的是当天各城市的中位数，避免三百条线叠在一起。")
st.altair_chart(chart, width="stretch")

rank = city_stats.sort_values("value", ascending=False).head(8)
rank["值"] = rank["value"].map(lambda item: format_metric(item, metric))
st.subheader("当前筛选里较高的城市")
st.dataframe(
    rank.loc[:, ["city", "province", "值", "n_days"]].rename(
        columns={"city": "城市", "province": "省份", "n_days": "有效日"}
    ),
    hide_index=True,
    width="stretch",
)
full_mean = summary["pm25"]["city_day_mean"]
st.caption(
    f"全样本、不随侧栏变化的 PM2.5 城市日平均是 {full_mean} μg/m³。"
    "换省份、城市、指标或日期后，上面的卡片和地图会变，这一句不变。"
)
