"""按年看城市 PM2.5 是否聚在一起。数字来自准备脚本，不随全国页的临时筛选改变。"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from src.paths import CHINA_LOCAL_PATH, CHINA_META_PATH, CHINA_SUMMARY_JSON
from src.ui import cards, hero

COLORS = {
    "高高": "#B42318",
    "低低": "#1F7A4D",
    "高低": "#C48A12",
    "低高": "#2F6F9F",
    "不显著": "#C5BFB4",
}


def _chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


hero(
    "邻近城市像不像",
    "Moran’s I 大于 0，表示年均 PM2.5 相近的城市在地图上也不远。这是空间格局，不是排放因果。",
    kicker="空间自相关 · 8 个最近城市",
)
if not CHINA_SUMMARY_JSON.exists() or not CHINA_LOCAL_PATH.exists():
    st.markdown(
        '<div class="error-box"><h3>还没有空间自相关结果</h3><p>先运行准备脚本。它会用城市代表点做权重，并写出每年的 Moran’s I。</p></div>',
        unsafe_allow_html=True,
    )
    st.code("python scripts/prepare_china.py", language="text")
    st.stop()

summary = json.loads(CHINA_SUMMARY_JSON.read_text(encoding="utf-8"))
annual = pd.DataFrame(summary["moran_annual"])
local = pd.read_parquet(CHINA_LOCAL_PATH)
meta = pd.read_parquet(CHINA_META_PATH)
winter = summary["moran_winter"]
summer = summary["moran_summer"]
cards([
    {"label": "冬季 Moran’s I", "value": "—" if winter.get("I") is None else f"{winter['I']:.3f}", "note": f"{winter.get('n')} 个城市，p={winter.get('p')}"},
    {"label": "夏季 Moran’s I", "value": "—" if summer.get("I") is None else f"{summer['I']:.3f}", "note": f"{summer.get('n')} 个城市，p={summer.get('p')}"},
    {"label": "进入某年计算的门槛", "value": "240 天", "note": "当年有效日少于此数的城市不参加这一年"},
])
st.caption("权重是每个城市最近的 8 个城市，再按行标准化。p 值来自 199 次打乱浓度。局部标签没有做多重比较校正，只能当探索，不能当一张确认过的热点名录。")

plot_rows = annual.dropna(subset=["I"])
if plot_rows.empty:
    st.markdown('<div class="empty-box"><h3>没有足够的城市年来计算 Moran’s I</h3><p>需要至少约 30 个有坐标、且当年有效日够数的城市。</p></div>', unsafe_allow_html=True)
    st.stop()

_chart(
    alt.Chart(plot_rows)
    .mark_line(point=True, color="#0B6E4F")
    .encode(
        x=alt.X("year:O", title=None),
        y=alt.Y("I:Q", title="Moran’s I"),
        tooltip=["year:O", alt.Tooltip("I:Q", format=".3f"), alt.Tooltip("p:Q", format=".3f"), "n:Q"],
    )
    .properties(height=280)
)

years = sorted(int(year) for year in local["year"].unique())
year = st.sidebar.selectbox("看哪一年的局部格局", years, index=len(years) - 1, key="sp_year")
part = local.loc[local["year"] == year].merge(meta[["city", "lat", "lon"]], on="city", how="left")
part = part.dropna(subset=["lat", "lon"])
counts = part["quadrant"].value_counts().to_dict()
st.subheader(f"{year} 年局部格局")
st.caption(
    "高高：自己高、邻居也高。低低：自己低、邻居也低。高低或低高是和邻居不一样。"
    f"这一年不显著 {counts.get('不显著', 0)} 个，高高 {counts.get('高高', 0)} 个，低低 {counts.get('低低', 0)} 个。"
)
if part.empty:
    st.markdown('<div class="empty-box"><h3>这一年没有可画的城市</h3><p>换一个年份。</p></div>', unsafe_allow_html=True)
    st.stop()

try:
    import folium
    from streamlit_folium import st_folium

    fmap = folium.Map(location=[35.2, 104.2], zoom_start=4, tiles="CartoDB positron")
    for row in part.itertuples(index=False):
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color=COLORS.get(row.quadrant, "#C5BFB4"),
            fill=True,
            fill_opacity=0.88,
            weight=0,
            tooltip=f"{row.city} {row.quadrant} {row.value:.1f}",
        ).add_to(fmap)
    signature = f"{year}-{counts}-{part['value'].sum():.1f}"
    try:
        st_folium(fmap, height=520, use_container_width=True, returned_objects=[], key=signature)
    except TypeError:
        st_folium(fmap, height=520, width=900, returned_objects=[], key=signature)
except Exception as exc:
    st.markdown(f'<div class="error-box"><h3>地图没有画出来</h3><p>{exc}</p></div>', unsafe_allow_html=True)

table = annual.rename(columns={"year": "年", "I": "Moran’s I", "p": "p 值", "n": "城市数", "min_days": "最少有效日", "k": "近邻数"})
try:
    st.dataframe(table, hide_index=True, width="stretch")
except TypeError:
    st.dataframe(table, hide_index=True)
