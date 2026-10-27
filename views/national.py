"""全国城市日值：地图、区域和时间都按当前筛选重算。"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.china_settings import (
    DAY_LIMITS,
    POLLUTANT_COLUMNS,
    POLLUTANT_UNITS,
    REGION_ORDER,
)
from src.paths import CHINA_DAILY_PATH, CHINA_FINDINGS_MD, CHINA_META_PATH
from src.ui import cards, hero

POLLUTANTS = list(POLLUTANT_COLUMNS)


@st.cache_data(show_spinner="正在读取全国城市日值…")
def load_national() -> pd.DataFrame | None:
    if not CHINA_DAILY_PATH.exists() or not CHINA_META_PATH.exists():
        return None
    daily = pd.read_parquet(CHINA_DAILY_PATH)
    meta = pd.read_parquet(CHINA_META_PATH)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.merge(meta, on="city", how="left")


def _filter(frame: pd.DataFrame, years: list[int], season: str, region: str, cities: list[str]) -> pd.DataFrame:
    out = frame.loc[frame["date"].dt.year.isin(years)].copy()
    if season == "冬":
        out = out.loc[out["date"].dt.month.isin([12, 1, 2])]
    elif season == "夏":
        out = out.loc[out["date"].dt.month.isin([6, 7, 8])]
    if region != "全部":
        out = out.loc[out["region"] == region]
    if cities:
        out = out.loc[out["city"].isin(cities)]
    return out


def _color(value: float, pollutant: str) -> str:
    if value != value:
        return "#C8C2B6"
    if pollutant == "PM2.5":
        if value <= 35:
            return "#1F7A4D"
        if value <= 75:
            return "#7CB342"
        if value <= 115:
            return "#F9A825"
        if value <= 150:
            return "#EF6C00"
        if value <= 250:
            return "#E53935"
        return "#6A1B9A"
    limit = DAY_LIMITS[pollutant]
    if value <= limit * 0.5:
        return "#1F7A4D"
    if value <= limit:
        return "#C48A12"
    return "#B42318"


def _city_map(points: pd.DataFrame, pollutant: str):
    import folium

    fmap = folium.Map(location=[35.2, 104.2], zoom_start=4, tiles="CartoDB positron")
    for row in points.itertuples(index=False):
        value = row.value
        text = row.city if value != value else f"{row.city} {value:.1f}（{int(row.n_days)} 天）"
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6.5,
            color=_color(float(value) if value == value else float("nan"), pollutant),
            fill=True,
            fill_opacity=0.86,
            weight=0,
            tooltip=text,
        ).add_to(fmap)
    return fmap


def _chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


data = load_national()
if data is None:
    hero("全国城市空气质量", "全国层还没有准备好。", kicker="监测城市日值")
    st.markdown(
        """
        <div class="error-box">
          <h3>还没有全国城市日值</h3>
          <p>北京 12 站的页面仍然可以看。全国层需要先把监测总站的城市小时文件汇总成日值。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("python scripts/prepare_china.py", language="text")
    st.stop()

span = f"{data['date'].min().date().isoformat()} 至 {data['date'].max().date().isoformat()}"
hero(
    "全国城市空气质量",
    "每个点是一个发布了日值的城市，不是县，也不是人口暴露。筛选改变后，下面的数字会重算。",
    kicker=f"监测城市日值 · {span}",
)

st.sidebar.header("全国筛选")
st.sidebar.caption("城市等权。不选具体城市时，使用当前区域里的全部城市。")
pollutant = st.sidebar.selectbox("指标", POLLUTANTS, index=0, key="cn_pollutant")
column = POLLUTANT_COLUMNS[pollutant]
unit = POLLUTANT_UNITS[pollutant]
years = sorted(int(year) for year in data["date"].dt.year.unique())
chosen_years = st.sidebar.multiselect("年份", years, default=[years[-1]], key="cn_years", placeholder="请选择年份")
region = st.sidebar.selectbox("区域", ["全部", *REGION_ORDER], key="cn_region")
season = st.sidebar.radio("季节", ["全年", "冬", "夏"], horizontal=True, key="cn_season")
region_cities = sorted(data.loc[data["region"] == region, "city"].unique()) if region != "全部" else sorted(data["city"].unique())
chosen_cities = st.sidebar.multiselect(
    "城市", region_cities, default=[], key=f"cn_cities_{region}", placeholder="不选则使用全部"
)
one_day = st.sidebar.checkbox("地图改看单独一天", key="cn_one_day")
day = None
if one_day and chosen_years:
    pool = data.loc[data["date"].dt.year.isin(chosen_years), "date"]
    if len(pool):
        day = st.sidebar.date_input(
            "这一天",
            value=pool.max().date(),
            min_value=pool.min().date(),
            max_value=pool.max().date(),
            key="cn_day",
        )
st.sidebar.caption(f"全库 {data['city'].nunique()} 个城市名 · {data['date'].min().date()} 至 {data['date'].max().date()}")
st.sidebar.caption("日均值要求当天至少 18 个有效小时。臭氧用日最大 8 小时，一氧化碳单位是毫克/立方米。")

if not chosen_years:
    st.markdown(
        '<div class="empty-box"><h3>还没有选择年份</h3><p>至少选一个年份后，地图和指标会按这个范围重算。</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

view = _filter(data, chosen_years, season, region, chosen_cities)
values = view.dropna(subset=[column])
if values.empty:
    st.markdown(
        '<div class="empty-box"><h3>这个筛选下没有有效城市日</h3><p>可以改区域、清空城市，或换一个指标。有的城市在有的年份还没有发布。</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

city_mean = values.groupby("city")[column].agg(value="mean", n_days="size").reset_index()
city_mean = city_mean.merge(data[["city", "lat", "lon", "region", "province"]].drop_duplicates("city"), on="city", how="left")
limit = DAY_LIMITS[pollutant]
over_rate = float((values[column] > limit).mean())
ranked = city_mean.sort_values("value", ascending=False)
top = ranked.iloc[0]
low = ranked.iloc[-1]
mean_value = float(city_mean["value"].mean())
cards([
    {"label": "城市等权平均", "value": f"{mean_value:.1f}" if pollutant != "CO" else f"{mean_value:.2f}", "note": unit, "accent": _color(mean_value, pollutant)},
    {"label": "有有效日的城市", "value": f"{len(city_mean)}", "note": f"有效城市日 {len(values):,}"},
    {"label": f"超过 {limit:g} 的城市日", "value": f"{over_rate * 100:.1f}%", "note": "分母是当前筛选里的有效城市日", "accent": "#B42318" if over_rate > 0.2 else "#0B6E4F"},
    {"label": "最高 / 最低", "value": f"{top['city']}", "note": f"{top['value']:.1f} 与 {low['city']} {low['value']:.1f}"},
])
st.markdown(
    f"当前筛选下，{pollutant} 的城市等权平均是 **{mean_value:.1f} {unit}**。"
    f"最高的是 {top['city']}（{top['value']:.1f}，{int(top['n_days'])} 天），最低的是 {low['city']}（{low['value']:.1f}）。"
    f"这是城市等权，小城市和大城市占同样的分量，所以它不是人群暴露。"
)

map_source = city_mean.dropna(subset=["lat", "lon", "value"])
map_title = "地图上是这段筛选里每个城市的平均"
if one_day and day is not None:
    day_rows = values.loc[values["date"] == pd.Timestamp(day)]
    map_source = (
        day_rows.groupby("city")[column]
        .agg(value="mean", n_days="size")
        .reset_index()
        .merge(data[["city", "lat", "lon", "region"]].drop_duplicates("city"), on="city", how="left")
        .dropna(subset=["lat", "lon", "value"])
    )
    map_title = f"地图只看 {day.isoformat()} 当天的城市值"
st.subheader(map_title)
st.caption("坐标是城市代表点，不是监测站。没有坐标的城市不会画出来。颜色只帮助读图，PM2.5 按 35 / 75 / 115 / 150 / 250 分段。")
missing_xy = int(city_mean["lat"].isna().sum())
if missing_xy:
    st.caption(f"有 {missing_xy} 个城市没有匹配到代表点，只出现在表里，不进入地图。")
if map_source.empty:
    st.markdown(
        '<div class="empty-box"><h3>这一天或这些城市没有可画的点</h3><p>换一天，或取消「地图改看单独一天」。</p></div>',
        unsafe_allow_html=True,
    )
else:
    try:
        fmap = _city_map(map_source, pollutant)
        from streamlit_folium import st_folium

        signature = f"{pollutant}-{region}-{season}-{tuple(chosen_years)}-{day}-{len(map_source)}-{map_source['value'].sum():.2f}"
        try:
            st_folium(fmap, height=520, use_container_width=True, returned_objects=[], key=signature)
        except TypeError:
            st_folium(fmap, height=520, width=900, returned_objects=[], key=signature)
    except Exception as exc:
        st.markdown(
            f'<div class="error-box"><h3>地图没有画出来</h3><p>{exc}</p></div>',
            unsafe_allow_html=True,
        )

left, right = st.columns(2)
month = values.copy()
month["月份"] = month["date"].dt.month
monthly = month.groupby("月份")[column].mean().rename("浓度").reset_index()
with left:
    st.subheader("各月平均")
    st.caption("先在城市日上等权，再按月平均。冬夏筛选打开时，图里只会剩下对应的月份。")
    _chart(
        alt.Chart(monthly)
        .mark_line(point=True, color="#0B6E4F")
        .encode(
            x=alt.X("月份:O", title=None, sort="ascending"),
            y=alt.Y("浓度:Q", title=f"{pollutant}（{unit}）"),
            tooltip=[alt.Tooltip("月份:O"), alt.Tooltip("浓度:Q", format=".1f")],
        )
        .properties(height=280)
    )
with right:
    st.subheader("城市平均的两端")
    ends = pd.concat([ranked.head(8), ranked.tail(8)]).drop_duplicates("city")
    ends = ends.rename(columns={"city": "城市", "value": "浓度"})
    _chart(
        alt.Chart(ends)
        .mark_bar(color="#0B6E4F")
        .encode(
            x=alt.X("浓度:Q", title=f"{pollutant}（{unit}）"),
            y=alt.Y("城市:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("城市:N"), alt.Tooltip("浓度:Q", format=".1f")],
        )
        .properties(height=280)
    )

if CHINA_FINDINGS_MD.exists():
    with st.expander("全时段预先算好的结论（不随筛选改变）"):
        st.markdown(CHINA_FINDINGS_MD.read_text(encoding="utf-8"))
