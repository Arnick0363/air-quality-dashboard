"""时间趋势、季节、日变化和站点对比。"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.metrics import city_daily, diurnal_profile, station_means
from src.settings import SEASON_ORDER, STANDARDS, station_zh, value_column
from src.ui import empty_state, filtered, get_filters, hero

state = get_filters()
hero("趋势与季节", "看月际变化、季节差、一天里的起伏，以及站与站的差别。")

if not state.ok:
    empty_state(state.reason)
    st.stop()

hourly, daily = filtered(state)
pollutant = state.pollutant
column = value_column(pollutant)
unit = STANDARDS[pollutant]["unit"]
if hourly.empty:
    empty_state("rows")
    st.stop()
if column not in daily.columns or daily[column].notna().sum() == 0:
    empty_state("days")
    st.stop()

city = city_daily(daily, pollutant).dropna(subset=[column])
if city.empty:
    empty_state("city")
    st.stop()

if any(year in state.years for year in (2013, 2017)):
    st.info("2013 年只有 3–12 月，2017 年只有 1–2 月。把它们和完整年的年均放在一起时，差异里有日历不完整的成分。")

monthly = city.copy()
monthly["月份"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
monthly = monthly.groupby("月份", as_index=False).agg(平均=(column, "mean"), 天数=(column, "count"))
month_chart = (
    alt.Chart(monthly)
    .mark_line(point=True, color="#0B6E4F")
    .encode(
        x=alt.X("月份:T", title="月份"),
        y=alt.Y("平均:Q", title=f"{pollutant}（{unit}）"),
        tooltip=["月份:T", alt.Tooltip("平均:Q", format=".1f"), "天数:Q"],
    )
    .properties(height=320)
)
st.subheader("月均变化")
st.altair_chart(month_chart, width="stretch")

season = (
    city.groupby("season", as_index=False)
    .agg(平均=(column, "mean"), 天数=(column, "count"))
)
season["季节"] = pd.Categorical(season["season"], SEASON_ORDER, ordered=True)
season = season.sort_values("季节")
season_chart = (
    alt.Chart(season)
    .mark_bar(color="#3E6B89")
    .encode(
        x=alt.X("季节:N", sort=SEASON_ORDER, title=None),
        y=alt.Y("平均:Q", title=f"{pollutant}（{unit}）"),
        tooltip=["季节:N", alt.Tooltip("平均:Q", format=".1f"), "天数:Q"],
    )
    .properties(height=300)
)
st.subheader("季节平均")
if season.empty:
    empty_state("days")
else:
    st.altair_chart(season_chart, width="stretch")
    bits = [f"{row['season']} {row['平均']:.1f}（{int(row['天数'])} 天）" for _, row in season.iterrows()]
    st.caption("城市日按季节平均：" + "；".join(bits) + "。")

profile = diurnal_profile(hourly, pollutant if pollutant != "O3" else "O3")
st.subheader("平均日变化")
st.caption("这一张用小时平均，不使用日最大 8 小时值，以便看出一天内部的起伏。")
if profile.empty:
    empty_state("rows")
else:
    hour_chart = (
        alt.Chart(profile)
        .mark_line(point=True, color="#C48A12")
        .encode(
            x=alt.X("hour:O", title="小时"),
            y=alt.Y("value:Q", title=f"{pollutant}（{unit}）"),
            tooltip=[alt.Tooltip("hour:O", title="小时"), alt.Tooltip("value:Q", title="平均", format=".1f"),
                     alt.Tooltip("n:Q", title="小时样本数")],
        )
        .properties(height=300)
    )
    st.altair_chart(hour_chart, width="stretch")
    peak = profile.loc[profile["value"].idxmax()]
    trough = profile.loc[profile["value"].idxmin()]
    st.caption(f"峰值在 {int(peak['hour'])} 时（{peak['value']:.1f}），谷值在 {int(trough['hour'])} 时（{trough['value']:.1f}）。")

    weekend = (
        hourly.dropna(subset=[pollutant])
        .groupby(["is_weekend", "hour"], as_index=False)[pollutant]
        .mean()
        .rename(columns={pollutant: "浓度"})
    )
    weekend["类型"] = weekend["is_weekend"].map({True: "周末", False: "工作日"})
    split = (
        alt.Chart(weekend)
        .mark_line()
        .encode(
            x=alt.X("hour:O", title="小时"),
            y=alt.Y("浓度:Q", title=unit),
            color=alt.Color("类型:N", scale=alt.Scale(range=["#0B6E4F", "#C48A12"])),
            tooltip=["类型:N", "hour:O", alt.Tooltip("浓度:Q", format=".1f", title=pollutant)],
        )
        .properties(height=280)
    )
    st.altair_chart(split, width="stretch")

yearly = city.groupby("year", as_index=False).agg(平均=(column, "mean"), 天数=(column, "count"))
yearly["年份"] = yearly["year"].astype(str)
yearly["完整年"] = yearly["year"].isin([2014, 2015, 2016]).map({True: "完整年", False: "不完整年"})
year_chart = (
    alt.Chart(yearly)
    .mark_bar()
    .encode(
        x=alt.X("年份:N", title=None),
        y=alt.Y("平均:Q", title=f"{pollutant}（{unit}）"),
        color=alt.Color("完整年:N", scale=alt.Scale(range=["#0B6E4F", "#C8BFAE"])),
        tooltip=["年份:N", alt.Tooltip("平均:Q", format=".1f"), "天数:Q", "完整年:N"],
    )
    .properties(height=300)
)
st.subheader("分年平均")
st.altair_chart(year_chart, width="stretch")

ranks = station_means(daily, pollutant)
ranks["站点"] = ranks["station"].map(station_zh)
st.subheader("站点对比")
st.dataframe(
    ranks.rename(columns={"value": "平均", "n_days": "有效天数"})[["站点", "平均", "有效天数"]],
    hide_index=True,
    width="stretch",
)
