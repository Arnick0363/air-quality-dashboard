"""相关、气象、层次聚类和重度污染过程。"""

from __future__ import annotations

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.figures import setup_font
from src.metrics import city_daily, correlation_matrix, find_episodes, pearson
from src.settings import EPISODE_PM25, METEO, POLLUTANTS, STANDARDS, WIND_ORDER, value_column
from src.spatial import cluster_stations, draw_dendrogram
from src.ui import empty_state, filtered, get_filters, hero, show_pyplot

state = get_filters()
hero("聚类与关联", "看污染物和气象量的关系、站点分组，以及连续超标过程。")

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

city = city_daily(daily, pollutant)
valid = city.dropna(subset=[column]) if column in city.columns else city.iloc[0:0]
if valid.empty:
    empty_state("city")
    st.stop()

setup_font()
matrix_cols = [col for col in POLLUTANTS + METEO if col in valid.columns]
st.subheader("城市日相关")
st.caption("相关在城市日序列上计算。日与日之间并不独立，所以不把 p 值当成严格显著性。")
corr = correlation_matrix(valid, matrix_cols)
fig, ax = plt.subplots(figsize=(8.2, 6.4))
sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, linewidths=0.3, linecolor="white")
ax.set_title("当前筛选的相关系数")
show_pyplot(fig)
plt.close(fig)

focus = []
for meteo in ("WSPM", "DEWP", "TEMP", "RAIN", "PRES"):
    if meteo in valid.columns:
        stat = pearson(valid[column], valid[meteo])
        if stat:
            focus.append({"气象量": meteo, "r": round(stat["r"], 3), "n": stat["n"]})
if focus:
    focus_df = pd.DataFrame(focus).sort_values("r", key=lambda s: s.abs(), ascending=False)
    strongest = focus_df.iloc[0]
    st.markdown(
        f"与 **{pollutant}** 线性相关最强的气象量是 **{strongest['气象量']}**，"
        f"r = {strongest['r']}（n = {int(strongest['n'])}）。"
    )
    st.dataframe(focus_df, hide_index=True, width="stretch")

scatter_cols = [col for col in ("WSPM", "DEWP") if col in valid.columns]
if scatter_cols:
    st.subheader("和风、露点的散点")
    plot_ready = valid.rename(columns={column: "浓度"})
    long = plot_ready.melt(id_vars=["浓度", "date"], value_vars=scatter_cols, var_name="气象量", value_name="气象值")
    scatter = (
        alt.Chart(long.dropna())
        .mark_circle(opacity=0.35, color="#0B6E4F")
        .encode(
            x=alt.X("气象值:Q", title="气象值"),
            y=alt.Y("浓度:Q", title=f"{pollutant}（{unit}）"),
            tooltip=[
                alt.Tooltip("date:T", title="日期"),
                alt.Tooltip("浓度:Q", format=".1f", title=pollutant),
                alt.Tooltip("气象值:Q", format=".2f"),
            ],
        )
        .properties(height=280)
        .facet(column=alt.Column("气象量:N", title=None))
    )
    st.altair_chart(scatter, width="stretch")

if "wd" in hourly.columns:
    st.subheader("风向")
    wind = hourly.dropna(subset=[pollutant, "wd"]).copy()
    wind = wind[wind["wd"].isin(WIND_ORDER)]
    if wind.empty:
        st.caption("当前筛选里没有可用的风向记录。")
    else:
        by_wind = wind.groupby("wd", as_index=False).agg(平均=(pollutant, "mean"), 小时数=(pollutant, "count"))
        by_wind["wd"] = pd.Categorical(by_wind["wd"], WIND_ORDER, ordered=True)
        by_wind = by_wind.sort_values("wd")
        wind_chart = (
            alt.Chart(by_wind)
            .mark_bar(color="#3E6B89")
            .encode(
                x=alt.X("wd:N", sort=WIND_ORDER, title="风向"),
                y=alt.Y("平均:Q", title=f"{pollutant}（{unit}）"),
                tooltip=["wd:N", alt.Tooltip("平均:Q", format=".1f"), "小时数:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(wind_chart, width="stretch")

st.subheader("站点层次聚类")
st.caption("ward 连接，特征是各污染物与风速的站点平均；若冬夏都有数据，再加入 PM2.5 冬夏比值。k 由轮廓系数在 2 到 5 之间选择。")
cluster = cluster_stations(daily)
if cluster is None:
    st.markdown(
        '<div class="empty-box"><h3>当前站点太少，或季节覆盖不够</h3>'
        "<p>层次聚类至少需要 4 个站点有完整特征。请多选几个站点；如果只要冬夏比值，请让时间范围同时包含冬季和夏季。</p></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"选定 **k = {cluster['k']}**，轮廓系数 **{cluster['silhouette']:.3f}**。"
        + ("特征包含冬夏 PM2.5 比值。" if cluster["used_winter_summer_ratio"] else "未使用冬夏比值。")
    )
    for name, members in cluster["members"].items():
        st.markdown(f"- **{name}**：{'、'.join(members)}")
    score_df = pd.DataFrame(
        {"k": [int(k) for k in cluster["silhouette_by_k"]], "轮廓系数": list(cluster["silhouette_by_k"].values())}
    )
    st.dataframe(score_df, hide_index=True, width="stretch")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    draw_dendrogram(cluster, ax)
    ax.set_title("树状图展示距离结构，分组以上面的 k 为准")
    show_pyplot(fig)
    plt.close(fig)

    table = cluster["table"]
    st.caption("下表给出各站所属分组。分组名称按组内 PM2.5 均值从高到低命名。")
    show = table.copy()
    show["站点"] = show["zh"]
    st.dataframe(
        show[["站点", "cluster", "zone", "PM2.5"]].rename(
            columns={"cluster": "分组", "zone": "原功能分区", "PM2.5": "PM2.5 均值"}
        ),
        hide_index=True,
        width="stretch",
    )

st.subheader("连续超标过程")
limit = STANDARDS[pollutant]["limit"]
threshold = EPISODE_PM25 if pollutant == "PM2.5" else limit
episodes = find_episodes(valid, column, threshold)
st.caption(f"把城市日尺度超过 {threshold:g} {unit} 的连续日期并成一次过程。PM2.5 使用 150，作为重度过程；其他指标使用各自的二级标准。")
if not episodes:
    st.markdown(
        '<div class="empty-box"><h3>没有连成过程的超标日</h3>'
        "<p>可以换一个污染物、放宽到全时段，或确认标准线以上确实没有连续天数。</p></div>",
        unsafe_allow_html=True,
    )
else:
    episode_df = pd.DataFrame(episodes)
    rename = {
        "start": "开始", "end": "结束", "days": "天数", "peak": "峰值",
        "peak_date": "峰值日期", "mean": "过程均值", "wspm": "风速", "dewp": "露点", "temp": "气温",
    }
    st.dataframe(episode_df.rename(columns=rename), hide_index=True, width="stretch")
    if "wspm" in episode_df.columns and "WSPM" in valid.columns:
        all_wind = float(valid["WSPM"].mean())
        weights = episode_df["days"].to_numpy()
        epi_wind = float((episode_df["wspm"] * weights).sum() / weights.sum())
        st.caption(f"过程期间按天数加权的平均风速是 {epi_wind:.2f} m/s，全部有效城市日是 {all_wind:.2f} m/s。")
