"""总览：指标卡、当前筛选的时序，以及全周期结论。"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.metrics import city_daily, exceedance, format_number, station_means
from src.settings import STANDARDS, station_zh, value_column
from src.ui import cards, empty_state, filtered, get_filters, hero, level_color, load_daily, read_findings


def _delta_text(current: float, baseline: float | None, pollutant: str) -> str:
    if baseline is None:
        return "—"
    delta = current - baseline
    if pollutant == "CO":
        return f"{delta:+,.0f}"
    return f"{delta:+.1f}"


def _sentences(pollutant, unit, mean_value, n_days, peak_value, peak_date, high_name, high_val, low_name, low_val, exc, winter, summer):
    rate = "没有足够的有效日来计算超标比例。"
    if exc["rate"] is not None:
        rate = f"超过所用标准的有 {exc['n_over']} 天，占有效城市日的 {exc['rate'] * 100:.1f}%。"
    sentences = [
        f"当前筛选下，{pollutant} 的城市日尺度平均是 {format_number(mean_value, pollutant)} {unit}，有效城市日 {n_days} 天。",
        f"日尺度峰值出现在 {peak_date}，为 {format_number(peak_value, pollutant)} {unit}。",
        f"最高的站点是{high_name}（{format_number(high_val, pollutant)}），最低的是{low_name}（{format_number(low_val, pollutant)}）。",
        rate,
    ]
    if len(winter) and len(summer):
        sentences.append(
            f"筛选范围内冬季平均 {format_number(float(winter.mean()), pollutant)}，"
            f"夏季平均 {format_number(float(summer.mean()), pollutant)}。"
        )
    else:
        sentences.append("当前范围没有同时包含冬季和夏季，所以这里不做季节差。")
    return sentences


state = get_filters()
hero("城市空气质量总览", "按站点、年份、日期和污染物重算浓度、超标天数和站点差异。")

if not state.ok:
    empty_state(state.reason)
else:
    hourly, daily = filtered(state)
    pollutant = state.pollutant
    column = value_column(pollutant)
    meta = STANDARDS[pollutant]
    if hourly.empty:
        empty_state("rows")
    elif column not in daily.columns or daily[column].notna().sum() == 0:
        empty_state("days")
    else:
        city = city_daily(daily, pollutant).dropna(subset=[column])
        if city.empty:
            empty_state("city")
        else:
            mean_value = float(city[column].mean())
            peak_idx = city[column].idxmax()
            peak_value = float(city.loc[peak_idx, column])
            peak_date = pd.Timestamp(city.loc[peak_idx, "date"]).date().isoformat()
            exc = exceedance(city, pollutant)
            ranks = station_means(daily, pollutant)
            high = ranks.iloc[0]
            low = ranks.iloc[-1]
            winter = city.loc[city["season"] == "冬", column]
            summer = city.loc[city["season"] == "夏", column]
            full = city_daily(load_daily(), pollutant).dropna(subset=[column])
            full_mean = float(full[column].mean()) if not full.empty else None
            cards([
                {
                    "label": f"{pollutant} 平均",
                    "value": format_number(mean_value, pollutant),
                    "note": f"{meta['unit']} · 城市日",
                    "accent": level_color(mean_value, meta["limit"]),
                },
                {
                    "label": "日尺度峰值",
                    "value": format_number(peak_value, pollutant),
                    "note": peak_date,
                    "accent": level_color(peak_value, meta["limit"]),
                },
                {
                    "label": "超标天数",
                    "value": f"{exc['n_over']} / {exc['n']}",
                    "note": f"标准 {format_number(meta['limit'], pollutant)} {meta['unit']}",
                    "accent": "#B42318" if exc["n_over"] else "#1F7A4D",
                },
                {
                    "label": "最高站点",
                    "value": station_zh(high["station"]),
                    "note": f"{format_number(float(high['value']), pollutant)} {meta['unit']}",
                    "accent": "#B42318",
                },
                {
                    "label": "最低站点",
                    "value": station_zh(low["station"]),
                    "note": f"{format_number(float(low['value']), pollutant)} {meta['unit']}",
                    "accent": "#1F7A4D",
                },
                {
                    "label": "相对全周期",
                    "value": _delta_text(mean_value, full_mean, pollutant),
                    "note": f"全周期全部站点 {format_number(full_mean, pollutant)} {meta['unit']}",
                    "accent": "#3E6B89",
                },
            ])
            st.markdown("##### 当前筛选")
            for sentence in _sentences(
                pollutant, meta["unit"], mean_value, int(len(city)), peak_value, peak_date,
                station_zh(high["station"]), float(high["value"]),
                station_zh(low["station"]), float(low["value"]),
                exc, winter, summer,
            ):
                st.markdown(f"- {sentence}")

            plotted = city.rename(columns={column: "value"})
            line = (
                alt.Chart(plotted)
                .mark_line(color="#0B6E4F")
                .encode(
                    x=alt.X("date:T", title="日期"),
                    y=alt.Y("value:Q", title=f"{pollutant}（{meta['unit']}）"),
                    tooltip=[
                        alt.Tooltip("date:T", title="日期"),
                        alt.Tooltip("value:Q", title=pollutant, format=".1f"),
                        alt.Tooltip("n_stations:Q", title="有效站点数"),
                    ],
                )
                .properties(height=320)
            )
            rule = (
                alt.Chart(pd.DataFrame({"y": [meta["limit"]]}))
                .mark_rule(strokeDash=[6, 4], color="#B42318")
                .encode(y="y:Q")
            )
            st.altair_chart(line + rule, width="stretch")
            st.caption("折线是所选站点合成的城市日序列，红色虚线是该指标使用的二级标准。臭氧为日最大 8 小时平均，其余为日均。")

            rank_df = ranks.copy()
            rank_df["站点"] = rank_df["station"].map(station_zh)
            bars = (
                alt.Chart(rank_df)
                .mark_bar(color="#0B6E4F")
                .encode(
                    x=alt.X("value:Q", title=f"{pollutant}（{meta['unit']}）"),
                    y=alt.Y("站点:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("站点:N"),
                        alt.Tooltip("value:Q", title="平均", format=".1f"),
                        alt.Tooltip("n_days:Q", title="有效天数"),
                    ],
                )
                .properties(height=max(260, 26 * len(rank_df)))
            )
            st.altair_chart(bars, width="stretch")

with st.expander("全周期结论（与 docs/findings.md 相同，不随筛选改变）", expanded=False):
    st.markdown(read_findings())
