"""仪表盘共用的样式、数据加载和筛选栏。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import streamlit as st

from src.metrics import station_daily
from src.paths import DAILY_PATH, FINDINGS_MD, HOURLY_PATH
from src.settings import POLLUTANTS, STANDARDS, STATIONS, station_zh

EMPTY_COPY = {
    "stations": "还没有选择站点。至少选一个站点后，指标卡和图表会按这个范围重新计算。",
    "years": "还没有选择年份。数据覆盖 2013–2017，其中 2013 与 2017 不是完整日历年。",
    "date": "请选择开始和结束两个日期，并且开始日期不晚于结束日期。",
    "rows": "当前筛选下没有小时记录。可以放宽年份、日期或季节范围。",
    "days": "有小时记录，但这个污染物没有达到「每天至少 18 个有效小时」的日期，所以日序列是空的。",
    "city": "有效站点太少，构不成城市日序列。可以多选几个站点，或放宽时间范围。",
}


@dataclass
class FilterState:
    ok: bool
    reason: str
    station_ids: list[str]
    years: list[int]
    start: date | None
    end: date | None
    season_mode: str
    pollutant: str


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.1rem; padding-bottom: 2.4rem; max-width: 1180px;}
        html, body, [class*="css"] {font-family: "Segoe UI", "Microsoft YaHei", "Noto Sans SC", sans-serif;}
        .hero-kicker {color: #0B6E4F; font-weight: 700; letter-spacing: 0.12em; font-size: 0.78rem; margin-bottom: 0.2rem;}
        .hero-title {font-size: 1.85rem; line-height: 1.25; margin: 0 0 0.35rem 0; color: #1C2830;}
        .hero-sub {color: #5C6B73; font-size: 0.98rem; margin: 0 0 0.8rem 0;}
        .stat-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 4px 0 18px;}
        .stat-card {background: #fff; border: 1px solid #E4E0D6; border-left: 4px solid var(--accent, #0B6E4F);
                    border-radius: 12px; padding: 12px 14px; min-height: 104px;}
        .stat-label {color: #5C6B73; font-size: 0.82rem;}
        .stat-value {color: #1C2830; font-size: 1.55rem; font-weight: 680; letter-spacing: -0.03em; margin-top: 2px;}
        .stat-note {color: #6D7C84; font-size: 0.78rem; margin-top: 4px; line-height: 1.35;}
        .story-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 0 0 16px;}
        .story-body {color: #1C2830; font-size: 0.95rem; line-height: 1.45; margin-top: 4px;}
        .empty-box, .error-box {border-radius: 12px; padding: 18px 18px 14px; margin: 8px 0 16px;}
        .empty-box {background: #F4F1E8; border: 1px dashed #C8BFAE;}
        .error-box {background: #FDECEC; border: 1px solid #F0C2C2;}
        .empty-box h3, .error-box h3 {margin: 0 0 6px 0; color: #1C2830; font-size: 1.05rem;}
        .section-note {color: #5C6B73; font-size: 0.9rem;}
        div[data-testid="stSidebar"] {background: #F7F5F0;}
        @media (max-width: 780px) {
          .hero-title {font-size: 1.35rem;}
          .stat-value {font-size: 1.28rem;}
          .block-container {padding-left: 0.7rem; padding-right: 0.7rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, kicker: str = "北京站点 · 小时尺度 · 2013–2017") -> None:
    st.markdown(
        f'<div class="hero-kicker">{kicker}</div>'
        f'<h1 class="hero-title">{title}</h1>'
        f'<p class="hero-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def cards(items: list[dict]) -> None:
    blocks = []
    for item in items:
        accent = item.get("accent", "#0B6E4F")
        blocks.append(
            f'<div class="stat-card" style="--accent:{accent}">'
            f'<div class="stat-label">{item["label"]}</div>'
            f'<div class="stat-value">{item["value"]}</div>'
            f'<div class="stat-note">{item.get("note", "")}</div></div>'
        )
    st.markdown(f'<div class="stat-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)


def empty_state(reason: str) -> None:
    text = EMPTY_COPY.get(reason, reason)
    st.markdown(
        f'<div class="empty-box"><h3>这个筛选下没有可计算的结果</h3><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def missing_data_page() -> None:
    st.markdown(
        """
        <div class="error-box">
          <h3>还没有处理好的数据文件</h3>
          <p>仪表盘需要 <code>data/processed/hourly.parquet</code>。原始压缩包不会放进版本库，需要先在本地准备一次。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("python scripts/prepare_data.py", language="text")
    st.caption("脚本会下载 UCI 北京多站点数据，清洗后写出小时表、日表、图表和 docs/findings.md。")
    if st.button("重新检查数据文件"):
        st.cache_data.clear()
        st.rerun()


@st.cache_data(show_spinner="正在读取小时数据…")
def load_hourly():
    if not HOURLY_PATH.exists():
        return None
    frame = pd.read_parquet(HOURLY_PATH)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


@st.cache_data(show_spinner="正在整理逐日数据…")
def load_daily():
    if DAILY_PATH.exists():
        frame = pd.read_parquet(DAILY_PATH)
        frame["date"] = pd.to_datetime(frame["date"])
        return frame
    hourly = load_hourly()
    if hourly is None:
        return None
    return station_daily(hourly)


def render_sidebar(hourly: pd.DataFrame) -> FilterState:
    st.sidebar.header("筛选")
    st.sidebar.caption("城市只有北京。年份和日期同时生效，结果取二者交集。")
    st.sidebar.selectbox("城市", ["北京"], key="bj_city")
    years = sorted(int(year) for year in hourly["year"].unique())
    chosen_years = st.sidebar.multiselect("年份", years, default=years, key="bj_years", placeholder="请选择年份")
    station_ids = sorted(STATIONS, key=station_zh)
    chosen_stations = st.sidebar.multiselect(
        "站点",
        station_ids,
        default=station_ids,
        format_func=station_zh,
        key="bj_stations",
        placeholder="请选择站点",
    )
    pollutant = st.sidebar.selectbox(
        "污染物",
        POLLUTANTS,
        format_func=lambda name: f"{name}（{STANDARDS[name]['unit']}）",
        key="bj_pollutant",
    )
    season_mode = st.sidebar.radio("季节范围", ["全年", "冬", "夏"], horizontal=True, key="bj_season_mode")
    min_date = hourly["datetime"].min().date()
    max_date = hourly["datetime"].max().date()
    picked = st.sidebar.date_input(
        "时间范围",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="bj_date_range",
    )
    st.sidebar.caption(
        f"全库 {len(hourly):,} 条小时记录 · {hourly['station'].nunique()} 站 · "
        f"{min_date.isoformat()} 至 {max_date.isoformat()}"
    )
    st.sidebar.caption("2013 年从 3 月开始，2017 年只到 2 月。日均值要求当天至少 18 个有效小时。")

    start = end = None
    if isinstance(picked, (tuple, list)):
        if len(picked) == 2:
            start, end = picked
    elif isinstance(picked, (date, datetime)):
        start = end = picked if isinstance(picked, date) else picked.date()

    if not chosen_stations:
        return FilterState(False, "stations", [], chosen_years, start, end, season_mode, pollutant)
    if not chosen_years:
        return FilterState(False, "years", chosen_stations, [], start, end, season_mode, pollutant)
    if start is None or end is None or start > end:
        return FilterState(False, "date", chosen_stations, chosen_years, start, end, season_mode, pollutant)
    return FilterState(True, "", chosen_stations, chosen_years, start, end, season_mode, pollutant)


def prepare_beijing() -> FilterState:
    hourly = load_hourly()
    if hourly is None:
        missing_data_page()
        st.stop()
    state = render_sidebar(hourly)
    st.session_state["filter_state"] = state
    return state


def get_filters() -> FilterState:
    state = st.session_state.get("filter_state")
    if state is None:
        st.error("请在项目根目录运行 `streamlit run app.py`，不要单独打开页面文件。")
        st.stop()
    return state


def read_findings() -> str:
    if FINDINGS_MD.exists():
        return FINDINGS_MD.read_text(encoding="utf-8")
    return "还没有生成 docs/findings.md。运行 python scripts/prepare_data.py 后，这里会显示全周期结论。"


def filtered(state: FilterState):
    from src.query import subset_daily, subset_hourly

    hourly = subset_hourly(
        load_hourly(), state.station_ids, state.years, state.start, state.end, state.season_mode
    )
    daily = subset_daily(
        load_daily(), state.station_ids, state.years, state.start, state.end, state.season_mode
    )
    return hourly, daily


def show_pyplot(fig) -> None:
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig)


def level_color(value: float | None, limit: float) -> str:
    if value is None or value != value:
        return "#8A8175"
    if value <= limit * 0.5:
        return "#1F7A4D"
    if value <= limit:
        return "#C48A12"
    return "#B42318"
