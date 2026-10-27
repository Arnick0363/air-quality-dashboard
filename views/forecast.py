"""次日 PM2.5：基线、岭回归和梯度提升用同一批时间切分样本。"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from src.paths import CHINA_FORECAST_PATH, CHINA_SUMMARY_JSON
from src.ui import cards, hero


def _chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


hero(
    "用昨天预测明天",
    "特征停在目标日的前一天。2024 年的目标日只拿来看，不拿来调参；对外说的误差是 2025 年。",
    kicker="时间切分 · 不是随机划分",
)
if not CHINA_SUMMARY_JSON.exists() or not CHINA_FORECAST_PATH.exists():
    st.markdown(
        '<div class="error-box"><h3>还没有预测结果</h3><p>准备脚本会训练模型并留下测试年的预测。页面本身不重新训练。</p></div>',
        unsafe_allow_html=True,
    )
    st.code("python scripts/prepare_china.py", language="text")
    st.stop()

summary = json.loads(CHINA_SUMMARY_JSON.read_text(encoding="utf-8"))
forecast = summary["forecast"]
test = forecast["splits"].get("test") or {}
valid = forecast["splits"].get("valid") or {}
if not test.get("persistence"):
    st.markdown('<div class="empty-box"><h3>测试年没有样本</h3><p>检查准备脚本是否包含 2025 年的目标日。</p></div>', unsafe_allow_html=True)
    st.stop()

persist = test["persistence"]["mae"]
ridge = test["ridge"]["mae"]
boost = test["boosting"]["mae"]
cards([
    {"label": "明天等于今天", "value": f"{persist:.2f}", "note": "测试集平均绝对误差 μg/m³", "accent": "#8A8175"},
    {"label": "岭回归", "value": f"{ridge:.2f}", "note": "同一批样本", "accent": "#2F6F9F"},
    {
        "label": "梯度提升",
        "value": f"{boost:.2f}",
        "note": f"技能分数 {test['boosting_skill']:.3f}" if test.get("boosting_skill") is not None else "技能分数没有算出",
        "accent": "#0B6E4F",
    },
    {"label": "测试样本", "value": f"{test['n']:,}", "note": f"{test.get('start')} 至 {test.get('end')}"},
])
st.markdown(
    f"2025 年这些城市日里，梯度提升的平均绝对误差是 **{boost:.2f} μg/m³**，"
    f"持续基线是 **{persist:.2f}**。技能分数 "
    f"{'没有算出' if test.get('boosting_skill') is None else format(test['boosting_skill'], '.3f')} "
    "表示相对基线少错了这么大的比例；"
    "它不是因果效应，也不能外推到没有发布的县。"
)
cls_model = test.get("class_boosting") or {}
cls_base = test.get("class_persistence") or {}
if cls_model:
    st.caption(
        f"若只判断下一天是否超过 75 μg/m³：当天已经超标这条规则的精确率 {cls_base.get('precision'):.3f}、召回 {cls_base.get('recall'):.3f}；"
        f"梯度提升的精确率 {cls_model.get('precision'):.3f}、召回 {cls_model.get('recall'):.3f}。"
        f"正样本比例 {cls_model.get('positive_rate'):.3f}，所以不能只看准确率。"
    )

predictions = pd.read_parquet(CHINA_FORECAST_PATH)
predictions["date"] = pd.to_datetime(predictions["date"])
cities = sorted(predictions["city"].unique())
default_city = "北京" if "北京" in cities else cities[0]
city = st.sidebar.selectbox("看哪个城市的测试年", cities, index=cities.index(default_city), key="fc_city")
one = predictions.loc[predictions["city"] == city].sort_values("date")
if one.empty:
    st.markdown('<div class="empty-box"><h3>这个城市在测试年没有预测</h3><p>换一个城市。缺测的下一天不会被当成样本。</p></div>', unsafe_allow_html=True)
else:
    st.subheader(f"{city} · 测试年")
    st.caption("灰线是把昨天的浓度直接当成今天。绿线是梯度提升。两条线都只能使用昨天及以前的信息。")
    long = one.melt(id_vars=["date"], value_vars=["observed", "persistence", "boosting"], var_name="来源", value_name="浓度")
    long["来源"] = long["来源"].map({"observed": "实测", "persistence": "明天等于今天", "boosting": "梯度提升"})
    _chart(
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("浓度:Q", title="PM2.5（μg/m³）"),
            color=alt.Color(
                "来源:N",
                scale=alt.Scale(
                    domain=["实测", "明天等于今天", "梯度提升"],
                    range=["#1C2830", "#B7B1A6", "#0B6E4F"],
                ),
            ),
            tooltip=[alt.Tooltip("date:T", title="日期"), "来源:N", alt.Tooltip("浓度:Q", format=".1f")],
        )
        .properties(height=320)
    )

regions = pd.DataFrame(forecast.get("test_by_region") or [])
if not regions.empty:
    st.subheader("测试误差的区域差异")
    st.caption("还是城市日等权，不是人口加权。某个区域误差更小，只说明这些城市更好猜，不说明模型理解了排放。")
    long_region = regions.melt(id_vars=["region", "n"], value_vars=["persistence_mae", "boosting_mae"], var_name="模型", value_name="误差")
    long_region["模型"] = long_region["模型"].map({"persistence_mae": "明天等于今天", "boosting_mae": "梯度提升"})
    _chart(
        alt.Chart(long_region)
        .mark_bar()
        .encode(
            x=alt.X("region:N", title=None, sort=list(regions.sort_values("boosting_mae")["region"])),
            y=alt.Y("误差:Q", title="平均绝对误差（μg/m³）"),
            color=alt.Color("模型:N", scale=alt.Scale(range=["#B7B1A6", "#0B6E4F"])),
            xOffset="模型:N",
            tooltip=["region:N", "模型:N", alt.Tooltip("误差:Q", format=".2f"), "n:Q"],
        )
        .properties(height=300)
    )

if valid.get("persistence"):
    with st.expander("2024 验证年（没有用来挑模型）"):
        st.write(
            f"验证年样本 {valid['n']}，基线 MAE {valid['persistence']['mae']:.2f}，"
            f"岭回归 {valid['ridge']['mae']:.2f}，梯度提升 {valid['boosting']['mae']:.2f}。"
            "超参数在写代码时就定了，没有按这一年或 2025 年去搜索。"
        )
st.caption("没有把这套表格式监测数据送进 YOLO。YOLO 识别的是图像里的物体框，不是城市浓度。")
