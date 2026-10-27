"""时间切分的梯度提升树，以及和「直接用昨天」的对比。"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.national_ui import (
    cached_national,
    missing_national_page,
    national_empty,
    render_national_sidebar,
)
from src.ui import cards, hero

hero(
    "预测当日 PM2.5",
    "特征只用昨天和更早的信息。测试集是最后一段日期，没有打乱。",
    kicker="全国城市 · 梯度提升树 · 对照持久性基线",
)

bundle = cached_national()
if bundle is None:
    missing_national_page()
    st.stop()
daily, summary, _lisa, pred = bundle
model = summary.get("model") or {}
if not model:
    st.markdown('<div class="empty-box"><h3>还没有模型结果</h3><p>运行 python scripts/prepare_national.py 之后，这里会给出测试集误差。</p></div>', unsafe_allow_html=True)
    st.stop()

cards([
    {"label": "测试 MAE", "value": f"{model['test_mae']}", "note": "μg/m³ · 梯度提升树", "accent": "#0B6E4F"},
    {"label": "测试 RMSE", "value": f"{model['test_rmse']}", "note": "同一批测试行", "accent": "#0B6E4F"},
    {"label": "持久性 MAE", "value": f"{model['persist_mae']}", "note": "直接用昨天的 PM2.5", "accent": "#8A8175"},
    {"label": "持久性 RMSE", "value": f"{model['persist_rmse']}", "note": f"切分点 {model['cutoff']}", "accent": "#8A8175"},
])
better = model.get("mae_improvement_vs_persist")
if better is None:
    st.caption("没有算出相对持久性基线的变化。")
elif better > 0:
    st.caption(f"测试集上，模型 MAE 比「用昨天」低 {better * 100:.1f}%。训练集 MAE 是 {model['train_mae']}，用来看有没有把训练段背下来。")
else:
    st.caption(f"测试集上，模型 MAE 没有低于持久性基线（相对变化 {better}）。昨天的浓度本身就已经很强。")
st.caption(
    f"训练 {model['train_start']} 至 {model['train_end']}，{model['n_train']} 行；"
    f"测试 {model['test_start']} 至 {model['test_end']}，{model['n_test']} 行；{model['n_cities']} 个城市。"
    "没有放入当天的其他污染物，全国这套表里也没有气象。"
)

importance = pd.DataFrame(model["importance"])
st.subheader("把某个特征打乱后，测试 MAE 增加多少")
st.altair_chart(
    alt.Chart(importance)
    .mark_bar(color="#0B6E4F")
    .encode(
        x=alt.X("mae_increase:Q", title="MAE 增加（μg/m³）"),
        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=160)),
        tooltip=["label", "mae_increase"],
    )
    .properties(height=280),
    width="stretch",
)
st.caption(f"置换重要性用了测试集中的 {model['importance_sample_n']} 行，重复 5 次。这是预测有用，不是因果贡献。")

state = render_national_sidebar(daily)
st.subheader("测试期里一座城市的三条线")
if pred.empty:
    national_empty("file")
    st.stop()
candidates = sorted(pred["city"].unique())
if state.cities:
    preferred = [city for city in state.cities if city in set(candidates)]
    default_city = preferred[0] if preferred else ("北京" if "北京" in candidates else candidates[0])
else:
    default_city = "北京" if "北京" in candidates else candidates[0]
city = st.selectbox("看哪一座城市的测试期", candidates, index=candidates.index(default_city))
one = pred.loc[pred["city"] == city, ["date", "y", "y_model", "y_persist"]].sort_values("date")
if state.ok and state.start and state.end:
    one = one.loc[(one["date"] >= pd.Timestamp(state.start)) & (one["date"] <= pd.Timestamp(state.end))]
if one.empty:
    national_empty("rows")
    st.stop()
long = one.melt("date", ["y", "y_model", "y_persist"], var_name="线", value_name="PM2.5")
long["线"] = long["线"].map({"y": "实际", "y_model": "模型", "y_persist": "昨天"})
st.altair_chart(
    alt.Chart(long)
    .mark_line()
    .encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("PM2.5:Q", title="μg/m³"),
        color=alt.Color("线:N", title=None),
        tooltip=["date", "线", alt.Tooltip("PM2.5:Q", format=".1f")],
    )
    .properties(height=320),
    width="stretch",
)
err_model = (one["y"] - one["y_model"]).abs().mean()
err_persist = (one["y"] - one["y_persist"]).abs().mean()
st.caption(
    f"{city} 在图上这些天的 MAE：模型 {err_model:.1f}，昨天 {err_persist:.1f}。"
    "这是一座城市的切片，不是上面那组全测试集数字。"
)
