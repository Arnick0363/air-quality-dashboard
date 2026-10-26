"""全周期分析摘要。文案里的数字全部来自传入的计算结果。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import (
    city_daily,
    diurnal_profile,
    exceedance,
    find_episodes,
    pearson,
    station_means,
)
from src.settings import (
    COMPLETE_YEARS,
    EPISODE_PM25,
    PM25_BANDS,
    POLLUTANTS,
    SEASON_ORDER,
    STANDARDS,
    STATIONS,
    pm25_category,
    station_zh,
    value_column,
)
from src.spatial import cluster_stations, loocv_idw_nn


def _round(value, digits=1):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def _mean(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return None
    return float(series.mean())


def build_summary(hourly: pd.DataFrame, daily: pd.DataFrame, cleaning: dict) -> tuple[dict, dict]:
    city = city_daily(daily, "PM2.5")
    col = "PM2.5"
    valid = city.dropna(subset=[col]).copy()
    stations = station_means(daily, "PM2.5")
    stations["zh"] = stations["station"].map(station_zh)
    stations["zone"] = stations["station"].map(lambda sid: STATIONS[sid]["zone"])

    by_season = valid.groupby("season")[col].agg(["mean", "count"])
    winter = float(by_season.loc["冬", "mean"]) if "冬" in by_season.index else None
    summer = float(by_season.loc["夏", "mean"]) if "夏" in by_season.index else None
    ratio = None if not winter or not summer else winter / summer

    by_year = []
    for year, part in valid.groupby("year"):
        by_year.append({
            "year": int(year),
            "mean": _round(part[col].mean()),
            "n_days": int(part[col].notna().sum()),
            "complete": int(year) in COMPLETE_YEARS,
        })

    by_month = (
        valid.groupby("month")[col].mean().rename("mean").reset_index().sort_values("mean")
    )
    month_low = month_high = None
    if not by_month.empty:
        low = by_month.iloc[0]
        high = by_month.iloc[-1]
        month_low = {"month": int(low["month"]), "mean": _round(low["mean"])}
        month_high = {"month": int(high["month"]), "mean": _round(high["mean"])}

    categories = {name: 0 for _, name in PM25_BANDS}
    for value in valid[col]:
        categories[pm25_category(float(value))] += 1

    exc75 = exceedance(valid, "PM2.5")
    heavy = valid[col] > EPISODE_PM25
    episodes = find_episodes(valid, col, EPISODE_PM25)
    episode_days = int(sum(item["days"] for item in episodes))
    all_wspm = _mean(valid["WSPM"]) if "WSPM" in valid.columns else None
    all_dewp = _mean(valid["DEWP"]) if "DEWP" in valid.columns else None
    def _weighted(key: str):
        pairs = [(item[key], item["days"]) for item in episodes if key in item]
        if not pairs:
            return None
        values, weights = zip(*pairs)
        return float(np.average(values, weights=weights))

    episode_wspm = _weighted("wspm")
    episode_dewp = _weighted("dewp")

    diurnal = diurnal_profile(hourly, "PM2.5")
    peak = trough = None
    if not diurnal.empty:
        peak_row = diurnal.loc[diurnal["value"].idxmax()]
        trough_row = diurnal.loc[diurnal["value"].idxmin()]
        peak = {"hour": int(peak_row["hour"]), "value": _round(peak_row["value"])}
        trough = {"hour": int(trough_row["hour"]), "value": _round(trough_row["value"])}

    weekend = _mean(valid.loc[valid["date"].dt.dayofweek >= 5, col])
    weekday = _mean(valid.loc[valid["date"].dt.dayofweek < 5, col])

    pollutant_means = []
    for name in POLLUTANTS:
        series = city_daily(daily, name)
        value_col = value_column(name)
        pollutant_means.append({
            "name": name,
            "mean": _round(_mean(series[value_col]), 1 if name != "CO" else 0),
            "unit": STANDARDS[name]["unit"],
            "agg": STANDARDS[name]["agg"],
            "limit": STANDARDS[name]["limit"],
            **{k: exceedance(series.dropna(subset=[value_col]), name)[k] for k in ("n", "n_over", "rate")},
        })

    cluster = cluster_stations(daily)
    station_for_idw = stations.dropna(subset=["value"])
    loocv = loocv_idw_nn(
        [STATIONS[sid]["lat"] for sid in station_for_idw["station"]],
        [STATIONS[sid]["lon"] for sid in station_for_idw["station"]],
        station_for_idw["value"].to_numpy(),
    )

    highest = stations.iloc[0]
    lowest = stations.iloc[-1]
    summary = {
        "period": {
            "start": pd.Timestamp(hourly["datetime"].min()).isoformat(sep=" "),
            "end": pd.Timestamp(hourly["datetime"].max()).isoformat(sep=" "),
            "rows": int(len(hourly)),
            "stations": int(hourly["station"].nunique()),
            "city_rule": "城市日至少 8 个站点有有效日值",
            "valid_day_rule": "站点日至少 18 个有效小时",
        },
        "cleaning": {
            "rows_in": cleaning.get("rows_in"),
            "rows_out": cleaning.get("rows_out"),
            "interp_limit_hours": cleaning.get("interp_limit_hours"),
            "missing_rate_before_interp": cleaning.get("missing_rate_before_interp"),
            "missing_rate_after_interp": cleaning.get("missing_rate_after_interp"),
            "pm25_gt_pm10_rate_among_pairs": cleaning.get("pm25_gt_pm10_rate_among_pairs"),
            "pm25_gt_pm10_rows": cleaning.get("pm25_gt_pm10_rows"),
            "rules": cleaning.get("rules"),
        },
        "pm25": {
            "city_valid_days": int(len(valid)),
            "mean": _round(valid[col].mean()),
            "winter_mean": _round(winter),
            "summer_mean": _round(summer),
            "winter_days": int(by_season.loc["冬", "count"]) if "冬" in by_season.index else 0,
            "summer_days": int(by_season.loc["夏", "count"]) if "夏" in by_season.index else 0,
            "winter_summer_ratio": _round(ratio, 2) if ratio is not None else None,
            "highest_station": highest["zh"],
            "highest_value": _round(highest["value"]),
            "highest_days": int(highest["n_days"]),
            "lowest_station": lowest["zh"],
            "lowest_value": _round(lowest["value"]),
            "lowest_days": int(lowest["n_days"]),
            "stations": stations.assign(
                value=lambda frame: frame["value"].round(1),
            )[["station", "zh", "zone", "value", "n_days"]].to_dict(orient="records"),
            "days_over_75": int(exc75["n_over"]),
            "over_75_rate": _round(exc75["rate"], 3) if exc75["rate"] is not None else None,
            "days_over_150": int(heavy.sum()),
            "over_150_rate": _round(float(heavy.mean()), 3) if len(valid) else None,
            "categories": categories,
            "years": by_year,
            "month_high": month_high,
            "month_low": month_low,
            "corr_wspm": pearson(valid[col], valid["WSPM"]) if "WSPM" in valid else None,
            "corr_dewp": pearson(valid[col], valid["DEWP"]) if "DEWP" in valid else None,
            "corr_temp": pearson(valid[col], valid["TEMP"]) if "TEMP" in valid else None,
            "diurnal_peak": peak,
            "diurnal_trough": trough,
            "weekday_mean": _round(weekday),
            "weekend_mean": _round(weekend),
            "episode_threshold": EPISODE_PM25,
            "episode_count": len(episodes),
            "episode_days": episode_days,
            "episode_wspm": _round(episode_wspm, 2) if episode_wspm is not None else None,
            "all_wspm": _round(all_wspm, 2) if all_wspm is not None else None,
            "episode_dewp": _round(episode_dewp, 2) if episode_dewp is not None else None,
            "all_dewp": _round(all_dewp, 2) if all_dewp is not None else None,
            "top_episodes": episodes[:8],
        },
        "pollutants": pollutant_means,
        "cluster": None if cluster is None else {
            "k": cluster["k"],
            "silhouette": _round(cluster["silhouette"], 3),
            "silhouette_by_k": {k: _round(v, 3) for k, v in cluster["silhouette_by_k"].items()},
            "used_winter_summer_ratio": cluster["used_winter_summer_ratio"],
            "members": cluster["members"],
            "feature_columns": cluster["feature_columns"],
        },
        "loocv": {key: (_round(val, 2) if isinstance(val, float) else val) for key, val in loocv.items()},
    }
    extras = {"city": valid, "daily": daily, "hourly": hourly, "cluster": cluster, "stations": stations}
    return summary, extras


def _pct(rate) -> str:
    if rate is None:
        return "—"
    return f"{rate * 100:.1f}%"


def _corr_text(item: dict | None) -> str:
    if not item:
        return "样本不足"
    return f"r={item['r']:.2f}（n={item['n']}）"


def render_findings_md(summary: dict) -> str:
    p = summary["pm25"]
    period = summary["period"]
    cleaning = summary["cleaning"]
    before = cleaning["missing_rate_before_interp"]["PM2.5"]
    after = cleaning["missing_rate_after_interp"]["PM2.5"]
    lines = [
        "# 全周期分析结论",
        "",
        "数字由 `scripts/prepare_data.py` 从清洗后的小时数据计算，不是手填估计。",
        f"时段：{period['start']} 至 {period['end']}。小时记录 {period['rows']} 条，站点 {period['stations']} 个。",
        f"站点日有效规则：{period['valid_day_rule']}。{period['city_rule']}。",
        "",
        "## 可以直接引用的发现",
        "",
    ]
    if p["winter_mean"] is not None and p["summer_mean"] is not None and p["winter_summer_ratio"] is not None:
        winter_sentence = (
            f"1. 冬季城市日均 PM2.5 为 {p['winter_mean']} μg/m³（{p['winter_days']} 天），"
            f"夏季为 {p['summer_mean']} μg/m³（{p['summer_days']} 天），"
            f"冬季是夏季的 {p['winter_summer_ratio']} 倍。"
        )
        if p["month_high"] and p["month_low"]:
            winter_sentence += (
                f"跨年平均后，最高的月份是 {p['month_high']['month']} 月"
                f"（{p['month_high']['mean']}），最低是 {p['month_low']['month']} 月"
                f"（{p['month_low']['mean']}）。"
            )
        lines.append(winter_sentence)

    year_map = {item["year"]: item for item in p["years"]}
    if 2014 in year_map and 2016 in year_map:
        y2014 = year_map[2014]["mean"]
        y2016 = year_map[2016]["mean"]
        delta = y2016 - y2014
        pct = delta / y2014 * 100 if y2014 else float("nan")
        lines.append(
            f"2. 只比较完整年份：2014 年城市日均 PM2.5 为 {y2014} μg/m³"
            f"（{year_map[2014]['n_days']} 天），2016 年为 {y2016} μg/m³"
            f"（{year_map[2016]['n_days']} 天），相差 {delta:+.1f} μg/m³（{pct:+.1f}%）。"
            "2013 年从 3 月开始，2017 年只到 2 月，不放进这次年均比较。"
        )

    lines.append(
        f"3. 全周期城市日均 PM2.5 为 {p['mean']} μg/m³，共 {p['city_valid_days']} 个有效城市日。"
        f"超过 24 小时二级标准 75 μg/m³ 的有 {p['days_over_75']} 天（{_pct(p['over_75_rate'])}），"
        f"超过 150 μg/m³ 的有 {p['days_over_150']} 天（{_pct(p['over_150_rate'])}）。"
        f"日均超过 150 的连续日合并后，得到 {p['episode_count']} 次重度过程、合计 {p['episode_days']} 天。"
    )
    if p["top_episodes"]:
        top = p["top_episodes"][0]
        lines.append(
            f"   峰值最高的一次是 {top['start']} 至 {top['end']}，"
            f"过程内城市日均峰值 {top['peak']:.1f} μg/m³（{top['peak_date']}）。"
        )
    if p["episode_wspm"] is not None and p["all_wspm"] is not None:
        lines.append(
            f"   这些重度过程按持续天数加权的平均风速是 {p['episode_wspm']} m/s，"
            f"全部有效城市日的平均风速是 {p['all_wspm']} m/s；"
            f"过程期间平均露点 {p['episode_dewp']} °C，全周期平均露点 {p['all_dewp']} °C。"
        )
    lines.append(
        f"4. 站点日均 PM2.5 最高的是{p['highest_station']}（{p['highest_value']} μg/m³，{p['highest_days']} 天），"
        f"最低的是{p['lowest_station']}（{p['lowest_value']} μg/m³，{p['lowest_days']} 天）。"
    )
    lines.append(
        f"5. 城市日尺度上，PM2.5 与风速 {_corr_text(p['corr_wspm'])}，"
        f"与露点 {_corr_text(p['corr_dewp'])}，与气温 {_corr_text(p['corr_temp'])}。"
        "日序列存在自相关，相关不是独立样本检验，这里只报告关联强度。"
    )
    if p["diurnal_peak"] and p["diurnal_trough"]:
        lines.append(
            f"6. 小时平均的日变化里，PM2.5 峰值在 {p['diurnal_peak']['hour']} 时"
            f"（{p['diurnal_peak']['value']} μg/m³），谷值在 {p['diurnal_trough']['hour']} 时"
            f"（{p['diurnal_trough']['value']} μg/m³）。"
            f"城市日均的工作日为 {p['weekday_mean']}，周末为 {p['weekend_mean']}。"
        )
    if summary["cluster"]:
        cluster = summary["cluster"]
        member_text = "；".join(
            f"{name}：{'、'.join(members)}" for name, members in cluster["members"].items()
        )
        lines.append(
            f"7. 层次聚类（ward，轮廓系数在 k=2..5 中选最大）取 k={cluster['k']}，"
            f"轮廓系数 {cluster['silhouette']}。分组：{member_text}。"
            + (
                "特征含六项污染物均值、风速，以及冬夏 PM2.5 比值。"
                if cluster["used_winter_summer_ratio"]
                else "当前样本没有同时覆盖冬夏，聚类没有使用冬夏比值。"
            )
        )
    loocv = summary["loocv"]
    if loocv.get("ok"):
        better = "逆距离加权" if loocv["idw_rmse"] <= loocv["nn_rmse"] else "最近邻"
        lines.append(
            f"8. 用 12 站 PM2.5 平均浓度做留一法：逆距离加权（幂次 2）RMSE 为 {loocv['idw_rmse']}，"
            f"最近邻 RMSE 为 {loocv['nn_rmse']}，误差更小的是{better}。"
            "站点只有 12 个，这个比较说明插值相对最近邻是否更贴近留下的站点，不能当成精细污染地图。"
        )
    lines += [
        "",
        "## 清洗",
        "",
        f"PM2.5 插值前缺失率 {before * 100:.2f}%，最多填补 6 小时内部缺口后为 {after * 100:.2f}%。",
        "降水和风向不插值。负值、超出物理范围的气象值，以及露点比气温高出 2°C 以上的露点，记为缺失。",
        f"PM2.5 高于 PM10 的小时，占两者都有值的记录的 {cleaning['pm25_gt_pm10_rate_among_pairs'] * 100:.2f}%。"
        "这组矛盾没有删行，只在局限里说明。",
        "",
        "## 各污染物城市日尺度",
        "",
        "| 指标 | 使用的日统计 | 平均 | 标准 | 超标天数 | 有效天数 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["pollutants"]:
        rate_days = item["n_over"]
        lines.append(
            f"| {item['name']} | {item['agg']} | {item['mean']} | {item['limit']} | {rate_days} | {item['n']} |"
        )
    lines += [
        "",
        "臭氧的「平均」和超标天数用日最大 8 小时平均，其他指标用日均。",
        "",
        "## 局限",
        "",
        "- 2013 和 2017 不是完整日历年。",
        "- 短缺口线性插值会低估尖峰，长缺口仍然缺失。",
        "- 城市日要求至少 8 个站点，覆盖不足的日期不进入城市序列。",
        "- 相关和季节对比没有做考虑自相关的推断。",
        "- 空间插值是描述工具。40 km 以外的格点不着色。",
        "- 站点坐标来自公开文献中的国控点位置，不是本次实测。",
        "",
    ]
    return "\n".join(lines) + "\n"
