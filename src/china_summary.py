"""把全国层算出的数字收成一份说明。文案只引用这份字典。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.autocorr import annual_global_moran, annual_local_moran, season_moran
from src.china_settings import DAY_LIMITS, LONG_TERM_MIN_DAYS, REGION_ORDER, YEAR_MIN_DAYS
def _round(value, digits=1):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def coverage_stats(daily: pd.DataFrame, meta: pd.DataFrame, file_report: dict) -> dict:
    located = meta.dropna(subset=["lat", "lon"])
    codes = meta["ad_code"].dropna().astype(str)
    valid = daily["pm25"].notna().sum()
    possible = len(daily)
    return {
        "cities": int(daily["city"].nunique()),
        "cities_with_coordinates": int(located["city"].nunique()),
        "cities_without_coordinates": meta.loc[meta["lat"].isna(), "city"].astype(str).tolist(),
        "prefecture_points": int((codes.str.endswith("00") & ~codes.str.endswith("0000")).sum()),
        "municipality_points": int(codes.str.endswith("0000").sum()),
        "county_points": int((~codes.str.endswith("00")).sum()),
        "date_start": str(daily["date"].min().date()),
        "date_end": str(daily["date"].max().date()),
        "rows": int(len(daily)),
        "pm25_valid_rows": int(valid),
        "pm25_valid_rate": float(valid / possible) if possible else None,
        "files_ok": file_report.get("ok", 0),
        "files_missing": file_report.get("missing", 0),
        "files_error": file_report.get("error", 0),
        "missing_day_sample": file_report.get("missing_days", [])[:20],
        "invalid_hours": file_report.get("invalid_hours", {}),
    }


def level_stats(daily: pd.DataFrame, meta: pd.DataFrame) -> dict:
    valid = daily.dropna(subset=["pm25"]).copy()
    city = valid.groupby("city")["pm25"].agg(mean="mean", n_days="size").reset_index()
    city = city.merge(meta[["city", "region", "province"]], on="city", how="left")
    long_term = city.loc[city["n_days"] >= LONG_TERM_MIN_DAYS].copy()
    regions = []
    for region in REGION_ORDER:
        part = long_term.loc[long_term["region"] == region]
        regions.append({
            "region": region,
            "cities": int(len(part)),
            "mean": None if part.empty else float(part["mean"].mean()),
        })
    ranked = long_term.sort_values("mean", ascending=False)
    top = ranked.head(5)[["city", "mean", "n_days", "region"]]
    bottom = ranked.tail(5).sort_values("mean")[["city", "mean", "n_days", "region"]]

    def _year_mean(year: int) -> dict:
        part = valid.loc[valid["date"].dt.year == year]
        counts = part.groupby("city").size()
        keep = counts[counts >= YEAR_MIN_DAYS].index
        means = part.loc[part["city"].isin(keep)].groupby("city")["pm25"].mean()
        return {"year": year, "cities": int(means.size), "mean": None if means.empty else float(means.mean())}

    y0 = int(valid["date"].dt.year.min())
    y1 = int(valid["date"].dt.year.max())
    early = _year_mean(y0)
    late = _year_mean(y1)
    both_cities = None
    balanced = None
    early_cities = valid.loc[valid["date"].dt.year == y0].groupby("city").size()
    late_cities = valid.loc[valid["date"].dt.year == y1].groupby("city").size()
    shared = early_cities.index.intersection(late_cities.index)
    shared = [city for city in shared if early_cities[city] >= YEAR_MIN_DAYS and late_cities[city] >= YEAR_MIN_DAYS]
    if shared:
        early_mean = valid.loc[(valid["date"].dt.year == y0) & valid["city"].isin(shared)].groupby("city")["pm25"].mean()
        late_mean = valid.loc[(valid["date"].dt.year == y1) & valid["city"].isin(shared)].groupby("city")["pm25"].mean()
        aligned = pd.DataFrame({"early": early_mean, "late": late_mean}).dropna()
        both_cities = int(len(aligned))
        if len(aligned):
            delta = float(aligned["late"].mean() - aligned["early"].mean())
            base = float(aligned["early"].mean())
            balanced = {
                "early_year": y0,
                "late_year": y1,
                "cities": both_cities,
                "early_mean": base,
                "late_mean": float(aligned["late"].mean()),
                "delta": delta,
                "pct": None if base == 0 else delta / base,
                "min_days_each_year": YEAR_MIN_DAYS,
            }
    exceed = {}
    for name, column, limit in (
        ("PM2.5", "pm25", DAY_LIMITS["PM2.5"]),
        ("PM10", "pm10", DAY_LIMITS["PM10"]),
        ("NO2", "no2", DAY_LIMITS["NO2"]),
        ("O3", "o3_mda8", DAY_LIMITS["O3"]),
        ("CO", "co", DAY_LIMITS["CO"]),
        ("SO2", "so2", DAY_LIMITS["SO2"]),
    ):
        series = daily[column].dropna()
        exceed[name] = {
            "n": int(series.size),
            "n_over": int((series > limit).sum()),
            "rate": float((series > limit).mean()) if len(series) else None,
            "limit": limit,
            "mean": float(series.mean()) if len(series) else None,
        }
    return {
        "unweighted_city_day_mean": float(valid["pm25"].mean()) if len(valid) else None,
        "city_count_any_pm25": int(city["city"].nunique()),
        "long_term_min_days": LONG_TERM_MIN_DAYS,
        "long_term_cities": int(len(long_term)),
        "long_term_mean": None if long_term.empty else float(long_term["mean"].mean()),
        "regions": regions,
        "highest": top.to_dict(orient="records"),
        "lowest": bottom.to_dict(orient="records"),
        "early_year": early,
        "late_year": late,
        "balanced": balanced,
        "exceedance": exceed,
    }


def build_summary(daily: pd.DataFrame, meta: pd.DataFrame, file_report: dict, ml: dict) -> tuple[dict, pd.DataFrame]:
    annual = annual_global_moran(daily, meta)
    local = annual_local_moran(daily, meta)
    winter = season_moran(daily, meta, {12, 1, 2})
    summer = season_moran(daily, meta, {6, 7, 8})
    summary = {
        "source": "中国环境监测总站全国城市空气质量实时发布平台，经王晓磊公开归档的城市小时文件汇总",
        "source_url": "https://quotsoft.net/air/",
        "coordinates": "和风天气公开城市列表中的城市级代表点，不是监测站坐标",
        "coordinate_url": "https://github.com/qwd/LocationList",
        "coverage": coverage_stats(daily, meta, file_report),
        "levels": level_stats(daily, meta),
        "moran_annual": annual.to_dict(orient="records"),
        "moran_winter": winter,
        "moran_summer": summer,
        "forecast": ml,
        "weighting": "城市等权，没有按人口加权，因此不是人群暴露",
        "hours_rule": "当天某污染物有效小时少于 18 个，该城市日不进入平均",
        "co_unit": "这套全国文件里的 CO 是毫克/立方米；北京 12 站那套是微克/立方米",
    }
    return summary, local


def _n(value, digits=1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "没有算出"
    return f"{float(value):.{digits}f}"


def render_markdown(summary: dict) -> str:
    cover = summary["coverage"]
    levels = summary["levels"]
    balanced = levels.get("balanced") or {}
    winter = summary["moran_winter"]
    summer = summary["moran_summer"]
    test = summary["forecast"]["splits"].get("test", {})
    lines = [
        "# 全国城市空气质量",
        "",
        f"数据来自{summary['source']}。单日文件地址在 {summary['source_url']} 。",
        f"时间是 {cover['date_start']} 至 {cover['date_end']}。发布口径里的城市名有 {cover['cities']} 个，其中 {cover['cities_with_coordinates']} 个匹配到了城市代表点坐标。",
        f"代表点里，地级代码 {cover['prefecture_points']} 个，直辖市或省级代表点 {cover['municipality_points']} 个，区县代码 {cover['county_points']} 个。这不是全国所有县，也没有把没有发布的乡镇算进来。",
        f"日历中没有汇总成功的日期 {cover['files_missing']} 天，成功日期 {cover['files_ok']} 天。PM2.5 有效城市日占已发布城市日的 {_n(100 * (cover['pm25_valid_rate'] or 0), 1)}%。",
        "",
        "## 浓度",
        "",
        f"全部有效城市日的 PM2.5 等权平均是 {_n(levels['unweighted_city_day_mean'])} μg/m³。这不是人口加权，也不能当成全国人群暴露。",
    ]
    if balanced:
        lines.append(
            f"同一批城市在 {balanced['early_year']} 和 {balanced['late_year']} 都至少有 {balanced['min_days_each_year']} 个有效日的，有 {balanced['cities']} 个。"
            f"它们的城市年平均从 {_n(balanced['early_mean'])} 变为 {_n(balanced['late_mean'])}，相差 {_n(balanced['delta'])} μg/m³（{_n(100 * balanced['pct'], 1)}%）。"
        )
    lines.append("长期平均要求每个城市至少有 " + str(levels["long_term_min_days"]) + f" 个有效日，这样的城市有 {levels['long_term_cities']} 个，城市等权平均 {_n(levels['long_term_mean'])} μg/m³。")
    lines.append("区域同样是城市等权：")
    for region in levels["regions"]:
        lines.append(f"- {region['region']}：{region['cities']} 个城市，平均 {_n(region['mean'])} μg/m³")
    if levels["highest"]:
        high = "、".join(f"{item['city']} {_n(item['mean'])}" for item in levels["highest"][:3])
        low = "、".join(f"{item['city']} {_n(item['mean'])}" for item in levels["lowest"][:3])
        lines.append(f"长期平均最高的几个城市是 {high}。最低的几个是 {low}。")
    exc = levels["exceedance"]["PM2.5"]
    lines.append(
        f"PM2.5 城市日超过 75 μg/m³ 的有 {exc['n_over']} 天，占有效城市日的 {_n(100 * exc['rate'], 1)}%（分母 {exc['n']}）。"
    )
    lines += [
        "",
        "## 空间自相关",
        "",
        f"权重是每个城市的 8 个最近城市，行标准化。一个点要进入某年的计算，当年至少 {YEAR_MIN_DAYS} 个有效日。p 值来自 199 次置换；0.005 是这个次数能给出的下限，表示没有一次打乱比观测更极端。这不是因果检验。",
    ]
    for row in summary["moran_annual"]:
        lines.append(f"- {row['year']}：Moran’s I = {_n(row.get('I'), 3)}，p = {_n(row.get('p'), 3)}，城市数 {row.get('n')}")
    lines.append(
        f"全时段冬季（12、1、2 月，城市至少 {winter.get('min_days')} 个冬季有效日）Moran’s I = {_n(winter.get('I'), 3)}（n={winter.get('n')}，p={_n(winter.get('p'), 3)}）。"
        f"夏季是 {_n(summer.get('I'), 3)}（n={summer.get('n')}，p={_n(summer.get('p'), 3)}）。"
    )
    lines += [
        "",
        "## 次日预测",
        "",
        "模型只使用目标日之前已经知道的浓度、月份和城市代表点。梯度提升、岭回归和「明天等于今天」用同一批样本。",
        f"测试目标日是 {test.get('start')} 至 {test.get('end')}，样本 {test.get('n')}。",
    ]
    if test.get("persistence"):
        lines.append(
            f"测试集平均绝对误差：持续基线 {_n(test['persistence']['mae'], 2)}，岭回归 {_n(test['ridge']['mae'], 2)}，梯度提升 {_n(test['boosting']['mae'], 2)} μg/m³。"
            f"梯度提升相对基线的技能分数是 {_n(test.get('boosting_skill'), 3)}。"
        )
        cls = test.get("class_boosting") or {}
        base = test.get("class_persistence") or {}
        lines.append(
            f"把下一天 PM2.5 是否超过 75 当成分类时，当天是否已经超过 75 的精确率 {_n(base.get('precision'), 3)}、召回 {_n(base.get('recall'), 3)}；"
            f"梯度提升的精确率 {_n(cls.get('precision'), 3)}、召回 {_n(cls.get('recall'), 3)}。正样本比例 {_n(cls.get('positive_rate'), 3)}。"
        )
    lines += [
        "",
        "## 不要说成",
        "",
        "- 不是人口暴露，城市等权会让小城市和大城市一样重。",
        "- 不是每一个县、乡镇都有监测。",
        "- 坐标是城市代表点，空间相关描述的是城市之间，不是街道。",
        "- 相关和可预测都不等于排放因果。",
        "- 没有用 YOLO 检测 AQI。这套数据不是图像，也没有检测框标注。",
        "",
    ]
    return "\n".join(lines) + "\n"
