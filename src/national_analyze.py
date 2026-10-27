"""从城市日表计算可以写进结论的数字。这里不下载数据。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.national_model import fit_pm25_model, build_supervised
from src.settings import MONTH_TO_SEASON, SEASON_ORDER
from src.spatial_stats import global_moran_test, inverse_distance_weights, knn_weights, local_moran

FOCUS_REGIONS = ["华北", "长三角", "珠三角", "成渝", "西北"]
MIN_DAYS_FOR_MEAN = 200
MIN_SEASON_DAYS = 30
KNN_K = 8
N_PERM = 999
SEED = 42


def _round(value, digits: int):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _city_means(daily: pd.DataFrame, mask: pd.Series | None = None) -> pd.DataFrame:
    part = daily if mask is None else daily.loc[mask]
    part = part.dropna(subset=["PM2.5"])
    return (
        part.groupby("city")
        .agg(pm25_mean=("PM2.5", "mean"), n_days=("PM2.5", "size"))
        .reset_index()
    )


def _moran_block(table: pd.DataFrame, value_col: str = "pm25_mean") -> dict:
    usable = table.dropna(subset=[value_col, "lat", "lon"]).copy()
    if len(usable) < 30:
        return {"ok": False, "n": int(len(usable)), "reason": "有坐标且达到天数门槛的城市少于 30"}
    values = usable[value_col].to_numpy(dtype=float)
    try:
        weights = knn_weights(usable["lat"].to_numpy(), usable["lon"].to_numpy(), k=KNN_K)
        tested = global_moran_test(values, weights, n_perm=N_PERM, seed=SEED)
        local = local_moran(values, weights, n_perm=N_PERM, seed=SEED)
    except ValueError as exc:
        return {"ok": False, "n": int(len(usable)), "reason": str(exc)}
    usable = usable.copy()
    usable["local_i"] = local["local_i"]
    usable["z"] = local["z"]
    usable["spatial_lag"] = local["spatial_lag"]
    usable["p_value"] = local["p_value"]
    usable["cluster"] = local["cluster"]
    counts = {name: int((usable["cluster"] == name).sum()) for name in ["高-高", "低-低", "高-低", "低-高", "不显著"]}
    inv_w, inv_idx = inverse_distance_weights(usable["lat"].to_numpy(), usable["lon"].to_numpy(), cutoff_km=400)
    inverse = {"ok": False, "cutoff_km": 400, "n": int(len(inv_idx))}
    if len(inv_idx) >= 30:
        inv_test = global_moran_test(values[inv_idx], inv_w, n_perm=N_PERM, seed=SEED)
        inverse = {
            "ok": True,
            "cutoff_km": 400,
            "n": int(len(inv_idx)),
            "I": _round(inv_test["I"], 3),
            "expected": _round(inv_test["expected"], 4),
            "p_value": _round(inv_test["p_value"], 3),
            "dropped_isolated": int(len(usable) - len(inv_idx)),
        }
    return {
        "ok": True,
        "n": int(tested["n"]),
        "k": KNN_K,
        "weight": "k 近邻，k=8，行标准化，不含自身",
        "I": _round(tested["I"], 3),
        "expected": _round(tested["expected"], 4),
        "p_value": _round(tested["p_value"], 3),
        "n_permutations": N_PERM,
        "seed": SEED,
        "lisa_counts": counts,
        "lisa_alpha": 0.05,
        "lisa_note": "局部 p 值未做多重比较校正",
        "inverse_distance_400km": inverse,
        "table": usable,
    }


def analyze_national(daily: pd.DataFrame, cities: pd.DataFrame, missing_dates: list[str] | None = None) -> dict:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    meta = cities.copy()
    merged = frame.merge(
        meta[["city", "province", "region", "lon", "lat", "matched", "adcode", "source_name"]],
        on="city",
        how="left",
    )
    merged["season"] = merged["date"].dt.month.map(MONTH_TO_SEASON)
    pm = merged.dropna(subset=["PM2.5"])
    city_mean = _city_means(merged)
    located = meta.loc[meta["matched"], ["city", "province", "region", "lon", "lat"]].merge(city_mean, on="city")
    stable = located.loc[located["n_days"] >= MIN_DAYS_FOR_MEAN].copy()
    moran = _moran_block(stable)

    winter_means = _city_means(merged, merged["season"] == "冬")
    summer_means = _city_means(merged, merged["season"] == "夏")
    seasonal_cities = located[["city", "province", "region", "lon", "lat"]].merge(
        winter_means.rename(columns={"pm25_mean": "winter_mean", "n_days": "winter_days"}), on="city"
    ).merge(
        summer_means.rename(columns={"pm25_mean": "summer_mean", "n_days": "summer_days"}), on="city"
    )
    seasonal_cities = seasonal_cities.loc[
        (seasonal_cities["winter_days"] >= MIN_SEASON_DAYS) & (seasonal_cities["summer_days"] >= MIN_SEASON_DAYS)
    ].copy()
    winter_moran = _moran_block(seasonal_cities.rename(columns={"winter_mean": "pm25_mean"}))
    summer_moran = _moran_block(seasonal_cities.rename(columns={"summer_mean": "pm25_mean"}))

    def _plain(block: dict) -> dict:
        return {key: value for key, value in block.items() if key != "table"}

    region_rows = []
    focus = pm.loc[pm["region"].isin(FOCUS_REGIONS)]
    for region in FOCUS_REGIONS:
        for season in SEASON_ORDER:
            part = focus.loc[(focus["region"] == region) & (focus["season"] == season), "PM2.5"]
            region_rows.append({
                "region": region,
                "season": season,
                "mean": _round(part.mean(), 1) if len(part) else None,
                "n_city_days": int(len(part)),
                "n_cities": int(focus.loc[(focus["region"] == region) & (focus["season"] == season), "city"].nunique()) if len(part) else 0,
            })
    region_overall = []
    for region in FOCUS_REGIONS:
        part = focus.loc[focus["region"] == region, "PM2.5"]
        winter = focus.loc[(focus["region"] == region) & (focus["season"] == "冬"), "PM2.5"]
        summer = focus.loc[(focus["region"] == region) & (focus["season"] == "夏"), "PM2.5"]
        region_overall.append({
            "region": region,
            "mean": _round(part.mean(), 1) if len(part) else None,
            "winter_mean": _round(winter.mean(), 1) if len(winter) else None,
            "summer_mean": _round(summer.mean(), 1) if len(summer) else None,
            "winter_summer_ratio": _round(winter.mean() / summer.mean(), 2) if len(winter) and len(summer) and summer.mean() else None,
            "n_city_days": int(len(part)),
            "n_cities": int(focus.loc[focus["region"] == region, "city"].nunique()),
        })

    ranked = stable.sort_values("pm25_mean", ascending=False)
    def _rank_rows(part: pd.DataFrame) -> list[dict]:
        rows = []
        for row in part.itertuples(index=False):
            rows.append({
                "city": row.city,
                "province": row.province,
                "region": row.region,
                "mean": _round(row.pm25_mean, 1),
                "n_days": int(row.n_days),
            })
        return rows

    years = {}
    year_raw = {}
    for year, part in pm.groupby(pm["date"].dt.year):
        city_year = part.groupby("city")["PM2.5"].mean()
        year_raw[int(year)] = float(part["PM2.5"].mean())
        years[str(int(year))] = {
            "city_day_mean": _round(part["PM2.5"].mean(), 1),
            "city_equal_mean": _round(city_year.mean(), 1),
            "n_city_days": int(len(part)),
            "n_cities": int(part["city"].nunique()),
        }
    year_change = None
    if 2023 in year_raw and 2024 in year_raw:
        year_change = {
            "from_year": 2023,
            "to_year": 2024,
            "from_mean": _round(year_raw[2023], 1),
            "to_mean": _round(year_raw[2024], 1),
            "delta": _round(year_raw[2024] - year_raw[2023], 1),
        }
    lisa_by_region = []
    if moran.get("ok"):
        for region, part in moran["table"].groupby("region"):
            lisa_by_region.append({
                "region": str(region),
                "n": int(len(part)),
                "hh": int((part["cluster"] == "高-高").sum()),
                "ll": int((part["cluster"] == "低-低").sum()),
                "hl": int((part["cluster"] == "高-低").sum()),
                "lh": int((part["cluster"] == "低-高").sum()),
            })

    winter_all = pm.loc[pm["season"] == "冬", "PM2.5"]
    summer_all = pm.loc[pm["season"] == "夏", "PM2.5"]
    model_input = merged.loc[merged["matched"] == True, ["date", "city", "PM2.5", "lat", "lon"]]  # noqa: E712
    fitted = fit_pm25_model(build_supervised(model_input, k=KNN_K))
    model_summary = fitted["summary"]
    for key in ("train_mae", "train_rmse", "test_mae", "test_rmse", "persist_mae", "persist_rmse"):
        model_summary[key] = _round(model_summary[key], 2)
    for item in model_summary["importance"]:
        item["mae_increase"] = _round(item["mae_increase"], 3)
        item["mae_increase_std"] = _round(item["mae_increase_std"], 3)
    if model_summary["persist_mae"]:
        model_summary["mae_improvement_vs_persist"] = _round(
            (model_summary["persist_mae"] - model_summary["test_mae"]) / model_summary["persist_mae"],
            3,
        )

    over_75 = int((pm["PM2.5"] > 75).sum())
    summary = {
        "source": {
            "name": "全国城市空气质量小时数据",
            "page": "https://quotsoft.net/air/",
            "url_template": "https://quotsoft.net/air/data/china_cities_YYYYMMDD.csv",
            "upstream": "中国环境监测总站全国城市空气质量实时发布平台",
            "license_note": "发布页写明欢迎分析研究，没有 SPDX 许可证。本仓库不收录原始逐小时文件。",
        },
        "crs_note": "城市代表点来自和风天气公开城市列表的 Latitude/Longitude。列表页没有单列坐标系名称。北京点（约 39.905°N, 116.405°E）与常见 WGS84 城市中心在数百米内，分析按 WGS84 城市代表点使用，不是监测站坐标。",
        "gazetteer": "https://github.com/qwd/LocationList/blob/master/China-City-List-latest.csv",
        "period": {
            "start": pd.Timestamp(frame["date"].min()).date().isoformat(),
            "end": pd.Timestamp(frame["date"].max()).date().isoformat(),
            "n_dates_present": int(frame["date"].nunique()),
            "n_missing_dates": int(len(missing_dates or [])),
            "missing_dates": list(missing_dates or []),
        },
        "coverage": {
            "n_cities_in_files": int(frame["city"].nunique()),
            "n_cities_matched": int(meta["matched"].sum()),
            "n_cities_unmatched": int((~meta["matched"].astype(bool)).sum()),
            "unmatched_cities": meta.loc[~meta["matched"].astype(bool), "city"].astype(str).tolist(),
            "n_cities_in_moran": int(moran.get("n", 0)),
            "min_days_for_city_mean": MIN_DAYS_FOR_MEAN,
            "daily_rule": "当天该指标至少 18 个有效小时才保留日均",
            "sentinel_rule": "负值，以及 PM2.5>3000、CO>50 mg/m³、AQI>2000 等哨兵上限，不进入日均",
        },
        "pm25": {
            "city_day_mean": _round(pm["PM2.5"].mean(), 1),
            "city_equal_mean": _round(stable["pm25_mean"].mean(), 1) if len(stable) else None,
            "n_city_days": int(len(pm)),
            "winter_mean": _round(winter_all.mean(), 1),
            "summer_mean": _round(summer_all.mean(), 1),
            "winter_n": int(len(winter_all)),
            "summer_n": int(len(summer_all)),
            "winter_summer_ratio": _round(winter_all.mean() / summer_all.mean(), 2) if len(summer_all) and summer_all.mean() else None,
            "days_over_75": over_75,
            "share_over_75": _round(over_75 / len(pm), 3) if len(pm) else None,
            "highest": _rank_rows(ranked.head(8)),
            "lowest": _rank_rows(ranked.tail(8).sort_values("pm25_mean")),
        },
        "years": years,
        "year_change": year_change,
        "lisa_by_region": lisa_by_region,
        "regions": region_overall,
        "region_seasons": region_rows,
        "moran_period_mean": _plain(moran),
        "moran_winter_mean": _plain(winter_moran),
        "moran_summer_mean": _plain(summer_moran),
        "interpolation": {
            "national_idw": False,
            "national_kriging": False,
            "reason": "全国点是城市代表点，间距经常超过 100 公里。样本量够拟合一个变异函数，但数据支撑是整座城市，插成连续面会被看成街道尺度的污染场。北京 12 站在约 50 公里内，IDW 只保留距最近站 40 公里以内的格点。",
        },
        "model": model_summary,
        "scale_note": "全国结果是监测城市的日尺度；北京案例是 12 个国控站的小时尺度。两套数字不能直接相减。",
    }
    lisa = moran["table"].copy() if moran.get("ok") else pd.DataFrame()
    return {"summary": summary, "lisa": lisa, "predictions": fitted["predictions"]}
