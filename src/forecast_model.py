"""用当天及更早的信息预测下一天的城市 PM2.5。

切分看的是目标日：训练目标日落在 2023 及以前，2024 年只作验证，2025 年才是测试。
超参数事先写死，不用验证年或测试年做搜索。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.china_settings import PM25_DAY_LIMIT, RANDOM_SEED, TEST_YEAR, VALID_YEAR

FEATURE_COLUMNS = [
    "pm25",
    "pm25_lag1",
    "pm25_lag7",
    "pm25_roll7",
    "pm10",
    "so2",
    "no2",
    "co",
    "o3_mda8",
    "month_sin",
    "month_cos",
    "dow",
    "lat",
    "lon",
]


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "n": int(len(y_true)),
    }


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    truth = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred).astype(bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "precision": precision,
        "recall": recall,
        "positive_rate": float(truth.mean()) if len(truth) else None,
        "predicted_positive_rate": float(pred.mean()) if len(pred) else None,
        "n": int(len(truth)),
    }


def make_supervised(daily: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """先补全每个城市的日历，再取滞后。缺的那一天不会被当成「昨天」。"""
    columns = ["date", "city", "pm25", "pm10", "so2", "no2", "co", "o3_mda8"]
    frame = daily[columns].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates(["city", "date"])
    dates = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    index = pd.MultiIndex.from_product([frame["city"].unique(), dates], names=["city", "date"])
    frame = frame.set_index(["city", "date"]).reindex(index).reset_index()
    frame = frame.sort_values(["city", "date"])
    grouped = frame.groupby("city", sort=False)
    frame["pm25_next"] = grouped["pm25"].shift(-1)
    frame["pm25_lag1"] = grouped["pm25"].shift(1)
    frame["pm25_lag7"] = grouped["pm25"].shift(7)
    frame["pm25_roll7"] = grouped["pm25"].transform(lambda series: series.rolling(7, min_periods=5).mean())
    frame["target_date"] = pd.to_datetime(frame["date"]) + pd.to_timedelta(1, unit="D")
    frame = frame.dropna(subset=["pm25", "pm25_next", "pm25_lag1", "pm25_lag7", "pm25_roll7"])
    frame = frame.merge(meta[["city", "lat", "lon", "region"]], on="city", how="inner")
    frame = frame.dropna(subset=["lat", "lon"])
    target_year = frame["target_date"].dt.year
    frame["split"] = np.where(
        target_year < VALID_YEAR,
        "train",
        np.where(target_year == VALID_YEAR, "valid", np.where(target_year == TEST_YEAR, "test", "drop")),
    )
    month = frame["date"].dt.month.to_numpy()
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)
    frame["dow"] = frame["date"].dt.dayofweek.to_numpy()
    return frame


def impute_with_train_median(train: pd.DataFrame, others: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.Series]:
    median = train[FEATURE_COLUMNS].median(numeric_only=True)
    filled_train = train.copy()
    filled_train[FEATURE_COLUMNS] = train[FEATURE_COLUMNS].fillna(median)
    filled_others = []
    for frame in others:
        copied = frame.copy()
        copied[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].fillna(median)
        filled_others.append(copied)
    return filled_train, filled_others, median


def _fit_models(train: pd.DataFrame):
    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["pm25_next"].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_scaled, y_train)
    boosting = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=40,
        random_state=RANDOM_SEED,
    )
    boosting.fit(x_train, y_train)
    classifier = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=40,
        random_state=RANDOM_SEED,
    )
    classifier.fit(x_train, (y_train >= PM25_DAY_LIMIT).astype(int))
    return scaler, ridge, boosting, classifier


def _predict(frame: pd.DataFrame, scaler, ridge, boosting, classifier) -> dict[str, np.ndarray]:
    x_values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    return {
        "persistence": frame["pm25"].to_numpy(dtype=float),
        "ridge": ridge.predict(scaler.transform(x_values)),
        "boosting": boosting.predict(x_values),
        "class_persist": (frame["pm25"].to_numpy(dtype=float) >= PM25_DAY_LIMIT),
        "class_model": classifier.predict(x_values).astype(bool),
    }


def evaluate_forecast(daily: pd.DataFrame, meta: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    supervised = make_supervised(daily, meta)
    train = supervised.loc[supervised["split"] == "train"].copy()
    valid = supervised.loc[supervised["split"] == "valid"].copy()
    test = supervised.loc[supervised["split"] == "test"].copy()
    if train.empty or test.empty:
        raise RuntimeError("训练或测试样本为空，无法做时间切分评估")
    train, (valid, test), _median = impute_with_train_median(train, [valid, test])
    scaler, ridge, boosting, classifier = _fit_models(train)

    report: dict = {
        "features": FEATURE_COLUMNS,
        "rule": "特征只用目标日的前一天及更早；其他污染物的缺失值只用训练集中位数填补",
        "hyperparameters": "事先固定，没有用 2024 或 2025 做搜索",
        "pm25_limit": PM25_DAY_LIMIT,
        "splits": {},
    }
    prediction_frame = None
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        if part.empty:
            report["splits"][name] = {"n": 0}
            continue
        predicted = _predict(part, scaler, ridge, boosting, classifier)
        y_true = part["pm25_next"].to_numpy(dtype=float)
        entry = {
            "n": int(len(part)),
            "start": str(part["target_date"].min().date()),
            "end": str(part["target_date"].max().date()),
            "persistence": regression_scores(y_true, predicted["persistence"]),
            "ridge": regression_scores(y_true, predicted["ridge"]),
            "boosting": regression_scores(y_true, predicted["boosting"]),
            "class_persistence": classification_scores(y_true >= PM25_DAY_LIMIT, predicted["class_persist"]),
            "class_boosting": classification_scores(y_true >= PM25_DAY_LIMIT, predicted["class_model"]),
        }
        persist_mae = entry["persistence"]["mae"]
        entry["boosting_skill"] = None if persist_mae == 0 else 1 - entry["boosting"]["mae"] / persist_mae
        entry["ridge_skill"] = None if persist_mae == 0 else 1 - entry["ridge"]["mae"] / persist_mae
        report["splits"][name] = entry
        if name == "test":
            by_region = []
            temp = part.copy()
            temp["abs_model"] = np.abs(predicted["boosting"] - y_true)
            temp["abs_persist"] = np.abs(predicted["persistence"] - y_true)
            for region, group in temp.groupby("region"):
                by_region.append({
                    "region": region,
                    "n": int(len(group)),
                    "boosting_mae": float(group["abs_model"].mean()),
                    "persistence_mae": float(group["abs_persist"].mean()),
                })
            report["test_by_region"] = by_region
            prediction_frame = pd.DataFrame({
                "city": part["city"].to_numpy(),
                "date": part["target_date"].to_numpy(),
                "region": part["region"].to_numpy(),
                "observed": y_true,
                "persistence": predicted["persistence"],
                "ridge": predicted["ridge"],
                "boosting": predicted["boosting"],
            })
    if prediction_frame is None:
        prediction_frame = pd.DataFrame()
    return report, prediction_frame
