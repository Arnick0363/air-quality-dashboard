"""把全国城市分析写成 findings 里的一节。数字只来自 summary，不在这里重算。"""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "没有算出"
    return f"{value}{suffix}"


def render_national_section(summary: dict) -> str:
    period = summary["period"]
    cover = summary["coverage"]
    pm = summary["pm25"]
    moran = summary["moran_period_mean"]
    model = summary["model"]
    lines = [
        "## 全国监测城市（日尺度）",
        "",
        "这一节和下面的北京站点不是同一套数据。全国是监测城市的日值，北京是 12 个国控站的小时值。",
        "",
        f"时段：{period['start']} 至 {period['end']}。文件里实际有 {period['n_dates_present']} 天，下载失败或源站缺失 {period['n_missing_dates']} 天。",
        f"小时表里出现的城市 {cover['n_cities_in_files']} 个，对上坐标的 {cover['n_cities_matched']} 个，没对上的 {cover['n_cities_unmatched']} 个。",
        "日均规则：当天至少 18 个有效小时。负值和明显的哨兵值不进入平均。CO 的单位是毫克/立方米，不要和北京小时表里的微克/立方米混用。",
        "臭氧这里是小时浓度的日均，不是日最大 8 小时。AQI 是小时 AQI 的日均，不是生态环境部发布的日 AQI。",
        "",
        "### 可以直接引用的发现",
        "",
    ]
    lines.append(
        f"1. 全部有效城市日的 PM2.5 平均是 {pm['city_day_mean']} μg/m³（{pm['n_city_days']} 个城市日）。"
        f"达到天数门槛的城市，先各自求平均再对城市等权，得到 {pm['city_equal_mean']} μg/m³。"
        "前一个数让数据更全的城市更有分量，后一个数一座城市一票。"
    )
    lines.append(
        f"2. 冬季城市日均 PM2.5 为 {pm['winter_mean']} μg/m³（{pm['winter_n']} 个城市日），"
        f"夏季为 {pm['summer_mean']} μg/m³（{pm['summer_n']} 个城市日），"
        f"冬季是夏季的 {pm['winter_summer_ratio']} 倍。"
        f"城市日 PM2.5 超过 75 μg/m³ 的有 {pm['days_over_75']} 天，占 {None if pm['share_over_75'] is None else round(pm['share_over_75'] * 100, 1)}%（分母 {pm['n_city_days']}）。"
        "75 是 GB 3095-2012 二级的 24 小时浓度，这里拿来做参照，不是正式评价。"
    )
    change = summary.get("year_change")
    if change:
        lines.append(
            f"3. 2023 年城市日均 PM2.5 为 {change['from_mean']} μg/m³，2024 年为 {change['to_mean']} μg/m³，"
            f"相差 {change['delta']} μg/m³。这是城市日等权到每一天，不是人口加权。"
        )
    else:
        lines.append("3. 年份没有同时覆盖 2023 和 2024，这里不写年均变化。")
    highest = pm["highest"][0]
    lowest = pm["lowest"][0]
    lines.append(
        f"4. 有效天数不少于 {cover['min_days_for_city_mean']} 天的城市里，"
        f"PM2.5 平均最高的是{highest['city']}（{highest['province']}，{highest['mean']} μg/m³，{highest['n_days']} 天），"
        f"最低的是{lowest['city']}（{lowest['province']}，{lowest['mean']} μg/m³，{lowest['n_days']} 天）。"
    )
    region_bits = []
    for row in summary["regions"]:
        region_bits.append(
            f"{row['region']} {row['mean']}（冬 {row['winter_mean']} / 夏 {row['summer_mean']}，冬夏比 {row['winter_summer_ratio']}，{row['n_cities']} 城）"
        )
    lines.append(
        "5. 区域是事先划定的，不是聚类分出来的。华北含京津冀晋和内蒙古，长三角含沪苏浙皖，"
        "珠三角只含广州、深圳、佛山、东莞、中山、珠海、惠州、江门、肇庆，成渝含四川和重庆，西北含陕甘青宁新。"
        "各区域城市日 PM2.5 平均：" + "；".join(region_bits) + "。"
    )
    if moran.get("ok"):
        inv = moran.get("inverse_distance_400km") or {}
        inv_text = "反距离权重这次没有形成足够样本。"
        if inv.get("ok"):
            inv_text = (
                f"作为敏感性，400 公里以内的反距离权重得到 I={inv['I']}（n={inv['n']}，p={inv['p_value']}），"
                f"有 {inv['dropped_isolated']} 个城市因为周围 400 公里内没有其他城市而被拿掉。"
            )
        lines.append(
            f"6. 用有效天数不少于 {cover['min_days_for_city_mean']} 天的城市平均 PM2.5 做全局 Moran’s I。"
            f"空间权重是 k=8 的近邻，行标准化，不含自身。n={moran['n']}，I={moran['I']}，"
            f"期望 {moran['expected']}，999 次置换的双侧 p={moran['p_value']}。"
            + (
                "I 高于期望，表示邻近城市的时段平均更相像。"
                if moran["I"] > moran["expected"]
                else "I 不高于期望，这批城市平均没有表现出正的空间聚集。"
            )
            + inv_text
        )
        counts = moran["lisa_counts"]
        lines.append(
            f"7. 同一批城市平均上的 LISA（局部 Moran’s I，α=0.05，置换 999 次，p 值未校正）："
            f"高-高 {counts['高-高']}，低-低 {counts['低-低']}，高-低 {counts['高-低']}，低-高 {counts['低-高']}，"
            f"不显著 {counts['不显著']}。高-高只说明这座城市和它的近邻一起高于这批城市的平均，不是排放源定位。"
        )
    else:
        lines.append(f"6. 全局 Moran’s I 没有做成：{moran.get('reason', '样本不足')}。")
        lines.append("7. 因此也没有 LISA 分类。")
    winter_m = summary.get("moran_winter_mean") or {}
    summer_m = summary.get("moran_summer_mean") or {}
    if winter_m.get("ok") and summer_m.get("ok"):
        lines.append(
            f"8. 冬季城市平均的 Moran’s I 为 {winter_m['I']}（n={winter_m['n']}，p={winter_m['p_value']}），"
            f"夏季为 {summer_m['I']}（n={summer_m['n']}，p={summer_m['p_value']}）。"
            "这是各城市冬、夏均值的截面，不是把所有城市日堆在一起算的假空间相关。"
        )
    lines.append(
        f"9. 预测任务：用昨日 PM2.5、7 日前 PM2.5、邻近城市昨日 PM2.5、月份正余弦、星期和城市代表点坐标，"
        f"预测当日 PM2.5。模型是 {model['model']}。时间切分点 {model['cutoff']}，"
        f"训练 {model['train_start']} 至 {model['train_end']}（{model['n_train']} 行），"
        f"测试 {model['test_start']} 至 {model['test_end']}（{model['n_test']} 行），城市 {model['n_cities']} 个。"
        f"测试集 MAE {model['test_mae']}、RMSE {model['test_rmse']}；"
        f"持久性基线（直接用昨天）MAE {model['persist_mae']}、RMSE {model['persist_rmse']}。"
        f"相对持久性基线，MAE 变化比例为 {model.get('mae_improvement_vs_persist')}（正数表示模型更小）。"
        f"训练集 MAE {model['train_mae']}，用来看是否明显背下了训练段。"
    )
    top_features = "、".join(
        f"{item['label']}（打乱后 MAE 增加 {item['mae_increase']}）" for item in model["importance"][:4]
    )
    lines.append(
        f"10. 置换重要性在测试集抽 {model['importance_sample_n']} 行上计算，重复 5 次。排在前面的是：{top_features}。"
        "重要性是预测有用，不是因果贡献。当天的其他污染物和当天的气象都没有放进特征，全国这套表里也没有气象。"
    )
    lines.append(
        "11. 全国城市没有做 IDW，也没有做克里金。"
        + summary["interpolation"]["reason"]
    )
    if cover["unmatched_cities"]:
        shown = "、".join(cover["unmatched_cities"][:12])
        extra = ""
        if len(cover["unmatched_cities"]) > 12:
            extra = f" 等 {len(cover['unmatched_cities'])} 个"
        lines.append(f"没有对上坐标、因此不进入地图和 Moran 的城市：{shown}{extra}。")
    lines.append("")
    lines.append("### 数据与坐标")
    lines.append("")
    lines.append(f"来源页面：{summary['source']['page']}。单日文件：`china_cities_YYYYMMDD.csv`。")
    lines.append(f"上游：{summary['source']['upstream']}。")
    lines.append(summary["source"]["license_note"])
    lines.append(f"坐标表：{summary['gazetteer']}。")
    lines.append(summary["crs_note"])
    lines.append("区域定义见上面第 5 条。珠三角不是整个广东省。")
    lines.append("")
    lines.append("### 全国这一层的局限")
    lines.append("")
    lines.append("- 城市日是监测网的城市平均，小城市和特大城市各算一个点，没有人口加权。")
    lines.append("- 坐标是城市代表点，不是每座城市里监测站的实测位置。站在城市的哪一侧，这里看不出来。")
    lines.append("- k 近邻可能把海岛和内陆、或者山脉两侧连成邻居。权重表示“名次上的邻近”，不是气团轨迹。")
    lines.append("- LISA 的 p 值没有做多重比较校正，显著城市的个数会偏多。")
    lines.append("- 模型能说明滞后和季节对下一天有没有用，不能说明减排政策的效果。")
    lines.append("")
    return "\n".join(lines)


def upsert_national_findings(findings_path: Path, summary: dict) -> None:
    section = render_national_section(summary).rstrip() + "\n"
    start = "<!-- national:start -->"
    end = "<!-- national:end -->"
    text = findings_path.read_text(encoding="utf-8") if findings_path.exists() else ""
    block = f"{start}\n{section}{end}\n"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        updated = before + block + after.lstrip("\n")
    else:
        updated = block + "\n" + text
    findings_path.write_text(updated, encoding="utf-8")


def dump_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
