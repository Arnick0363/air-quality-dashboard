"""生成面试问答 PDF。数字只从已经算好的 JSON 里取。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src.paths import CHINA_SUMMARY_JSON, INTERVIEW_PDF, NATIONAL_SUMMARY_JSON, SUMMARY_JSON, YOLO_DIR

INK = HexColor("#1C2830")
GREEN = HexColor("#0B6E4F")
MUTED = HexColor("#5C6B73")
PAPER = HexColor("#F7F5F0")
LINE = HexColor("#E4E0D6")


def _load() -> tuple[dict, dict, dict | None, dict | None]:
    if not NATIONAL_SUMMARY_JSON.exists():
        raise SystemExit("还没有 outputs/national_summary.json，先运行 python scripts/prepare_national.py")
    national = json.loads(NATIONAL_SUMMARY_JSON.read_text(encoding="utf-8"))
    beijing = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    yolo_path = YOLO_DIR / "summary.json"
    fail_path = YOLO_DIR / "failure.json"
    yolo = json.loads(yolo_path.read_text(encoding="utf-8")) if yolo_path.exists() else None
    failure = json.loads(fail_path.read_text(encoding="utf-8")) if fail_path.exists() and yolo is None else None
    return national, beijing, yolo, failure


def _styles() -> dict[str, ParagraphStyle]:
    regular = Path(r"C:\Windows\Fonts\msyh.ttc")
    bold = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    if not regular.exists():
        raise SystemExit("找不到微软雅黑：C:\\Windows\\Fonts\\msyh.ttc")
    pdfmetrics.registerFont(TTFont("YaHei", str(regular), subfontIndex=0))
    if bold.exists():
        pdfmetrics.registerFont(TTFont("YaHeiBd", str(bold), subfontIndex=0))
        bold_name = "YaHeiBd"
    else:
        bold_name = "YaHei"
    registerFontFamily("YaHei", normal="YaHei", bold=bold_name, italic="YaHei", boldItalic=bold_name)
    return {
        "cover": ParagraphStyle("cover", fontName=bold_name, fontSize=22, leading=30, textColor=INK, alignment=TA_LEFT),
        "kicker": ParagraphStyle("kicker", fontName=bold_name, fontSize=9, leading=14, textColor=GREEN),
        "h1": ParagraphStyle("h1", fontName=bold_name, fontSize=14, leading=20, textColor=INK, spaceBefore=8, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName=bold_name, fontSize=12, leading=17, textColor=GREEN, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", fontName="YaHei", fontSize=10.5, leading=17, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6),
        "small": ParagraphStyle("small", fontName="YaHei", fontSize=9, leading=13, textColor=MUTED, spaceAfter=4),
        "toc": ParagraphStyle("toc", fontName="YaHei", fontSize=11, leading=18, textColor=INK),
        "q": ParagraphStyle("q", fontName=bold_name, fontSize=11, leading=16, textColor=INK, spaceBefore=2, spaceAfter=3),
        "cell": ParagraphStyle("cell", fontName="YaHei", fontSize=8.5, leading=12, textColor=INK),
        "cellh": ParagraphStyle("cellh", fontName=bold_name, fontSize=8.5, leading=12, textColor=white),
    }


class Mark(Paragraph):
    def __init__(self, text: str, style, title: str):
        super().__init__(text, style)
        self.toc_title = title


def _p(text: str, style) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def _draw_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(GREEN)
    canvas.rect(0, A4[1] - 8 * mm, A4[0], 8 * mm, fill=1, stroke=0)
    canvas.setFillColor(INK)
    canvas.setFont("YaHei", 8)
    canvas.drawString(18 * mm, 10 * mm, "空间大数据分析 · 面试备忘 · 只写本项目算过的结果")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def _region_sentence(national: dict) -> str:
    bits = []
    for row in national["regions"]:
        bits.append(f"{row['region']} {row['mean']}")
    return "、".join(bits)


def _yolo_paragraph(yolo: dict | None, failure: dict | None) -> str:
    reason = (failure or {}).get("reason") or "没有拿到许可清楚、且带标注的公开图像。"
    if yolo and yolo.get("images"):
        return (
            "虽然本地曾经尝试过推理，仪表盘最终没有保留视觉页。"
            "原因是没有一套许可清楚、并且自带标注的公开图像可以和监测网分开交付。"
            "下面不引用任何检测框、mAP 或训练曲线。"
        )
    return (
        "仪表盘里没有 YOLO 页面。Wikimedia 图片接口在本机下载失败（SSL 中断），手头也没有另一套许可清楚、并且自带框标注的道路图像。"
        "所以不展示检测框，不写 mAP，也不画训练曲线。YOLO11 仍然是检测模型：输入是图像，输出是框和类别，不是 μg/m³。"
    )


def _questions(national: dict, beijing: dict, yolo_text: str) -> list[tuple[str, str, str]]:
    pm = national["pm25"]
    moran = national["moran_period_mean"]
    model = national["model"]
    bj = beijing["pm25"]
    loocv = beijing["loocv"]
    top = model["importance"][0]
    direction = "高于" if moran.get("I", 0) > moran.get("expected", 0) else "不高于"
    return [
        (
            "为什么全国城市和北京站点要分开？",
            f"因为它们回答的不是同一个空间问题。全国是 {national['coverage']['n_cities_in_files']} 个监测城市的日值，一个点代表一座城市；北京是 12 个站的小时值，点与点只隔几公里到几十公里。我把全国城市日均 PM2.5 {pm['city_day_mean']} 和北京城市日均 {bj['mean']} 放在一起讲，只是说明两个尺度都看过，不会把它们相减，说成北京比全国高或低多少。",
            "尺度一变，方差和相关都会变。城市平均会把区内的高低站抹平，所以北京内部东四和定陵的差别，在全国那张图上根本看不见。这就是可修面积元问题：分区方式变了，统计结果可以变。若拿城市平均去推断某个街区，就是生态谬误。",
        ),
        (
            "Moran’s I 在你这里到底是什么意思？",
            f"它看的是：一座城市的 PM2.5 时段平均，和它空间上的邻居像不像。我用 k=8 的近邻，行标准化，不含自己。n={moran.get('n')}，I={moran.get('I')}，期望 {moran.get('expected')}，999 次置换的双侧 p={moran.get('p_value')}。I {direction}期望。期望大约是 -1/(n-1)，不是 0，因为权重矩阵有约束。",
            "p 值来自把城市的数值随机重排，权重不动。这样不假设浓度服从正态分布。它检验的是‘没有空间聚集’这个零假设，不是某一座城市的成因。置换次数有限，p 最小也就是 1/1000 这个量级，我不会把 p 说成 0。",
        ),
        (
            "为什么权重用 k 近邻，不用反距离？",
            "东部城市密、西部城市稀。若用一个固定距离，东部会连进很多邻居，西部可能一个邻居都没有。k 近邻保证每个城市都有 8 个邻居，比较好解释。代价是：海南、河西走廊这类地方，第 8 个邻居可能已经很远，权重表示的是名次上的邻近，不是气团能不能过去。",
            f"所以我又做了 400 公里以内的反距离作为敏感性，结果写在 findings 里：I={ (moran.get('inverse_distance_400km') or {}).get('I') }，n={ (moran.get('inverse_distance_400km') or {}).get('n') }，p={ (moran.get('inverse_distance_400km') or {}).get('p_value') }。两种权重如果方向一致，我才敢说聚集不是某一种权重造出来的。",
        ),
        (
            "LISA 的高-高是不是找到了污染源？",
            f"不是。高-高只表示：这座城市高于这批城市的平均，它的近邻也高于平均，而且局部统计在未校正的 α=0.05 下显著。全样本里高-高 {moran.get('lisa_counts', {}).get('高-高')} 个，低-低 {moran.get('lisa_counts', {}).get('低-低')} 个。低-低是一起偏低，高-低和低-高是和邻居反着来。",
            "我没有做多重比较校正。城市有几百个，即使没有真实聚集，也会有一些点的 p 值小于 0.05。所以个数只能当描述，不能当‘全国有多少个源’。另外用的是时段平均，一次短污染过程不会在这张图上单独出现。",
        ),
        (
            "IDW 和克里金为什么只在一个尺度上做？",
            f"北京 12 个站分布在大约几十公里里，我用幂次 2 的逆距离加权，并且距最近站超过 40 公里的格点不着色。留一法里 IDW 的 RMSE 是 {loocv['idw_rmse']}，最近邻是 {loocv['nn_rmse']}。这只说明长期平均在站点之间比较平滑，不是小时污染地图。",
            "全国城市点经常相距一百公里以上。样本量其实够拟合一个变异函数，但每个点的支撑是整座城市的平均，不是一个可移动的点浓度。把这样的点克里金成连续面，面试时会被看成我把城市值说成了街道场。所以全国不做克里金，也不做 IDW 色斑。",
        ),
        (
            "模型任务是什么，为什么要时间切分？",
            f"任务是预测当日城市 PM2.5。特征只有昨日、7 日前、邻近城市昨日、月份正余弦、星期和代表点坐标。切分点是 {model['cutoff']}，测试 MAE {model['test_mae']}，RMSE {model['test_rmse']}；直接用昨天的 MAE {model['persist_mae']}，RMSE {model['persist_rmse']}。相对变化比例是 {model.get('mae_improvement_vs_persist')}，正数才表示模型的 MAE 更小。",
            "空气质量前后天相关很强。如果随机打乱再划分，测试集里会混进‘未来的邻居天’，误差会好看，但那不是预测。持久性基线就是承认这种相关：先问昨天够不够用，再问模型多出来的部分值不值。训练集 MAE 是 {train}，如果它远小于测试，我就知道模型把训练段背下来了。".format(train=model["train_mae"]),
        ),
        (
            "会不会泄漏？为什么不用大的深度学习去刷分？",
            f"泄漏指用了预测时还不知道的信息。我没有放当天的 PM10、NO2，也没有放当天的气象。全国这套城市表里本来就没有气象。空间滞后用的是邻居的昨天，不是邻居的今天。重要性最高的是{top['label']}，打乱后 MAE 增加 {top['mae_increase']}。",
            "深度网络可以拟合更弯的关系，但这里的主要信号是滞后和季节，样本量是城市日，不是图像。把模型做大，测试误差未必稳定地好于昨天，面试时也不好解释每一个输入。我用 sklearn 的 HistGradientBoosting，是为了在 CPU 上跑完，并能做置换重要性。它不是因果模型，不能用来声称某项政策降了多少浓度。",
        ),
        (
            "你是不是用 YOLO 测空气质量？",
            "不是。这个仪表盘里没有检测页。YOLO11 吃的是图像，吐的是框和类别，监测城市日值是表格，两边对不上。我没有拿到许可清楚、并且自带标注的公开图像，所以没有跑检测，也没有检测框、mAP 或训练曲线可以报。",
            yolo_text + " 如果面试官追问能不能加一个回归头去预测 PM2.5，我会拒绝这个说法：照片里的车和城市日均浓度之间隔着排放因子、气象、化学转化和完全不同的时间，硬接在一起是在编故事。",
        ),
        (
            "MAUP 和生态谬误，用你自己的图怎么讲？",
            "可修面积元问题说的是：空间单元怎么划，结果就可能怎么变。我的区域是事先写死的，珠三角不是整个广东，华北包含了内蒙古。如果把内蒙古从华北拿掉，华北的平均会变。所以区域柱状图是在这个定义下的对比，不是唯一的中国分区。",
            f"生态谬误是用区域平均去说个体。全国城市日均 {pm['city_day_mean']}，不能推出某个街区的人吸到的就是这个数。北京站点层能看到东四 {bj['highest_value']}、定陵 {bj['lowest_value']}，这已经是站与站的差别；再往室内或路边走，数据里没有。两层放在一起，就是为了把这个边界讲清楚。",
        ),
        (
            "这套作品最不该夸大的地方是什么？",
            f"第一，不是全国每小时全站点。城市日把一天收成一个数，站点内部看不见。第二，坐标是城市代表点，不是每个监测子站的实测位置，对不上的城市有 {national['coverage']['n_cities_unmatched']} 个，地图里没有它们。第三，相关和预测都不是因果。第四，主任务是空间统计和表格回归，不是目标检测。",
            "下一步如果继续做，我会优先补气象和真实的监测站坐标，而不是把模型换成更大的网络。有了站坐标，才谈得上在城市内部做变异函数；有了气象，才可以把风速、边界层放进预测，并且仍然只用当时已经知道的值。政策评估需要干预和对照，这个项目没有。",
        ),
    ]


def build() -> int:
    national, beijing, yolo, failure = _load()
    styles = _styles()
    pm = national["pm25"]
    moran = national["moran_period_mean"]
    model = national["model"]
    cover = national["coverage"]
    period = national["period"]
    bj = beijing["pm25"]
    yolo_text = _yolo_paragraph(yolo, failure)
    questions = _questions(national, beijing, yolo_text)

    def story_body(toc: list[tuple[str, int]] | None) -> list:
        items: list = []
        items.append(Paragraph("面试备忘", styles["kicker"]))
        items.append(Spacer(1, 4 * mm))
        items.append(Paragraph("中国城市空气质量的两个尺度", styles["cover"]))
        items.append(Spacer(1, 3 * mm))
        items.append(_p(
            "给澳门科技大学空间大数据分析学硕士面试用。下面每句数字都能在 docs/findings.md 和 outputs 的 JSON 里对上。没跑成功的部分会写失败原因，不补数。",
            styles["body"],
        ))
        items.append(_p(
            f"全国：{period['start']} 至 {period['end']}，监测城市日值。北京站点：2013-03 至 2017-02，12 个国控站的小时值。",
            styles["small"],
        ))
        items.append(Spacer(1, 6 * mm))
        items.append(Paragraph("目录", styles["h1"]))
        titles = [
            "三十秒可以怎么讲",
            "为什么是两个尺度",
            "数据从哪来，坐标系是什么",
            "Moran’s I 和 LISA 怎么读",
            "插值为什么只做在北京站点",
            "预测模型：切分、基线和泄漏",
            "YOLO11 做了什么，没做什么",
            "可能被追问的十个问题",
            "中英名词对照",
            "局限和下一步",
        ]
        if not toc:
            for index, title in enumerate(titles, start=1):
                items.append(Paragraph(f"{index}. {escape(title)}", styles["toc"]))
        else:
            for index, (title, page) in enumerate(toc, start=1):
                items.append(Paragraph(f"{index}. {escape(title)}　　第 {page} 页", styles["toc"]))
        items.append(PageBreak())

        sections: list[tuple[str, list]] = []

        block = []
        block.append(_p(
            f"我想看两件事：中国这些监测城市的 PM2.5 在空间上是不是成片的，以及只凭昨天和更早的信息，能不能比‘直接报昨天’更接近今天。"
            f"全国数据是 {period['start']} 到 {period['end']} 的城市日值，实际落到文件里 {period['n_dates_present']} 天，源站或下载缺失 {period['n_missing_dates']} 天。"
            f"城市名 {cover['n_cities_in_files']} 个，对上代表点坐标的 {cover['n_cities_matched']} 个。"
            f"全部有效城市日的 PM2.5 平均是 {pm['city_day_mean']} μg/m³，冬季 {pm['winter_mean']}，夏季 {pm['summer_mean']}。",
            styles["body"],
        ))
        block.append(_p(
            f"空间上我用 k=8 近邻做了 Moran’s I，I={moran.get('I')}，期望 {moran.get('expected')}，置换 p={moran.get('p_value')}。"
            f"预测上，时间切分之后梯度提升树的测试 MAE 是 {model['test_mae']}，持久性基线是 {model['persist_mae']}。"
            f"北京那一层是另一套小时数据，城市日均 PM2.5 {bj['mean']}，用来看站与站的差别，不和全国数字相减。"
            f"监测表不是图像，仪表盘没有 YOLO 页面，也不报告检测框或 mAP。",
            styles["body"],
        ))
        sections.append(("三十秒可以怎么讲", block))

        block = []
        block.append(_p(
            "全国层回答的是城市与城市之间。一个点是一座监测城市一天的平均，小城市和特大城市在这个平均里各算一个点，没有人口加权。"
            "北京层回答的是一座城市内部。12 个站有的在城区，有的在定陵、怀柔这种更靠外的点，小时数据能看到日内起伏和一次重污染过程。",
            styles["body"],
        ))
        block.append(_p(
            f"北京全周期城市日均 {bj['mean']} μg/m³，冬 {bj['winter_mean']}，夏 {bj['summer_mean']}，冬夏比 {bj['winter_summer_ratio']}。"
            f"站点日均最高是{bj['highest_station']} {bj['highest_value']}，最低是{bj['lowest_station']} {bj['lowest_value']}。"
            "这些句子只属于 2013 到 2017 的北京站点。全国的冬夏比是另一批年份、另一种空间单元，方向可以都是冬天更高，数值不能互相替代。",
            styles["body"],
        ))
        block.append(_p(
            "可修面积元问题在这里很具体：我把珠三角定义成九个城市，而不是整个广东省。换一套边界，柱状图会变。"
            "生态谬误则是：不能用城市日均去说某一个路口。仪表盘把两层做成两组页面，就是为了让这个差别先被看见。",
            styles["body"],
        ))
        sections.append(("为什么是两个尺度", block))

        block = []
        block.append(_p(
            "全国小时宽表来自 quotsoft.net/air 按日期公开的 china_cities_YYYYMMDD.csv，上游是中国环境监测总站的城市实时发布。"
            "页面写了欢迎分析研究，没有 SPDX 许可证。原始逐小时文件不放进仓库，脚本下载后立刻收成日值。",
            styles["body"],
        ))
        block.append(_p(
            f"清洗只做了几件能说清楚的事。负值，以及 PM2.5 超过 3000、CO 超过 50 mg/m³、AQI 超过 2000 这类哨兵上限，不进入平均。"
            f"当天有效小时少于 18 的指标留空。CO 在这套表里是毫克/立方米，北京小时表里的 CO 是微克/立方米。"
            f"臭氧是小时值的日均，不是日最大 8 小时。AQI 是小时 AQI 的日均，不是正式的日 AQI。"
            f"缺失日期 {period['n_missing_dates']} 天"
            + (f"（{'、'.join(period.get('missing_dates') or [])}）" if period.get("missing_dates") else "")
            + "，不插成假的全国日。"
            + (
                f"有效天数不少于 200 天的城市里，PM2.5 平均最高是{pm['highest'][0]['city']} {pm['highest'][0]['mean']}，最低是{pm['lowest'][0]['city']} {pm['lowest'][0]['mean']}。"
                if pm.get("highest") and pm.get("lowest")
                else ""
            ),
            styles["body"],
        ))
        block.append(_p(national["crs_note"], styles["body"]))
        unmatched = cover.get("unmatched_cities") or []
        if unmatched:
            block.append(_p("没有对上坐标、因此不进地图和 Moran 的城市：" + "、".join(unmatched) + "。", styles["body"]))
        block.append(_p(
            "北京数据是 UCI 的 Beijing Multi-Site Air-Quality Data，2013-03-01 到 2017-02-28，12 站小时值。"
            "站点坐标来自公开文献里的国控点位置，保留三位小数，也不是这次实测。短缺口最多线性补 6 小时，降水和风向不补。",
            styles["body"],
        ))
        sections.append(("数据从哪来，坐标系是什么", block))

        block = []
        block.append(_p(
            f"进入 Moran 的是有效天数不少于 {cover['min_days_for_city_mean']} 天的城市平均，n={moran.get('n')}。"
            "先把每个城市收成一个数，再看这些数在空间上的排列。这样避免把同一座城市的相邻日期当成独立的空间样本。",
            styles["body"],
        ))
        block.append(_p(
            f"I={moran.get('I')}，期望 {moran.get('expected')}，p={moran.get('p_value')}。权重：{moran.get('weight')}。"
            "读的时候先看 I 和期望的方向，再看置换检验有没有把这个方向留在很小的 p 值上。",
            styles["body"],
        ))
        inv = moran.get("inverse_distance_400km") or {}
        if inv.get("ok"):
            block.append(_p(
                f"反距离、400 公里截断的敏感性：I={inv['I']}，n={inv['n']}，p={inv['p_value']}，"
                f"因为周围没有邻居而被去掉的城市 {inv['dropped_isolated']} 个。",
                styles["body"],
            ))
        counts = moran.get("lisa_counts") or {}
        block.append(_p(
            f"LISA 在同一批城市平均上计算，α=0.05，置换同样 999 次，p 值没有校正。"
            f"高-高 {counts.get('高-高')}，低-低 {counts.get('低-低')}，高-低 {counts.get('高-低')}，低-高 {counts.get('低-高')}，不显著 {counts.get('不显著')}。"
            "高-高不是源，低-低不是清洁证明，它们都是相对这批城市平均而言的。",
            styles["body"],
        ))
        winter = national.get("moran_winter_mean") or {}
        summer = national.get("moran_summer_mean") or {}
        if winter.get("ok") and summer.get("ok"):
            block.append(_p(
                f"把各城市的冬季平均和夏季平均分开再做一次：冬 I={winter['I']}（n={winter['n']}，p={winter['p_value']}），"
                f"夏 I={summer['I']}（n={summer['n']}，p={summer['p_value']}）。这仍然是截面，不是日序列的空间相关。",
                styles["body"],
            ))
        block.append(_p(
            "区域对比是定义好的五块：" + _region_sentence(national) + "。单位是 μg/m³，统计是城市日平均。"
            "珠三角只含广州、深圳、佛山、东莞、中山、珠海、惠州、江门、肇庆，不是整个广东省。冬夏比写在 findings 的表里。",
            styles["body"],
        ))
        sections.append(("Moran’s I 和 LISA 怎么读", block))

        block = []
        block.append(_p(
            f"北京留一法：IDW RMSE {beijing['loocv']['idw_rmse']}，最近邻 RMSE {beijing['loocv']['nn_rmse']}，幂次 {beijing['loocv']['power']}。"
            "IDW 在这里是描述工具。40 公里掩膜的意思是：离开站点太远，我就不涂颜色，避免一张图看起来像整个华北都有了连续场。",
            styles["body"],
        ))
        block.append(_p(national["interpolation"]["reason"], styles["body"]))
        block.append(_p(
            "克里金需要协方差函数，也就是变异函数：距离增加时，两点差异的方差怎么变。城市代表点的距离大多落在‘城市与城市’这个尺度上，拟合出来的变程描述的也是这个尺度。"
            "样本点个数不是唯一的门槛。支撑（support）不对，模型再完整也回答不了街道问题。所以这个项目把‘没做全国克里金’写成方法选择，而不是漏做。",
            styles["body"],
        ))
        sections.append(("插值为什么只做在北京站点", block))

        block = []
        block.append(_p(
            f"模型是 {model['model']}。训练 {model['train_start']} 至 {model['train_end']}，{model['n_train']} 行；"
            f"测试 {model['test_start']} 至 {model['test_end']}，{model['n_test']} 行；城市 {model['n_cities']} 个。",
            styles["body"],
        ))
        block.append(_p(
            f"测试 MAE {model['test_mae']}，RMSE {model['test_rmse']}。持久性基线 MAE {model['persist_mae']}，RMSE {model['persist_rmse']}。"
            f"训练集 MAE {model['train_mae']}，RMSE {model['train_rmse']}。相对持久性基线的 MAE 变化比例 {model.get('mae_improvement_vs_persist')}。",
            styles["body"],
        ))
        feature_bits = "；".join(
            f"{item['label']} {item['mae_increase']}" for item in model["importance"][:5]
        )
        block.append(_p(
            f"置换重要性（测试集 {model['importance_sample_n']} 行，重复 5 次）排在前面的是：{feature_bits}。单位是打乱该特征后 MAE 增加的 μg/m³。",
            styles["body"],
        ))
        block.append(_p(
            "基线的作用是给一个最低标准。若模型只比昨天好一点点，我就不会把这个好一点点讲成‘机器学习显著提升了空气质量预测’。它说明的是：在已经知道昨天的情况下，季节和邻居的昨天还有没有剩余信息。",
            styles["body"],
        ))
        sections.append(("预测模型：切分、基线和泄漏", block))

        block = []
        block.append(_p(
            "YOLOv11 是单阶段目标检测。它在图像上找框，并为框选一个类别。它不输出 μg/m³。把检测模型拿去回归一张表格里的 AQI，既用错了输入，也会在追问里站不住。",
            styles["body"],
        ))
        block.append(_p(yolo_text, styles["body"]))
        block.append(_p(
            "就算以后补上合格的图像，监测回答的仍是浓度，照片回答的只是某个画面里有没有车。空间尺度一个是城市或站点，一个是一张照片的视场；时间尺度一个是日或小时平均，一个是快门那一瞬间。本项目没有把两者相加，地图上也不叠检测框。",
            styles["body"],
        ))
        sections.append(("YOLO11 做了什么，没做什么", block))

        block = []
        for index, (question, spoken, logic) in enumerate(questions, start=1):
            chunk = [
                Paragraph(f"{index}. {escape(question)}", styles["q"]),
                Paragraph("<b>口述：</b>" + escape(spoken), styles["body"]),
                Paragraph("<b>再往下讲：</b>" + escape(logic), styles["body"]),
            ]
            block.append(KeepTogether(chunk))
        sections.append(("可能被追问的十个问题", block))

        glossary = [
            ("AQI", "空气质量指数", "由几种污染物的分指数取最大得到的无量纲指数。本项目里的全国 AQI 只是小时 AQI 的日均，不是正式日 AQI。"),
            ("PM2.5", "细颗粒物", "空气动力学直径小于或等于 2.5 微米的颗粒物。本项目的主要指标，单位 μg/m³。"),
            ("空间自相关", "spatial autocorrelation", "邻近位置的取值比随机排列更相像（正）或更相反（负）。Moran’s I 是其中一种全局指标。"),
            ("空间异质性", "spatial heterogeneity", "关系或方差随地点变化。一个全局 I 概括不了所有局部，所以还要看 LISA。"),
            ("可修面积元问题", "MAUP", "空间单元的大小和划法会改变统计结果。区域边界是分析者选定的。"),
            ("生态谬误", "ecological fallacy", "用集合体的相关或平均，去推断个体。城市平均不能代替街区或个人暴露。"),
            ("空间插值", "spatial interpolation", "用已知点估计未测点。IDW 按距离加权。没有掩膜时，图会看起来比数据更连续。"),
            ("协方差函数", "covariance function", "克里金用来描述‘距离越远，两点越不像’的函数。支撑不对时，函数拟合得再好也解释错尺度。"),
            ("时间序列切分", "temporal split", "训练日期全部早于测试日期。用来避免未来信息进测试集。"),
            ("基线模型", "baseline", "一个简单、诚实的对照。这里是持久性基线：预测值等于昨天。"),
            ("检测与回归", "detection vs regression", "检测输出框和类别；回归输出一个连续数。YOLO 做的是检测，浓度模型才是回归。"),
        ]
        rows = [[
            Paragraph("中文", styles["cellh"]),
            Paragraph("英文", styles["cellh"]),
            Paragraph("在本项目里怎么用", styles["cellh"]),
        ]]
        for zh, en, note in glossary:
            rows.append([
                Paragraph(escape(zh), styles["cell"]),
                Paragraph(escape(en), styles["cell"]),
                Paragraph(escape(note), styles["cell"]),
            ])
        table = Table(rows, colWidths=[32 * mm, 42 * mm, 100 * mm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GREEN),
            ("BACKGROUND", (0, 1), (-1, -1), PAPER),
            ("GRID", (0, 0), (-1, -1), 0.3, LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        sections.append(("中英名词对照", [table, Spacer(1, 3 * mm), _p("表里的英文用来听懂提问。回答时先用中文把定义说完，再补英文，避免只抛缩写。", styles["body"])]))

        block = []
        block.append(_p(
            "数据不是全国每小时、每一个子站。城市日平均会掩盖城区和郊区的差别，也会让数据更全的日子更有分量。坐标是代表点，西部城市的第 8 个近邻可能远到不符合‘邻’的直觉。",
            styles["body"],
        ))
        block.append(_p(
            "LISA 的 p 值没有校正。模型没有气象，也没有做因果识别：看不到政策冲击，没有对照城市。没有把检测模型接进浓度，因为输入不是图像，也没有许可清楚的标注集。",
            styles["body"],
        ))
        block.append(_p(
            "如果继续做，顺序应该是：补上可公开、可复现的气象和站点坐标；仍然用时间切分和持久性基线；在站点尺度上再讨论变异函数。不会为了作品集把 YOLO 的类别计数加进 PM2.5 的回归公式。",
            styles["body"],
        ))
        if CHINA_SUMMARY_JSON.exists():
            china = json.loads(CHINA_SUMMARY_JSON.read_text(encoding="utf-8"))
            bal = china["levels"]["balanced"]
            long_cover = china["coverage"]
            held_out = china["forecast"]["splits"]["test"]
            winter_long = china["moran_winter"]
            summer_long = china["moran_summer"]
            longer = []
            longer.append(_p(
                "仪表盘全国页用的是 2023–2024。同一仓库里还有一份 2016-01-01 至 2025-12-31 的城市日值，"
                f"发布口径里的城市名 {long_cover['cities']} 个，对上代表点 {long_cover['cities_with_coordinates']} 个，"
                f"日历成功 {long_cover['files_ok']} 天、缺失 {long_cover['files_missing']} 天。"
                "这不是县和乡镇，也没有人口加权。",
                styles["body"],
            ))
            longer.append(_p(
                f"同一批在 {bal['early_year']} 和 {bal['late_year']} 都至少有 {bal['min_days_each_year']} 个有效日的城市有 {bal['cities']} 个。"
                f"城市年平均从 {bal['early_mean']:.1f} 到 {bal['late_mean']:.1f} μg/m³，变化 {bal['delta']:.1f}（{bal['pct']*100:.1f}%）。"
                "不要把这组数和上面 2023 年 32.0、2024 年 30.4 相减，时段和城市集合都不一样。",
                styles["body"],
            ))
            longer.append(_p(
                f"这份长序列上，冬季 Moran’s I 是 {winter_long['I']:.3f}（n={winter_long['n']}），夏季是 {summer_long['I']:.3f}（n={summer_long['n']}）。"
                f"次日预测把 2025 整年留作测试：梯度提升 MAE {held_out['boosting']['mae']:.2f}，明天等于今天 {held_out['persistence']['mae']:.2f}，"
                f"技能分数 {held_out['boosting_skill']:.3f}，样本 {held_out['n']}。"
                "页面上画的是 2024 年下半年那次切分。两套都按时间切开，都没有随机划分。",
                styles["body"],
            ))
            sections.append(("更长的 2016–2025，页面上不混用", longer))

        sections.append(("局限和下一步", block))

        for title, flowables in sections:
            items.append(Mark(title, styles["h1"], title))
            items.extend(flowables)
            items.append(Spacer(1, 2 * mm))
        return items

    INTERVIEW_PDF.parent.mkdir(parents=True, exist_ok=True)
    frame = Frame(18 * mm, 16 * mm, A4[0] - 36 * mm, A4[1] - 32 * mm, showBoundary=0)
    collected: list[tuple[str, int]] = []

    class Doc(BaseDocTemplate):
        def afterFlowable(self, flowable):
            title = getattr(flowable, "toc_title", None)
            if title:
                collected.append((title, int(self.page)))

    def render(toc):
        collected.clear()
        document = Doc(
            str(INTERVIEW_PDF),
            pagesize=A4,
            title="空间大数据分析面试问答",
            author="空气质量时空仪表盘",
        )
        document.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_draw_page)])
        document.build(story_body(toc))
        return list(collected)

    first = render(None)
    render(first)
    # 第二次构建才会把目录页码写进去。页数以文件为准。
    data = INTERVIEW_PDF.read_bytes()
    page_count = data.count(b"/Type /Page\n") or data.count(b"/Type /Page")
    # 父节点 /Pages 也会被数进去，减去一次。
    pages_nodes = data.count(b"/Type /Pages")
    print(f"写出 {INTERVIEW_PDF}")
    print(f"目录条目 {len(first)}，页面标记约 {page_count}，Pages 节点 {pages_nodes}")
    return page_count


if __name__ == "__main__":
    build()
