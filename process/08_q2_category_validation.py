"""问题二：五类关键词业务画像、逻辑检查与推广单元结构汇总。

本脚本只读取现有 ``q2_keyword_scores.csv``，不重新计算 CRITIC 权重、
不重新运行 KMeans，也不修改任何已有分类标签或 result2.xlsx。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.set_loglevel("error")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"

SCORES_CSV = TABLE_DIR / "q2_keyword_scores.csv"
PROFILE_CSV = TABLE_DIR / "q2_category_profile.csv"
PROFILE_CHECK_CSV = TABLE_DIR / "q2_category_profile_check.csv"
BY_UNIT_CSV = TABLE_DIR / "q2_category_by_unit.csv"
STRUCTURE_FIGURE = FIGURE_DIR / "q2_category_structure_by_unit.png"

CATEGORY_COLUMN = "最终类别"
CATEGORY_ORDER = ["黄金词", "重点词", "潜力词", "问题词", "无效词"]
ID_COLUMNS = ["方案ID", "推广单元ID"]
PROFILE_INDICATORS = [
    "消费额",
    "CPC",
    "点击量",
    "浏览量",
    "浏览深度",
    "跳出率",
    "非跳出率",
    "平均访问时长_秒",
]
SCORE_INDICATORS = ["CostScore", "BenefitScore"]
REQUIRED_COLUMNS = [
    *ID_COLUMNS,
    CATEGORY_COLUMN,
    *PROFILE_INDICATORS,
    *SCORE_INDICATORS,
]


def save_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """统一保存 UTF-8 BOM CSV，并保留足够数值精度。"""
    dataframe.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def load_scores() -> pd.DataFrame:
    """读取现有分类结果，验证字段和类别，但不改写原文件。"""
    if not SCORES_CSV.exists():
        raise FileNotFoundError(f"未找到现有评分表：{SCORES_CSV}")
    data = pd.read_csv(
        SCORES_CSV,
        dtype={"方案ID": "string", "推广单元ID": "string"},
    )
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"q2_keyword_scores.csv 缺少字段：{'、'.join(missing)}")
    unexpected = sorted(set(data[CATEGORY_COLUMN].dropna()) - set(CATEGORY_ORDER))
    if unexpected:
        raise ValueError(f"评分表存在非正式分类标签：{unexpected}")
    if data[CATEGORY_COLUMN].isna().any():
        raise ValueError("评分表的最终类别存在缺失值。")
    return data


def build_category_profile(data: pd.DataFrame) -> pd.DataFrame:
    """按五类汇总规模、原始业务指标和模型得分。"""
    grouped = data.groupby(CATEGORY_COLUMN, observed=True)
    profile = pd.DataFrame(index=CATEGORY_ORDER)
    profile.index.name = "类别"
    profile["n"] = grouped.size().reindex(CATEGORY_ORDER, fill_value=0).astype(int)
    profile["占全部记录比例"] = profile["n"] / len(data)

    for indicator in PROFILE_INDICATORS:
        statistics = grouped[indicator].agg(["median", "mean"]).reindex(CATEGORY_ORDER)
        profile[f"{indicator}中位数"] = statistics["median"]
        profile[f"{indicator}均值"] = statistics["mean"]
    for indicator in SCORE_INDICATORS:
        profile[f"{indicator}中位数"] = grouped[indicator].median().reindex(CATEGORY_ORDER)

    return profile.reset_index()


def format_named_values(values: dict[str, float]) -> str:
    """将类别中位数压缩成适合检查说明的文本。"""
    return "，".join(f"{name}={value:.6g}" for name, value in values.items())


def build_profile_checks(data: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """程序化检查分类的成本、效益、无效词和原始指标业务方向。"""
    indexed = profile.set_index("类别")
    rows: list[dict[str, object]] = []

    low_cost = {
        category: float(indexed.loc[category, "CostScore中位数"])
        for category in ["黄金词", "潜力词"]
    }
    high_cost = {
        category: float(indexed.loc[category, "CostScore中位数"])
        for category in ["重点词", "问题词"]
    }
    cost_passed = bool(
        all(np.isfinite([*low_cost.values(), *high_cost.values()]))
        and max(low_cost.values()) < min(high_cost.values())
    )
    rows.append(
        {
            "check_name": "成本侧低成本类与高成本类分离",
            "passed": cost_passed,
            "description": (
                "要求黄金词、潜力词的 CostScore 中位数均低于重点词、问题词；"
                f"低成本类：{format_named_values(low_cost)}；"
                f"高成本类：{format_named_values(high_cost)}。"
            ),
        }
    )

    high_benefit = {
        category: float(indexed.loc[category, "BenefitScore中位数"])
        for category in ["黄金词", "重点词"]
    }
    low_benefit = {
        category: float(indexed.loc[category, "BenefitScore中位数"])
        for category in ["潜力词", "问题词"]
    }
    benefit_passed = bool(
        all(np.isfinite([*high_benefit.values(), *low_benefit.values()]))
        and min(high_benefit.values()) > max(low_benefit.values())
    )
    rows.append(
        {
            "check_name": "效益侧高效益类与低效益类分离",
            "passed": benefit_passed,
            "description": (
                "要求黄金词、重点词的 BenefitScore 中位数均高于潜力词、问题词；"
                f"高效益类：{format_named_values(high_benefit)}；"
                f"低效益类：{format_named_values(low_benefit)}。"
            ),
        }
    )

    invalid = data.loc[data[CATEGORY_COLUMN].eq("无效词")]
    zero_shares = {
        indicator: float(invalid[indicator].eq(0).mean())
        for indicator in ["消费额", "点击量", "浏览量"]
    }
    invalid_passed = bool(len(invalid) > 0 and min(zero_shares.values()) >= 0.90)
    rows.append(
        {
            "check_name": "无效词零消费零流量特征",
            "passed": invalid_passed,
            "description": (
                "以各项为0的记录占比不低于90%作为“主要表现为0”的判据；"
                + "，".join(f"{name}=0占比{share:.2%}" for name, share in zero_shares.items())
                + "。"
            ),
        }
    )

    # 原始指标采用类别中位数作宽松的组合检查。只有两个成本指标均反向，且
    # 五个正向效益指标至少四个反向，才判定为“明显”违反业务解释。
    cost_columns = ["消费额中位数", "CPC中位数"]
    benefit_columns = [
        "点击量中位数",
        "浏览量中位数",
        "浏览深度中位数",
        "非跳出率中位数",
        "平均访问时长_秒中位数",
    ]

    golden_high_cost_flags = [
        indexed.loc["黄金词", column]
        >= min(indexed.loc["重点词", column], indexed.loc["问题词", column])
        for column in cost_columns
    ]
    golden_low_benefit_flags = [
        indexed.loc["黄金词", column]
        <= max(indexed.loc["潜力词", column], indexed.loc["问题词", column])
        for column in benefit_columns
    ]
    golden_high_cost = all(golden_high_cost_flags)
    golden_low_benefit_count = int(sum(golden_low_benefit_flags))
    golden_passed = not (golden_high_cost and golden_low_benefit_count >= 4)
    rows.append(
        {
            "check_name": "黄金词原始指标业务解释",
            "passed": bool(golden_passed),
            "description": (
                "明显反向定义为：消费额、CPC中位数均不低于两个高成本类的较低值，"
                "且5个正向效益指标中至少4个不高于两个低效益类的较高值；"
                f"本类高成本条件={'是' if golden_high_cost else '否'}，"
                f"低效益反向指标数={golden_low_benefit_count}/5。"
            ),
        }
    )

    problem_low_cost_flags = [
        indexed.loc["问题词", column]
        <= max(indexed.loc["黄金词", column], indexed.loc["潜力词", column])
        for column in cost_columns
    ]
    problem_high_benefit_flags = [
        indexed.loc["问题词", column]
        >= min(indexed.loc["黄金词", column], indexed.loc["重点词", column])
        for column in benefit_columns
    ]
    problem_low_cost = all(problem_low_cost_flags)
    problem_high_benefit_count = int(sum(problem_high_benefit_flags))
    problem_passed = not (problem_low_cost and problem_high_benefit_count >= 4)
    rows.append(
        {
            "check_name": "问题词原始指标业务解释",
            "passed": bool(problem_passed),
            "description": (
                "明显反向定义为：消费额、CPC中位数均不高于两个低成本类的较高值，"
                "且5个正向效益指标中至少4个不低于两个高效益类的较低值；"
                f"本类低成本条件={'是' if problem_low_cost else '否'}，"
                f"高效益反向指标数={problem_high_benefit_count}/5。"
            ),
        }
    )
    return pd.DataFrame(rows, columns=["check_name", "passed", "description"])


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """先聚合后安全相除，分母为0时保留 NaN。"""
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = denominator.ne(0) & denominator.notna() & numerator.notna()
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def build_category_by_unit(data: pd.DataFrame) -> pd.DataFrame:
    """按方案和推广单元汇总五类结构及先求和后相除的整体指标。"""
    grouped = data.groupby(ID_COLUMNS, sort=True, observed=True)
    totals = grouped.agg(
        总关键词投放实例数=(CATEGORY_COLUMN, "size"),
        总消费额=("消费额", "sum"),
        总点击量=("点击量", "sum"),
        总浏览量=("浏览量", "sum"),
    )
    counts = (
        pd.crosstab([data["方案ID"], data["推广单元ID"]], data[CATEGORY_COLUMN])
        .reindex(index=totals.index, columns=CATEGORY_ORDER, fill_value=0)
        .astype(int)
    )

    result = totals[["总关键词投放实例数"]].copy()
    for category in CATEGORY_ORDER:
        result[f"{category}数量"] = counts[category]
    for category in CATEGORY_ORDER:
        result[f"{category}比例"] = counts[category] / result["总关键词投放实例数"]

    result["黄金词+重点词 数量"] = counts["黄金词"] + counts["重点词"]
    result["黄金词+重点词 比例"] = (
        result["黄金词+重点词 数量"] / result["总关键词投放实例数"]
    )
    result["问题词+无效词 数量"] = counts["问题词"] + counts["无效词"]
    result["问题词+无效词 比例"] = (
        result["问题词+无效词 数量"] / result["总关键词投放实例数"]
    )

    for column in ["总消费额", "总点击量", "总浏览量"]:
        result[column] = totals[column]
    result["单元总体CPC"] = safe_ratio(result["总消费额"], result["总点击量"])
    result["单元总体浏览深度"] = safe_ratio(result["总浏览量"], result["总点击量"])

    return result.reset_index().sort_values(ID_COLUMNS, kind="stable").reset_index(drop=True)


def configure_chinese_font() -> None:
    """配置常见中文字体回退。"""
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_unit_structure_figure(by_unit: pd.DataFrame) -> None:
    """绘制各推广单元五类关键词占比的100%堆积柱状图。"""
    configure_chinese_font()
    colors = {
        "黄金词": "#E6B800",
        "重点词": "#D95F5F",
        "潜力词": "#55A868",
        "问题词": "#8172B3",
        "无效词": "#9E9E9E",
    }
    positions = np.arange(len(by_unit))
    bottoms = np.zeros(len(by_unit), dtype=float)
    figure_width = max(10.0, 0.72 * len(by_unit))
    figure, axis = plt.subplots(figsize=(figure_width, 6.2))

    for category in CATEGORY_ORDER:
        values = by_unit[f"{category}比例"].to_numpy(dtype=float)
        axis.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.72,
            color=colors[category],
            label=category,
        )
        bottoms += values

    axis.set_xticks(positions)
    axis.set_xticklabels(by_unit["推广单元ID"], rotation=35, ha="right")
    axis.set_xlabel("推广单元ID")
    axis.set_ylabel("各类别关键词占比")
    axis.set_title("各推广单元五类关键词结构")
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
    axis.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=False)
    figure.tight_layout()
    figure.savefig(STRUCTURE_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)


def print_final_summary(
    profile: pd.DataFrame,
    checks: pd.DataFrame,
    by_unit: pd.DataFrame,
) -> None:
    """只打印用户指定的五项最终摘要。"""
    core_columns = [
        "类别",
        "n",
        "消费额中位数",
        "CPC中位数",
        "点击量中位数",
        "浏览深度中位数",
        "跳出率中位数",
        "平均访问时长_秒中位数",
        "CostScore中位数",
        "BenefitScore中位数",
    ]
    print("1. q2_category_profile 核心摘要")
    print(profile[core_columns].to_string(index=False))

    all_passed = bool(checks["passed"].all())
    print(f"2. category_profile_check 是否全部通过：{'是' if all_passed else '否'}")
    if not all_passed:
        for description in checks.loc[~checks["passed"], "description"]:
            print(f"提醒：{description}")

    unit_core_columns = [
        "方案ID",
        "推广单元ID",
        "总关键词投放实例数",
        "黄金词数量",
        "重点词数量",
        "问题词数量",
        "无效词数量",
    ]
    print("3. 每个推广单元关键词分类数量")
    print(by_unit[unit_core_columns].to_string(index=False))

    top_good = by_unit.nlargest(3, "黄金词+重点词 比例", keep="first")
    print("4. 黄金+重点比例最高的前3个推广单元")
    print(
        top_good[["方案ID", "推广单元ID", "黄金词+重点词 比例"]]
        .to_string(index=False)
    )

    top_problem = by_unit.nlargest(3, "问题词+无效词 比例", keep="first")
    print("5. 问题+无效比例最高的前3个推广单元")
    print(
        top_problem[["方案ID", "推广单元ID", "问题词+无效词 比例"]]
        .to_string(index=False)
    )


def main() -> None:
    data = load_scores()
    profile = build_category_profile(data)
    checks = build_profile_checks(data, profile)
    by_unit = build_category_by_unit(data)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(profile, PROFILE_CSV)
    save_csv(checks, PROFILE_CHECK_CSV)
    save_csv(by_unit, BY_UNIT_CSV)
    save_unit_structure_figure(by_unit)

    print_final_summary(profile, checks, by_unit)


if __name__ == "__main__":
    main()
