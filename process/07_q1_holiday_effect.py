"""问题一：基于 2025 年官方放假安排的假日效应描述与组间检验。

官方依据：国务院办公厅《关于2025年部分节假日安排的通知》
（国办发明电〔2024〕12号）。
https://www.gov.cn/zhengce/zhengceku/202411/content_6986383.htm

本脚本只分析不同日期类型之间的统计关联和分布差异，不作因果解释。
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.set_loglevel("error")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "daily_summary.csv"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"

SUMMARY_CSV = TABLE_DIR / "q1_holiday_summary.csv"
OVERALL_TESTS_CSV = TABLE_DIR / "q1_holiday_overall_tests.csv"
POSTHOC_CSV = TABLE_DIR / "q1_holiday_posthoc.csv"
EFFECT_SIZES_CSV = TABLE_DIR / "q1_holiday_effect_sizes.csv"

ALPHA = 0.05
DATE_TYPES = [
    "普通工作日",
    "普通周末",
    "元旦",
    "春节",
    "清明节",
    "劳动节",
    "端午节",
    "国庆/中秋假期",
]
# 元旦仅有 1 个观测日：保留在描述统计与图形中，不进入正式推断。
INFERENCE_DATE_TYPES = [date_type for date_type in DATE_TYPES if date_type != "元旦"]
HOLIDAY_TYPES = ["元旦", "春节", "清明节", "劳动节", "端午节", "国庆/中秋假期"]
METRICS = ["总消费额", "总点击量", "CTR", "CPC", "新注册数"]

# 官方 2025 年完整放假日期。用闭区间生成，避免手工漏日。
HOLIDAY_RANGES = {
    "元旦": [("2025-01-01", "2025-01-01")],
    "春节": [("2025-01-28", "2025-02-04")],
    "清明节": [("2025-04-04", "2025-04-06")],
    "劳动节": [("2025-05-01", "2025-05-05")],
    "端午节": [("2025-05-31", "2025-06-02")],
    "国庆/中秋假期": [("2025-10-01", "2025-10-08")],
}

# 官方通知明确规定的周末调休上班日。这些日期应归为普通工作日。
MAKEUP_WORKDAYS = {
    pd.Timestamp("2025-01-26"),
    pd.Timestamp("2025-02-08"),
    pd.Timestamp("2025-04-27"),
    pd.Timestamp("2025-09-28"),
    pd.Timestamp("2025-10-11"),
}


def configure_chinese_font() -> None:
    """设置常用中文字体回退顺序。"""
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_daily_summary() -> pd.DataFrame:
    """读取并校验日级汇总数据。"""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"未找到日级汇总文件：{INPUT_CSV}")

    daily = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    required = {"日期", *METRICS}
    missing = required.difference(daily.columns)
    if missing:
        missing_text = "、".join(sorted(missing))
        raise ValueError(f"daily_summary.csv 缺少字段：{missing_text}")

    daily["日期"] = pd.to_datetime(daily["日期"], errors="coerce")
    if daily["日期"].isna().any():
        raise ValueError("daily_summary.csv 中存在无法解析的日期。")
    if daily["日期"].duplicated().any():
        raise ValueError("daily_summary.csv 中存在重复日期。")

    for metric in METRICS:
        daily[metric] = pd.to_numeric(daily[metric], errors="raise")

    daily = daily.sort_values("日期").reset_index(drop=True)
    date_steps = daily["日期"].diff().dropna()
    if not date_steps.eq(pd.Timedelta(days=1)).all():
        raise ValueError("日期序列不连续，无法完整执行日历分类。")
    return daily


def build_holiday_lookup() -> dict[pd.Timestamp, str]:
    """将官方假期区间展开为“日期—假期名称”映射。"""
    lookup: dict[pd.Timestamp, str] = {}
    for holiday_name, ranges in HOLIDAY_RANGES.items():
        for start, end in ranges:
            for date in pd.date_range(start=start, end=end, freq="D"):
                if date in lookup:
                    raise ValueError(f"假期日期重复：{date:%Y-%m-%d}")
                lookup[date] = holiday_name
    return lookup


def classify_dates(daily: pd.DataFrame) -> pd.DataFrame:
    """按“假期优先、调休其次、星期最后”的顺序标记日期类型。"""
    holiday_lookup = build_holiday_lookup()

    def classify(date: pd.Timestamp) -> str:
        if date in holiday_lookup:
            return holiday_lookup[date]
        if date in MAKEUP_WORKDAYS:
            return "普通工作日"
        if date.dayofweek >= 5:
            return "普通周末"
        return "普通工作日"

    classified = daily.copy()
    classified["日期类型"] = pd.Categorical(
        classified["日期"].map(classify),
        categories=DATE_TYPES,
        ordered=True,
    )
    if classified["日期类型"].isna().any():
        raise ValueError("存在未能分类的日期。")

    # 对官方调休日做显式复核，防止后续逻辑调整时误归为普通周末。
    classified_lookup = classified.set_index("日期")["日期类型"].astype(str)
    for date in MAKEUP_WORKDAYS:
        if date not in classified_lookup.index or classified_lookup.loc[date] != "普通工作日":
            raise ValueError(f"调休上班日分类错误：{date:%Y-%m-%d}")
    return classified


def safe_shapiro(values: np.ndarray) -> tuple[float, float, bool, str]:
    """执行 Shapiro-Wilk；样本不足或常数样本时标记为不可检验。"""
    if len(values) < 3:
        return np.nan, np.nan, False, "样本量小于3"
    if np.all(values == values[0]):
        return np.nan, np.nan, False, "组内无变异"
    result = stats.shapiro(values)
    return (
        float(result.statistic),
        float(result.pvalue),
        bool(result.pvalue >= ALPHA),
        "可检验",
    )


def build_descriptive_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """计算各日期类型的描述性统计和组内正态性检查。"""
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        for date_type in DATE_TYPES:
            values = (
                daily.loc[daily["日期类型"] == date_type, metric]
                .dropna()
                .to_numpy(dtype=float)
            )
            shapiro_w, shapiro_p, normality_passed, normality_note = safe_shapiro(values)
            rows.append(
                {
                    "指标": metric,
                    "日期类型": date_type,
                    "样本数": len(values),
                    "均值": float(np.mean(values)),
                    "中位数": float(np.median(values)),
                    "标准差": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "第一四分位数": float(np.quantile(values, 0.25)),
                    "第三四分位数": float(np.quantile(values, 0.75)),
                    "Shapiro_W": shapiro_w,
                    "Shapiro_p值": shapiro_p,
                    "正态性通过": normality_passed,
                    "正态性检验说明": normality_note,
                }
            )
    return pd.DataFrame(rows)


def metric_groups(
    daily: pd.DataFrame,
    metric: str,
    date_types: list[str],
) -> dict[str, np.ndarray]:
    """提取某指标在指定正式推断日期类型下的非缺失观测。"""
    groups = {
        date_type: daily.loc[daily["日期类型"] == date_type, metric]
        .dropna()
        .to_numpy(dtype=float)
        for date_type in date_types
    }
    empty = [date_type for date_type, values in groups.items() if len(values) == 0]
    if empty:
        raise ValueError(f"指标 {metric} 的以下日期类型无有效观测：{empty}")
    return groups


def holm_adjust(p_values: list[float]) -> np.ndarray:
    """使用 Holm step-down 方法校正 p 值。"""
    p_array = np.asarray(p_values, dtype=float)
    number = len(p_array)
    order = np.argsort(p_array)
    adjusted_sorted = np.empty(number, dtype=float)
    running_max = 0.0

    for rank, original_index in enumerate(order):
        adjusted_value = (number - rank) * p_array[original_index]
        running_max = max(running_max, adjusted_value)
        adjusted_sorted[rank] = min(running_max, 1.0)

    adjusted = np.empty(number, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def omega_squared(groups: list[np.ndarray]) -> float:
    """计算单因素 ANOVA 的 omega squared。"""
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
    ss_within = sum(((group - group.mean()) ** 2).sum() for group in groups)
    df_between = len(groups) - 1
    df_within = len(all_values) - len(groups)
    ms_within = ss_within / df_within
    denominator = ss_between + ss_within + ms_within
    if denominator == 0:
        return np.nan
    return max(0.0, (ss_between - df_between * ms_within) / denominator)


def epsilon_squared(statistic: float, sample_size: int, group_count: int) -> float:
    """计算 Kruskal-Wallis 的 epsilon squared。"""
    denominator = sample_size - group_count
    if denominator <= 0:
        return np.nan
    return max(0.0, (statistic - group_count + 1) / denominator)


def rank_biserial(first: np.ndarray, second: np.ndarray) -> float:
    """计算 Mann-Whitney 秩二分相关，正值表示第一组整体偏大。"""
    u_statistic = stats.mannwhitneyu(
        first,
        second,
        alternative="two-sided",
        method="asymptotic",
    ).statistic
    return 2 * u_statistic / (len(first) * len(second)) - 1


def hedges_g(first: np.ndarray, second: np.ndarray) -> float:
    """计算两独立样本 Hedges' g，正值表示第一组均值偏大。"""
    n_first, n_second = len(first), len(second)
    degrees_of_freedom = n_first + n_second - 2
    pooled_variance = (
        (n_first - 1) * np.var(first, ddof=1)
        + (n_second - 1) * np.var(second, ddof=1)
    ) / degrees_of_freedom
    if pooled_variance <= 0:
        return np.nan
    cohens_d = (first.mean() - second.mean()) / np.sqrt(pooled_variance)
    correction = 1 - 3 / (4 * (n_first + n_second) - 9)
    return cohens_d * correction


def dunn_posthoc(
    metric: str,
    groups: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """执行 Dunn 两两检验，并使用 Holm 法校正。"""
    group_names = list(groups)
    values = np.concatenate([groups[date_type] for date_type in group_names])
    labels = np.concatenate(
        [np.repeat(date_type, len(groups[date_type])) for date_type in group_names]
    )
    ranks = stats.rankdata(values, method="average")
    sample_size = len(values)

    _, tie_counts = np.unique(values, return_counts=True)
    tie_sum = np.sum(tie_counts**3 - tie_counts)
    tie_correction = tie_sum / (12 * (sample_size - 1))
    rank_variance = sample_size * (sample_size + 1) / 12 - tie_correction
    mean_ranks = {
        date_type: float(ranks[labels == date_type].mean()) for date_type in group_names
    }

    rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for first_name, second_name in combinations(group_names, 2):
        first = groups[first_name]
        second = groups[second_name]
        denominator = np.sqrt(rank_variance * (1 / len(first) + 1 / len(second)))
        z_statistic = (mean_ranks[first_name] - mean_ranks[second_name]) / denominator
        p_value = float(2 * stats.norm.sf(abs(z_statistic)))
        raw_p_values.append(p_value)
        rows.append(
            {
                "指标": metric,
                "事后方法": "Dunn检验",
                "组1": first_name,
                "组1样本数": len(first),
                "组2": second_name,
                "组2样本数": len(second),
                "统计量名称": "z",
                "统计量": float(z_statistic),
                "原始p值": p_value,
                "校正方法": "Holm",
                "效应量名称": "rank_biserial_r",
                "效应量": rank_biserial(first, second),
            }
        )

    adjusted_p_values = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(rows, adjusted_p_values):
        row["校正p值"] = float(adjusted_p)
        row["校正后显著"] = bool(adjusted_p < ALPHA)
    return rows


def parametric_posthoc(
    metric: str,
    groups: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """执行等方差两独立样本 t 检验，并使用 Holm 法校正。"""
    group_names = list(groups)
    rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for first_name, second_name in combinations(group_names, 2):
        first = groups[first_name]
        second = groups[second_name]
        result = stats.ttest_ind(first, second, equal_var=True)
        raw_p_values.append(float(result.pvalue))
        rows.append(
            {
                "指标": metric,
                "事后方法": "两独立样本t检验",
                "组1": first_name,
                "组1样本数": len(first),
                "组2": second_name,
                "组2样本数": len(second),
                "统计量名称": "t",
                "统计量": float(result.statistic),
                "原始p值": float(result.pvalue),
                "校正方法": "Holm",
                "效应量名称": "Hedges_g",
                "效应量": hedges_g(first, second),
            }
        )

    adjusted_p_values = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(rows, adjusted_p_values):
        row["校正p值"] = float(adjusted_p)
        row["校正后显著"] = bool(adjusted_p < ALPHA)
    return rows


def analyze_metric(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """完成单个指标的前提检查、整体检验和事后比较。"""
    groups = metric_groups(daily, metric, INFERENCE_DATE_TYPES)
    values = [groups[date_type] for date_type in INFERENCE_DATE_TYPES]
    metric_summary = summary.loc[
        (summary["指标"] == metric)
        & summary["日期类型"].isin(INFERENCE_DATE_TYPES)
    ]

    all_normality_testable = metric_summary["Shapiro_p值"].notna().all()
    all_normality_passed = bool(
        all_normality_testable and metric_summary["正态性通过"].all()
    )
    valid_shapiro = metric_summary["Shapiro_p值"].dropna()
    minimum_shapiro_p = float(valid_shapiro.min()) if len(valid_shapiro) else np.nan

    levene = stats.levene(*values, center="median")
    equal_variance_passed = bool(levene.pvalue >= ALPHA)
    anova_conditions = all_normality_passed and equal_variance_passed

    if anova_conditions:
        test = stats.f_oneway(*values)
        test_name = "单因素ANOVA"
        statistic_name = "F"
        effect_name = "omega_squared"
        effect_value = omega_squared(values)
    else:
        test = stats.kruskal(*values)
        test_name = "Kruskal-Wallis"
        statistic_name = "H"
        effect_name = "epsilon_squared"
        effect_value = epsilon_squared(
            float(test.statistic),
            sum(len(group) for group in values),
            len(values),
        )

    significant = bool(test.pvalue < ALPHA)
    posthoc_rows: list[dict[str, object]] = []
    if significant:
        posthoc_rows = (
            parametric_posthoc(metric, groups)
            if anova_conditions
            else dunn_posthoc(metric, groups)
        )

    overall_row = {
        "指标": metric,
        "显著性水平": ALPHA,
        "推断范围": "排除元旦（单日样本）",
        "纳入日期类型": "；".join(INFERENCE_DATE_TYPES),
        "各组样本量": "；".join(
            f"{date_type}:{len(groups[date_type])}"
            for date_type in INFERENCE_DATE_TYPES
        ),
        "组数": len(INFERENCE_DATE_TYPES),
        "总样本数": sum(len(group) for group in values),
        "所有组正态性可检验": all_normality_testable,
        "所有组正态性通过": all_normality_passed,
        "最小有效Shapiro_p值": minimum_shapiro_p,
        "Levene统计量": float(levene.statistic),
        "Levene_p值": float(levene.pvalue),
        "方差齐性通过": equal_variance_passed,
        "采用检验": test_name,
        "统计量名称": statistic_name,
        "统计量": float(test.statistic),
        "p值": float(test.pvalue),
        "整体差异显著": significant,
        "整体效应量名称": effect_name,
        "整体效应量": effect_value,
        "事后比较方法": (
            "两独立样本t检验+Holm"
            if significant and anova_conditions
            else "Dunn检验+Holm"
            if significant
            else "整体差异不显著，未执行"
        ),
        "事后比较总数": len(posthoc_rows),
        "校正后显著比较数": sum(
            bool(row["校正后显著"]) for row in posthoc_rows
        ),
    }
    return overall_row, posthoc_rows


def build_holiday_effect_sizes(daily: pd.DataFrame) -> pd.DataFrame:
    """计算各具体假期相对普通工作日的中位数变化率和秩效应量。"""
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        baseline = (
            daily.loc[daily["日期类型"] == "普通工作日", metric]
            .dropna()
            .to_numpy(dtype=float)
        )
        baseline_median = float(np.median(baseline))
        for holiday in HOLIDAY_TYPES:
            holiday_values = (
                daily.loc[daily["日期类型"] == holiday, metric]
                .dropna()
                .to_numpy(dtype=float)
            )
            holiday_median = float(np.median(holiday_values))
            relative_change = (
                (holiday_median - baseline_median) / baseline_median
                if baseline_median != 0
                else np.nan
            )
            rows.append(
                {
                    "指标": metric,
                    "假期": holiday,
                    "基准组": "普通工作日",
                    "假期样本数": len(holiday_values),
                    "基准组样本数": len(baseline),
                    "假期中位数": holiday_median,
                    "普通工作日中位数": baseline_median,
                    "中位数相对变化率": relative_change,
                    "效应量名称": "rank_biserial_r",
                    "效应量": rank_biserial(holiday_values, baseline),
                    "效应量方向": "正值表示假期整体偏高，负值表示假期整体偏低",
                }
            )
    return pd.DataFrame(rows)


def save_boxplot(
    daily: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    as_percent: bool = False,
) -> None:
    """为一个指标绘制日期类型箱线图。"""
    scale = 100 if as_percent else 1
    values = [
        (daily.loc[daily["日期类型"] == date_type, metric] * scale)
        .dropna()
        .to_numpy()
        for date_type in DATE_TYPES
    ]

    figure, axis = plt.subplots(figsize=(13, 6.5))
    boxplot = axis.boxplot(
        values,
        tick_labels=DATE_TYPES,
        patch_artist=True,
        widths=0.62,
        medianprops={"color": "#B22222", "linewidth": 1.8},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": "#777777",
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
    )
    colors = [
        "#8EC9E6",
        "#B8D8E8",
        "#F6D186",
        "#F3B562",
        "#F4C095",
        "#E8A87C",
        "#DDA15E",
        "#D47766",
    ]
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)

    axis.set_title(title)
    axis.set_xlabel("日期类型")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_all_boxplots(daily: pd.DataFrame) -> None:
    """生成五个指标的日期类型箱线图。"""
    configure_chinese_font()
    plot_specs = [
        ("总消费额", "日期类型与每日总消费分布", "每日总消费（元）", "q1_holiday_total_cost_boxplot.png", False),
        ("总点击量", "日期类型与每日总点击分布", "每日总点击量", "q1_holiday_total_clicks_boxplot.png", False),
        ("CTR", "日期类型与每日点击率分布", "CTR（%）", "q1_holiday_ctr_boxplot.png", True),
        ("CPC", "日期类型与每日单次点击成本分布", "CPC（元/次）", "q1_holiday_cpc_boxplot.png", False),
        ("新注册数", "日期类型与每日新注册数分布", "新注册数", "q1_holiday_registrations_boxplot.png", False),
    ]
    for metric, title, ylabel, filename, as_percent in plot_specs:
        save_boxplot(daily, metric, title, ylabel, filename, as_percent)


def format_p_value(value: float) -> str:
    """格式化终端摘要中的 p 值。"""
    return f"{value:.3e}" if value < 0.001 else f"{value:.4f}"


def print_summary(daily: pd.DataFrame, overall: pd.DataFrame) -> None:
    """打印日期类型样本数和整体检验的简洁摘要。"""
    counts = daily["日期类型"].value_counts(sort=False).reindex(DATE_TYPES)
    print("日期类型样本数")
    print(counts.to_string())
    print("\n假日效应整体检验摘要（alpha=0.05）")
    for row in overall.itertuples(index=False):
        significance = "显著" if row.整体差异显著 else "不显著"
        print(
            f"{row.指标}: {row.采用检验} {row.统计量名称}={row.统计量:.4f}, "
            f"p={format_p_value(row.p值)}, "
            f"{row.整体效应量名称}={row.整体效应量:.4f}; "
            f"整体差异{significance}, "
            f"Holm校正后显著两两比较 {row.校正后显著比较数}/{row.事后比较总数}。"
        )
    print("注：结果仅描述日期类型之间的统计差异，不构成假期的因果效应。")


def main() -> None:
    daily = classify_dates(load_daily_summary())
    descriptive_summary = build_descriptive_summary(daily)

    overall_rows: list[dict[str, object]] = []
    posthoc_rows: list[dict[str, object]] = []
    for metric in METRICS:
        overall, posthoc = analyze_metric(daily, descriptive_summary, metric)
        overall_rows.append(overall)
        posthoc_rows.extend(posthoc)

    overall_df = pd.DataFrame(overall_rows)
    posthoc_columns = [
        "指标",
        "事后方法",
        "组1",
        "组1样本数",
        "组2",
        "组2样本数",
        "统计量名称",
        "统计量",
        "原始p值",
        "校正方法",
        "校正p值",
        "校正后显著",
        "效应量名称",
        "效应量",
    ]
    posthoc_df = pd.DataFrame(posthoc_rows, columns=posthoc_columns)
    effect_sizes = build_holiday_effect_sizes(daily)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    descriptive_summary.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    overall_df.to_csv(
        OVERALL_TESTS_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    posthoc_df.to_csv(
        POSTHOC_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    effect_sizes.to_csv(
        EFFECT_SIZES_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )

    plot_all_boxplots(daily)
    print_summary(daily, overall_df)


if __name__ == "__main__":
    main()
