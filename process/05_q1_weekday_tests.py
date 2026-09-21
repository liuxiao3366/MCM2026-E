"""问题一：对日级指标进行星期效应统计检验。

检验流程：
1. 对周一至周日各组分别进行 Shapiro-Wilk 正态性检验；
2. 使用基于中位数的 Levene 检验检查方差齐性；
3. 七组均满足正态性且满足方差齐性时使用单因素 ANOVA，
   否则优先使用 Kruskal-Wallis 检验；
4. 整体差异显著时进行两两比较，并采用 Holm 法校正 p 值；
5. 报告整体效应量及两两比较效应量。

本脚本只检验星期分组之间的统计差异，不作因果解释。
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "daily_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "tables"

ASSUMPTION_CSV = OUTPUT_DIR / "q1_weekday_assumption_tests.csv"
OVERALL_CSV = OUTPUT_DIR / "q1_weekday_overall_tests.csv"
POSTHOC_CSV = OUTPUT_DIR / "q1_weekday_posthoc.csv"

ALPHA = 0.05
WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
METRICS = ["总消费额", "总点击量", "CTR", "CPC", "新注册数"]


def load_daily_summary() -> pd.DataFrame:
    """读取日级汇总表并构造有序星期变量。"""
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

    weekday_mapping = dict(enumerate(WEEKDAYS))
    daily["星期几"] = pd.Categorical(
        daily["日期"].dt.dayofweek.map(weekday_mapping),
        categories=WEEKDAYS,
        ordered=True,
    )
    return daily


def metric_groups(daily: pd.DataFrame, metric: str) -> dict[str, np.ndarray]:
    """按星期提取某指标的非缺失观测。"""
    groups = {
        weekday: daily.loc[daily["星期几"] == weekday, metric]
        .dropna()
        .to_numpy(dtype=float)
        for weekday in WEEKDAYS
    }
    empty_groups = [weekday for weekday, values in groups.items() if len(values) == 0]
    if empty_groups:
        raise ValueError(f"指标 {metric} 的以下星期组没有有效观测：{empty_groups}")
    return groups


def holm_adjust(p_values: list[float]) -> np.ndarray:
    """使用 Holm step-down 方法校正一组 p 值。"""
    p_array = np.asarray(p_values, dtype=float)
    number = len(p_array)
    order = np.argsort(p_array)
    adjusted_sorted = np.empty(number, dtype=float)

    running_max = 0.0
    for rank, original_index in enumerate(order):
        adjusted = (number - rank) * p_array[original_index]
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(running_max, 1.0)

    adjusted = np.empty(number, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def omega_squared(groups: list[np.ndarray]) -> float:
    """计算单因素 ANOVA 的 omega squared 效应量。"""
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


def epsilon_squared(kruskal_statistic: float, sample_size: int, groups: int) -> float:
    """计算 Kruskal-Wallis 的 epsilon squared 效应量。"""
    denominator = sample_size - groups
    if denominator <= 0:
        return np.nan
    return max(0.0, (kruskal_statistic - groups + 1) / denominator)


def hedges_g(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个独立样本的 Hedges' g，并保留方向。"""
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


def rank_biserial(first: np.ndarray, second: np.ndarray) -> float:
    """由 Mann-Whitney U 计算秩二分相关，并保留方向。"""
    u_statistic = stats.mannwhitneyu(
        first,
        second,
        alternative="two-sided",
        method="asymptotic",
    ).statistic
    return 2 * u_statistic / (len(first) * len(second)) - 1


def parametric_posthoc(
    metric: str,
    groups: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """执行等方差独立样本 t 检验，并使用 Holm 法校正。"""
    raw_rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []

    for first_name, second_name in combinations(WEEKDAYS, 2):
        first = groups[first_name]
        second = groups[second_name]
        result = stats.ttest_ind(first, second, equal_var=True, nan_policy="omit")
        raw_p_values.append(float(result.pvalue))
        raw_rows.append(
            {
                "指标": metric,
                "事后方法": "两独立样本t检验",
                "组1": first_name,
                "组2": second_name,
                "统计量名称": "t",
                "统计量": float(result.statistic),
                "原始p值": float(result.pvalue),
                "校正方法": "Holm",
                "效应量名称": "Hedges_g",
                "效应量": hedges_g(first, second),
            }
        )

    adjusted_p_values = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(raw_rows, adjusted_p_values):
        row["校正p值"] = float(adjusted_p)
        row["校正后显著"] = bool(adjusted_p < ALPHA)
    return raw_rows


def dunn_posthoc(
    metric: str,
    groups: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """执行 Dunn 秩和事后检验，并使用 Holm 法校正。"""
    values = np.concatenate([groups[weekday] for weekday in WEEKDAYS])
    labels = np.concatenate(
        [np.repeat(weekday, len(groups[weekday])) for weekday in WEEKDAYS]
    )
    ranks = stats.rankdata(values, method="average")
    sample_size = len(values)

    _, tie_counts = np.unique(values, return_counts=True)
    tie_sum = np.sum(tie_counts**3 - tie_counts)
    tie_correction = tie_sum / (12 * (sample_size - 1))
    rank_variance = sample_size * (sample_size + 1) / 12 - tie_correction

    mean_ranks = {
        weekday: float(ranks[labels == weekday].mean()) for weekday in WEEKDAYS
    }
    raw_rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []

    for first_name, second_name in combinations(WEEKDAYS, 2):
        first = groups[first_name]
        second = groups[second_name]
        denominator = np.sqrt(
            rank_variance * (1 / len(first) + 1 / len(second))
        )
        z_statistic = (mean_ranks[first_name] - mean_ranks[second_name]) / denominator
        p_value = float(2 * stats.norm.sf(abs(z_statistic)))
        raw_p_values.append(p_value)
        raw_rows.append(
            {
                "指标": metric,
                "事后方法": "Dunn检验",
                "组1": first_name,
                "组2": second_name,
                "统计量名称": "z",
                "统计量": float(z_statistic),
                "原始p值": p_value,
                "校正方法": "Holm",
                "效应量名称": "rank_biserial_r",
                "效应量": rank_biserial(first, second),
            }
        )

    adjusted_p_values = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(raw_rows, adjusted_p_values):
        row["校正p值"] = float(adjusted_p)
        row["校正后显著"] = bool(adjusted_p < ALPHA)
    return raw_rows


def analyze_metric(
    daily: pd.DataFrame,
    metric: str,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """完成单个指标的前提、整体及事后检验。"""
    groups = metric_groups(daily, metric)
    group_values = [groups[weekday] for weekday in WEEKDAYS]

    assumption_rows: list[dict[str, object]] = []
    shapiro_p_values: list[float] = []
    for weekday in WEEKDAYS:
        result = stats.shapiro(groups[weekday])
        shapiro_p_values.append(float(result.pvalue))
        assumption_rows.append(
            {
                "指标": metric,
                "前提检验": "Shapiro-Wilk正态性检验",
                "组别": weekday,
                "样本量": len(groups[weekday]),
                "统计量": float(result.statistic),
                "p值": float(result.pvalue),
                "通过条件": bool(result.pvalue >= ALPHA),
            }
        )

    levene_result = stats.levene(*group_values, center="median")
    assumption_rows.append(
        {
            "指标": metric,
            "前提检验": "Levene方差齐性检验（中位数）",
            "组别": "全部星期",
            "样本量": sum(len(group) for group in group_values),
            "统计量": float(levene_result.statistic),
            "p值": float(levene_result.pvalue),
            "通过条件": bool(levene_result.pvalue >= ALPHA),
        }
    )

    normality_passed = all(p_value >= ALPHA for p_value in shapiro_p_values)
    equal_variance_passed = bool(levene_result.pvalue >= ALPHA)
    anova_conditions = normality_passed and equal_variance_passed

    if anova_conditions:
        overall_result = stats.f_oneway(*group_values)
        test_name = "单因素ANOVA"
        statistic_name = "F"
        effect_name = "omega_squared"
        effect_value = omega_squared(group_values)
    else:
        overall_result = stats.kruskal(*group_values)
        test_name = "Kruskal-Wallis"
        statistic_name = "H"
        effect_name = "epsilon_squared"
        effect_value = epsilon_squared(
            float(overall_result.statistic),
            sum(len(group) for group in group_values),
            len(group_values),
        )

    overall_significant = bool(overall_result.pvalue < ALPHA)
    posthoc_rows: list[dict[str, object]] = []
    if overall_significant:
        if anova_conditions:
            posthoc_rows = parametric_posthoc(metric, groups)
        else:
            posthoc_rows = dunn_posthoc(metric, groups)

    significant_pairs = sum(
        bool(row["校正后显著"]) for row in posthoc_rows
    )
    overall_row = {
        "指标": metric,
        "显著性水平": ALPHA,
        "正态性全部通过": normality_passed,
        "最小Shapiro_p值": min(shapiro_p_values),
        "Levene统计量": float(levene_result.statistic),
        "Levene_p值": float(levene_result.pvalue),
        "方差齐性通过": equal_variance_passed,
        "采用检验": test_name,
        "统计量名称": statistic_name,
        "统计量": float(overall_result.statistic),
        "p值": float(overall_result.pvalue),
        "整体差异显著": overall_significant,
        "效应量名称": effect_name,
        "效应量": effect_value,
        "事后比较方法": (
            "两独立样本t检验+Holm"
            if overall_significant and anova_conditions
            else "Dunn检验+Holm"
            if overall_significant
            else "整体差异不显著，未执行"
        ),
        "事后比较总数": len(posthoc_rows),
        "校正后显著比较数": significant_pairs,
    }
    return assumption_rows, overall_row, posthoc_rows


def format_p_value(value: float) -> str:
    """为终端摘要格式化 p 值。"""
    return f"{value:.3e}" if value < 0.001 else f"{value:.4f}"


def print_summary(overall: pd.DataFrame) -> None:
    """打印整体检验的简洁摘要。"""
    print("星期效应统计检验摘要（alpha=0.05）")
    for row in overall.itertuples(index=False):
        normality = "通过" if row.正态性全部通过 else "未通过"
        variance = "通过" if row.方差齐性通过 else "未通过"
        significance = "显著" if row.整体差异显著 else "不显著"
        print(
            f"{row.指标}: 正态性{normality}, 方差齐性{variance}; "
            f"{row.采用检验} {row.统计量名称}={row.统计量:.4f}, "
            f"p={format_p_value(row.p值)}, "
            f"{row.效应量名称}={row.效应量:.4f}; "
            f"整体差异{significance}, "
            f"Holm校正后显著两两比较 {row.校正后显著比较数}/{row.事后比较总数}。"
        )
    print("注：上述结果仅表示不同星期组的分布存在或不存在统计差异，不代表因果关系。")


def main() -> None:
    daily = load_daily_summary()
    all_assumptions: list[dict[str, object]] = []
    all_overall: list[dict[str, object]] = []
    all_posthoc: list[dict[str, object]] = []

    for metric in METRICS:
        assumptions, overall, posthoc = analyze_metric(daily, metric)
        all_assumptions.extend(assumptions)
        all_overall.append(overall)
        all_posthoc.extend(posthoc)

    assumption_df = pd.DataFrame(all_assumptions)
    overall_df = pd.DataFrame(all_overall)
    posthoc_columns = [
        "指标",
        "事后方法",
        "组1",
        "组2",
        "统计量名称",
        "统计量",
        "原始p值",
        "校正方法",
        "校正p值",
        "校正后显著",
        "效应量名称",
        "效应量",
    ]
    posthoc_df = pd.DataFrame(all_posthoc, columns=posthoc_columns)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assumption_df.to_csv(
        ASSUMPTION_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    overall_df.to_csv(
        OVERALL_CSV,
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

    print_summary(overall_df)


if __name__ == "__main__":
    main()
