"""问题一：SEM 日级投放指标与新注册数的时间滞后关联分析。

本脚本包含：
1. 每日总消费额、总点击量与未来 0--7 天新注册数的 Pearson、
   Spearman 相关分析；
2. 以当天新注册数为因变量、当天及前 1--3 天点击量为主要解释变量的
   分布滞后 OLS 模型，并控制星期和月份；
3. 将点击量滞后扩展至 0--7 天的敏感性分析；
4. HAC 稳健标准误、调整 R 方、VIF、滞后项联合检验与累计效应检验。

注意：结果仅用于描述时间关联结构，不用于识别因果效应。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.set_loglevel("error")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "daily_summary.csv"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"

CORRELATION_CSV = TABLE_DIR / "q1_lag_correlations.csv"
COEFFICIENT_CSV = TABLE_DIR / "q1_distributed_lag_coefficients.csv"
MODEL_SUMMARY_CSV = TABLE_DIR / "q1_distributed_lag_model_summary.csv"
CORRELATION_FIGURE = FIGURE_DIR / "q1_lag_correlation_curves.png"

MAX_CORRELATION_LAG = 7
PRIMARY_MODEL_LAG = 3
SENSITIVITY_MODEL_LAG = 7
HAC_MAX_LAG = 7
ALPHA = 0.05

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
AD_METRICS = ["总消费额", "总点击量"]


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
    """读取并校验连续的日级汇总数据。"""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"未找到日级汇总文件：{INPUT_CSV}")

    daily = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    required = {"日期", "总消费额", "总点击量", "新注册数"}
    missing = required.difference(daily.columns)
    if missing:
        missing_text = "、".join(sorted(missing))
        raise ValueError(f"daily_summary.csv 缺少字段：{missing_text}")

    daily["日期"] = pd.to_datetime(daily["日期"], errors="coerce")
    if daily["日期"].isna().any():
        raise ValueError("daily_summary.csv 中存在无法解析的日期。")
    if daily["日期"].duplicated().any():
        raise ValueError("daily_summary.csv 中存在重复日期。")

    for column in ["总消费额", "总点击量", "新注册数"]:
        daily[column] = pd.to_numeric(daily[column], errors="raise")

    daily = daily.sort_values("日期").reset_index(drop=True)
    date_steps = daily["日期"].diff().dropna()
    if not date_steps.eq(pd.Timedelta(days=1)).all():
        raise ValueError("日期序列不连续，不能直接用行位移表示自然日滞后。")
    return daily


def calculate_lag_correlations(daily: pd.DataFrame) -> pd.DataFrame:
    """计算投放指标与未来第 0--7 天注册数的相关系数。

    在第 t 行使用 ``新注册数.shift(-lag)``，所以 lag=k 的含义始终是：
    第 t 天广告投放指标对应第 t+k 天新注册数。
    """
    rows: list[dict[str, object]] = []

    for metric in AD_METRICS:
        for lag in range(MAX_CORRELATION_LAG + 1):
            future_registrations = daily["新注册数"].shift(-lag)
            paired = pd.DataFrame(
                {
                    "投放指标": daily[metric],
                    "未来注册数": future_registrations,
                }
            ).dropna()

            pearson = stats.pearsonr(paired["投放指标"], paired["未来注册数"])
            spearman = stats.spearmanr(paired["投放指标"], paired["未来注册数"])
            rows.append(
                {
                    "投放指标": metric,
                    "lag": lag,
                    "lag定义": f"第t天{metric}对应第t+{lag}天新注册数",
                    "样本对数": len(paired),
                    "Pearson_r": float(pearson.statistic),
                    "Pearson_p值": float(pearson.pvalue),
                    "Spearman_rho": float(spearman.statistic),
                    "Spearman_p值": float(spearman.pvalue),
                }
            )
    return pd.DataFrame(rows)


def plot_lag_correlations(correlations: pd.DataFrame) -> None:
    """绘制 Pearson、Spearman 系数随滞后天数的变化。"""
    configure_chinese_font()
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)

    for axis, metric in zip(axes, AD_METRICS):
        data = correlations.loc[correlations["投放指标"] == metric]
        axis.plot(
            data["lag"],
            data["Pearson_r"],
            marker="o",
            linewidth=2,
            color="#2F6B9A",
            label="Pearson",
        )
        axis.plot(
            data["lag"],
            data["Spearman_rho"],
            marker="s",
            linewidth=2,
            color="#C94C4C",
            label="Spearman",
        )
        axis.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
        axis.set_xticks(range(MAX_CORRELATION_LAG + 1))
        axis.set_xlabel("滞后天数 k")
        axis.set_title(f"{metric}与未来新注册数")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        axis.legend(frameon=False)

    axes[0].set_ylabel("相关系数")
    figure.suptitle(
        "SEM 投放指标与未来第 k 天新注册数的滞后相关",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.01,
        "lag=k：第 t 天投放指标对应第 t+k 天新注册数",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(CORRELATION_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_model_data(
    daily: pd.DataFrame,
    max_lag: int,
) -> tuple[pd.Series, pd.DataFrame, list[str], dict[str, str]]:
    """构造分布滞后模型数据。

    对当天注册数 R_t，点击量 ``shift(k)`` 表示 C_{t-k}。为便于解释和
    改善数值尺度，点击量除以 1000，系数表示每增加 1000 次点击对应的
    新注册数关联变化。
    """
    model_data = daily[["日期", "总点击量", "新注册数"]].copy()
    lag_columns: list[str] = []
    variable_types: dict[str, str] = {"const": "截距"}

    for lag in range(max_lag + 1):
        column = f"点击量_lag{lag}"
        model_data[column] = model_data["总点击量"].shift(lag) / 1000.0
        lag_columns.append(column)
        variable_types[column] = "点击滞后项"

    weekday = pd.Categorical(
        model_data["日期"].dt.dayofweek,
        categories=range(7),
        ordered=True,
    )
    month = pd.Categorical(
        model_data["日期"].dt.month,
        categories=range(1, 13),
        ordered=True,
    )
    weekday_dummies = pd.get_dummies(
        weekday,
        prefix="星期",
        drop_first=True,
        dtype=float,
    )
    month_dummies = pd.get_dummies(
        month,
        prefix="月份",
        drop_first=True,
        dtype=float,
    )
    weekday_dummies.columns = [
        f"星期_{WEEKDAY_NAMES[int(column.split('_')[1])]}"
        for column in weekday_dummies.columns
    ]
    month_dummies.columns = [
        f"月份_{int(column.split('_')[1])}月" for column in month_dummies.columns
    ]
    variable_types.update({column: "星期控制变量" for column in weekday_dummies.columns})
    variable_types.update({column: "月份控制变量" for column in month_dummies.columns})

    predictors = pd.concat(
        [model_data[lag_columns], weekday_dummies, month_dummies],
        axis=1,
    )
    combined = pd.concat([model_data[["新注册数"]], predictors], axis=1).dropna()
    response = combined["新注册数"].astype(float)
    design = sm.add_constant(combined.drop(columns="新注册数"), has_constant="add")
    return response, design, lag_columns, variable_types


def calculate_vif(design: pd.DataFrame) -> dict[str, float]:
    """计算每个非截距解释变量相对于完整设计矩阵的 VIF。"""
    values = design.to_numpy(dtype=float)
    vif: dict[str, float] = {"const": np.nan}
    for index, column in enumerate(design.columns):
        if column == "const":
            continue
        vif[column] = float(variance_inflation_factor(values, index))
    return vif


def fit_distributed_lag_model(
    daily: pd.DataFrame,
    max_lag: int,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """拟合一个带星期、月份控制变量的分布滞后 OLS 模型。"""
    response, design, lag_columns, variable_types = build_model_data(daily, max_lag)
    fitted = sm.OLS(response, design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_MAX_LAG},
        use_t=True,
    )
    vif = calculate_vif(design)
    confidence_intervals = fitted.conf_int(alpha=ALPHA)

    coefficient_rows: list[dict[str, object]] = []
    for variable in fitted.params.index:
        coefficient_rows.append(
            {
                "模型": model_name,
                "最大点击滞后天数": max_lag,
                "变量": variable,
                "变量类型": variable_types[variable],
                "系数单位": (
                    "每增加1000次点击对应的新注册数变化"
                    if variable in lag_columns
                    else "相对基准组的注册数差异"
                    if variable != "const"
                    else "新注册数"
                ),
                "系数": float(fitted.params[variable]),
                "HAC标准误": float(fitted.bse[variable]),
                "t值": float(fitted.tvalues[variable]),
                "p值": float(fitted.pvalues[variable]),
                "95%置信区间下限": float(confidence_intervals.loc[variable, 0]),
                "95%置信区间上限": float(confidence_intervals.loc[variable, 1]),
                "VIF": vif[variable],
                "样本量": int(fitted.nobs),
                "R2": float(fitted.rsquared),
                "调整R2": float(fitted.rsquared_adj),
            }
        )

    parameter_names = list(fitted.params.index)
    joint_restriction = np.zeros((len(lag_columns), len(parameter_names)))
    cumulative_restriction = np.zeros(len(parameter_names))
    for row, lag_column in enumerate(lag_columns):
        column_index = parameter_names.index(lag_column)
        joint_restriction[row, column_index] = 1
        cumulative_restriction[column_index] = 1

    joint_test = fitted.wald_test(
        joint_restriction,
        use_f=False,
        scalar=True,
    )
    cumulative_test = fitted.t_test(cumulative_restriction)

    click_vif = [vif[column] for column in lag_columns]
    summary_row = {
        "模型": model_name,
        "最大点击滞后天数": max_lag,
        "因变量": "当天新注册数",
        "主要解释变量": f"当天及前1至{max_lag}天总点击量",
        "点击系数单位": "每1000次点击",
        "控制变量": "星期虚拟变量（周一为基准）；月份虚拟变量（1月为基准）",
        "样本量": int(fitted.nobs),
        "R2": float(fitted.rsquared),
        "调整R2": float(fitted.rsquared_adj),
        "协方差估计": f"Newey-West HAC(maxlags={HAC_MAX_LAG})",
        "AIC": float(fitted.aic),
        "BIC": float(fitted.bic),
        "Durbin_Watson": float(durbin_watson(fitted.resid)),
        "点击滞后项联合Wald统计量": float(np.asarray(joint_test.statistic).squeeze()),
        "点击滞后项联合检验自由度": len(lag_columns),
        "点击滞后项联合p值": float(joint_test.pvalue),
        "点击滞后项累计系数": float(np.asarray(cumulative_test.effect).squeeze()),
        "累计系数HAC标准误": float(np.asarray(cumulative_test.sd).squeeze()),
        "点击滞后项累计p值": float(cumulative_test.pvalue),
        "点击滞后项最大VIF": max(click_vif),
        "点击滞后项平均VIF": float(np.mean(click_vif)),
    }
    return pd.DataFrame(coefficient_rows), summary_row


def format_p_value(value: float) -> str:
    """格式化终端摘要中的 p 值。"""
    return f"{value:.3e}" if value < 0.001 else f"{value:.4f}"


def print_summary(
    correlations: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    """只打印滞后相关和两个分布滞后模型的简洁摘要。"""
    print("滞后相关摘要：lag=k 表示第t天投放对应第t+k天注册")
    for metric in AD_METRICS:
        metric_rows = correlations.loc[correlations["投放指标"] == metric]
        pearson_row = metric_rows.loc[metric_rows["Pearson_r"].abs().idxmax()]
        spearman_row = metric_rows.loc[metric_rows["Spearman_rho"].abs().idxmax()]
        print(
            f"{metric}: |Pearson|最大在lag={int(pearson_row['lag'])}, "
            f"r={pearson_row['Pearson_r']:.4f}, "
            f"p={format_p_value(pearson_row['Pearson_p值'])}; "
            f"|Spearman|最大在lag={int(spearman_row['lag'])}, "
            f"rho={spearman_row['Spearman_rho']:.4f}, "
            f"p={format_p_value(spearman_row['Spearman_p值'])}。"
        )

    print("\n分布滞后模型摘要")
    for row in model_summary.itertuples(index=False):
        print(
            f"{row.模型}: n={row.样本量}, 调整R2={row.调整R2:.4f}, "
            f"点击滞后项联合Wald={row.点击滞后项联合Wald统计量:.4f}, "
            f"联合p={format_p_value(row.点击滞后项联合p值)}, "
            f"累计系数={row.点击滞后项累计系数:.4f}, "
            f"累计p={format_p_value(row.点击滞后项累计p值)}, "
            f"最大VIF={row.点击滞后项最大VIF:.2f}。"
        )
    print("注：结果描述的是控制星期和月份后的时间关联结构，不构成因果归因。")


def main() -> None:
    daily = load_daily_summary()
    correlations = calculate_lag_correlations(daily)
    plot_lag_correlations(correlations)

    primary_coefficients, primary_summary = fit_distributed_lag_model(
        daily,
        PRIMARY_MODEL_LAG,
        "主模型_lag0至3",
    )
    sensitivity_coefficients, sensitivity_summary = fit_distributed_lag_model(
        daily,
        SENSITIVITY_MODEL_LAG,
        "敏感性模型_lag0至7",
    )
    coefficients = pd.concat(
        [primary_coefficients, sensitivity_coefficients],
        ignore_index=True,
    )
    model_summary = pd.DataFrame([primary_summary, sensitivity_summary])

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(
        CORRELATION_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    coefficients.to_csv(
        COEFFICIENT_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    model_summary.to_csv(
        MODEL_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )

    print_summary(correlations, model_summary)


if __name__ == "__main__":
    main()
