"""问题二：关键词投放实例的成本—效益五分类诊断模型。

分类对象保持 Sheet3 原始粒度：(方案ID, 推广单元ID, 关键词)，不跨单元合并。
主模型采用“CRITIC客观赋权 + 成本/效益分别进行一维KMeans(K=2)”；同时
给出等权KMeans和CRITIC得分中位数阈值两组稳健性对照。

本脚本只生成诊断表和图，不读取或修改最终 result2.xlsx。
结构性“/”保持为不可定义，不进行均值填补。
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.set_loglevel("error")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CANDIDATES = [
    PROJECT_ROOT / "Attachment1.xlsx",
    PROJECT_ROOT / "附件1.xlsx",
    PROJECT_ROOT / "data" / "raw" / "Attachment1.xlsx",
    PROJECT_ROOT / "data" / "raw" / "Attachment" / "Attachment1.xlsx",
]
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"

SCORES_CSV = TABLE_DIR / "q2_keyword_scores.csv"
WEIGHTS_CSV = TABLE_DIR / "q2_critic_weights.csv"
THRESHOLDS_CSV = TABLE_DIR / "q2_thresholds.csv"
CATEGORY_COUNTS_CSV = TABLE_DIR / "q2_category_counts.csv"
STABILITY_CSV = TABLE_DIR / "q2_stability_comparison.csv"
ANOMALIES_CSV = TABLE_DIR / "q2_anomalies.csv"
DATA_QUALITY_CSV = TABLE_DIR / "q2_data_quality_summary.csv"

SCATTER_FIGURE = FIGURE_DIR / "q2_cost_benefit_scatter.png"
COST_CLUSTER_FIGURE = FIGURE_DIR / "q2_cost_score_clusters.png"
BENEFIT_CLUSTER_FIGURE = FIGURE_DIR / "q2_benefit_score_clusters.png"

RANDOM_STATE = 2026
N_INIT = 200
BOUNDARY_QUANTILE = 0.05

REQUIRED_COLUMNS = [
    "序号",
    "关键词",
    "方案ID",
    "推广单元ID",
    "消费额",
    "点击量",
    "浏览量",
    "跳出率",
    "平均访问时长",
]
ID_COLUMNS = ["方案ID", "推广单元ID", "关键词"]
RAW_NUMERIC_COLUMNS = ["消费额", "点击量", "浏览量"]
CATEGORY_ORDER = ["黄金词", "重点词", "潜力词", "问题词", "无效词"]
NORMAL_CATEGORY_ORDER = ["黄金词", "重点词", "潜力词", "问题词"]

COST_INDICATORS = ["消费额", "CPC"]
BENEFIT_INDICATORS = ["点击量", "浏览深度", "非跳出率", "平均访问时长_秒"]
NORMALIZED_COLUMNS = {
    "消费额": "消费额_标准化",
    "CPC": "CPC_标准化",
    "点击量": "点击量_标准化",
    "浏览深度": "浏览深度_标准化",
    "非跳出率": "非跳出率_标准化",
    "平均访问时长_秒": "平均访问时长_标准化",
}
PREPROCESS_DESCRIPTIONS = {
    "消费额": "log1p后百分位秩标准化至[0,1]",
    "CPC": "log1p后百分位秩标准化至[0,1]",
    "点击量": "log1p后百分位秩标准化至[0,1]",
    "浏览深度": "百分位秩标准化至[0,1]",
    "非跳出率": "百分位秩标准化至[0,1]",
    "平均访问时长_秒": "百分位秩标准化至[0,1]",
}


def find_input_file() -> Path:
    """按常用位置查找附件1。"""
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in INPUT_CANDIDATES)
    raise FileNotFoundError(f"未找到 Attachment1.xlsx，已检查：\n{searched}")


def normalize_identifier(value: object) -> str:
    """将Excel编码稳定转换成字符串，避免整数编码出现.0。"""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def is_undefined_marker(value: object) -> bool:
    """识别结构性的斜杠和空值。"""
    if pd.isna(value):
        return True
    return str(value).strip() in {"", "/"}


def parse_rate(value: object) -> float:
    """把跳出率统一转换到0--1；斜杠保持NaN。"""
    if is_undefined_marker(value):
        return np.nan
    text = str(value).strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    parsed = float(value)
    return parsed / 100.0 if 1 < parsed <= 100 else parsed


def parse_duration_seconds(value: object) -> float:
    """把平均访问时长统一为秒；斜杠保持NaN。"""
    if is_undefined_marker(value):
        return np.nan
    if isinstance(value, time):
        return float(value.hour * 3600 + value.minute * 60 + value.second)
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if 0 < abs(numeric) < 1:
            return numeric * 86400.0
        return numeric
    return float(pd.to_timedelta(str(value).strip(), errors="raise").total_seconds())


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """安全逐项除法，分母为0时返回NaN。"""
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = numerator.notna() & denominator.notna() & denominator.ne(0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def append_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> None:
    """向异常原因字符串追加标记，不删除原记录。"""
    existing = reasons.loc[mask]
    reasons.loc[mask] = np.where(existing.eq(""), reason, existing + "；" + reason)


def load_and_clean() -> pd.DataFrame:
    """读取Sheet3并生成可追溯的清洗字段、派生指标和异常标记。"""
    raw = pd.read_excel(find_input_file(), sheet_name="Sheet3", dtype=object)
    raw.columns = raw.columns.map(lambda column: str(column).strip())
    missing = sorted(set(REQUIRED_COLUMNS).difference(raw.columns))
    if missing:
        raise ValueError(f"Sheet3缺少字段：{'、'.join(missing)}")
    data = raw[REQUIRED_COLUMNS].copy()

    sequence = pd.to_numeric(data["序号"], errors="coerce")
    if sequence.isna().any() or (~np.isclose(sequence, np.round(sequence))).any():
        raise ValueError("Sheet3序号存在缺失或非整数，无法稳定追溯。")
    data["序号"] = sequence.round().astype("int64")

    for column in ID_COLUMNS:
        data[column] = data[column].map(normalize_identifier).astype("string")
        if data[column].eq("").any():
            raise ValueError(f"Sheet3字段{column}存在空编码。")
    for column in RAW_NUMERIC_COLUMNS:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].notna() & converted.isna()
        if invalid.any():
            examples = data.loc[invalid, column].astype(str).unique()[:5]
            raise ValueError(f"Sheet3字段{column}存在非数值：{examples.tolist()}")
        data[column] = converted.astype(float)

    data["跳出率_原值"] = data["跳出率"].astype("string")
    data["平均访问时长_原值"] = data["平均访问时长"].astype("string")
    data["跳出率不可定义"] = data["跳出率"].map(is_undefined_marker)
    data["平均访问时长不可定义"] = data["平均访问时长"].map(is_undefined_marker)
    data["跳出率"] = data["跳出率"].map(parse_rate).astype(float)
    data["平均访问时长_秒"] = data["平均访问时长"].map(
        parse_duration_seconds
    ).astype(float)
    data = data.drop(columns="平均访问时长")

    data["CPC"] = safe_divide(data["消费额"], data["点击量"])
    data["浏览深度"] = safe_divide(data["浏览量"], data["点击量"])
    data["非跳出率"] = 1.0 - data["跳出率"]
    data["zero_effect"] = (
        data["消费额"].eq(0) & data["点击量"].eq(0) & data["浏览量"].eq(0)
    )

    reasons = pd.Series("", index=data.index, dtype="string")
    append_reason(reasons, data["消费额"].gt(0) & data["点击量"].eq(0), "消费>0但点击=0")
    append_reason(reasons, data["点击量"].eq(0) & data["浏览量"].gt(0), "点击=0但浏览>0")
    append_reason(
        reasons,
        data["点击量"].gt(0) & data["跳出率"].isna(),
        "点击>0但跳出率缺失",
    )
    append_reason(
        reasons,
        data["点击量"].gt(0) & data["平均访问时长_秒"].isna(),
        "点击>0但平均访问时长缺失",
    )
    append_reason(reasons, data[RAW_NUMERIC_COLUMNS].lt(0).any(axis=1), "投入或流量字段为负")
    append_reason(
        reasons,
        data["跳出率"].notna() & ~data["跳出率"].between(0, 1),
        "跳出率超出[0,1]",
    )
    append_reason(
        reasons,
        data["平均访问时长_秒"].notna() & data["平均访问时长_秒"].lt(0),
        "平均访问时长为负",
    )
    append_reason(reasons, data["序号"].duplicated(keep=False), "原始序号重复")
    append_reason(reasons, data.duplicated(ID_COLUMNS, keep=False), "投放实例三元组重复")
    data["anomaly_reasons"] = reasons
    data["anomaly_flag"] = reasons.ne("")

    required_scores = ["消费额", "CPC", "点击量", "浏览深度", "非跳出率", "平均访问时长_秒"]
    finite_and_defined = data[required_scores].notna().all(axis=1)
    valid_ranges = (
        data["消费额"].ge(0)
        & data["CPC"].ge(0)
        & data["点击量"].gt(0)
        & data["浏览深度"].ge(0)
        & data["非跳出率"].between(0, 1)
        & data["平均访问时长_秒"].ge(0)
    )
    data["常规评分可用"] = ~data["zero_effect"] & finite_and_defined & valid_ranges
    return data.sort_values("序号").reset_index(drop=True)


def percentile_rank_01(values: pd.Series) -> pd.Series:
    """使用稳健百分位秩将有效序列映射到闭区间[0,1]。"""
    if values.isna().any():
        raise ValueError("百分位秩输入包含缺失值。")
    if len(values) == 1:
        return pd.Series(0.5, index=values.index, dtype=float)
    ranks = values.rank(method="average")
    return (ranks - 1.0) / (len(values) - 1.0)


def build_normalized_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """对1337条常规可评分记录进行log1p与百分位秩标准化。"""
    scoreable = data.loc[data["常规评分可用"]].copy()
    transformations = {
        "消费额": np.log1p(scoreable["消费额"]),
        "CPC": np.log1p(scoreable["CPC"]),
        "点击量": np.log1p(scoreable["点击量"]),
        "浏览深度": scoreable["浏览深度"],
        "非跳出率": scoreable["非跳出率"],
        "平均访问时长_秒": scoreable["平均访问时长_秒"],
    }
    normalized = pd.DataFrame(index=scoreable.index)
    for indicator, values in transformations.items():
        if (~np.isfinite(values)).any() or values.lt(0).any():
            raise ValueError(f"指标{indicator}存在无法进行稳健标准化的值。")
        normalized[NORMALIZED_COLUMNS[indicator]] = percentile_rank_01(values)
    return normalized


def critic_weights(
    normalized: pd.DataFrame,
    indicators: list[str],
    group_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """明确实现CRITIC：C_j=sigma_j*sum_k(1-r_jk)，再归一化。"""
    columns = [NORMALIZED_COLUMNS[indicator] for indicator in indicators]
    matrix = normalized[columns]
    standard_deviations = matrix.std(axis=0, ddof=1)
    correlations = matrix.corr(method="pearson")
    if standard_deviations.le(0).any() or correlations.isna().any().any():
        raise ValueError(f"{group_name}指标存在常数列或不可定义相关系数。")
    conflict = (1.0 - correlations).sum(axis=1)
    information = standard_deviations * conflict
    if information.sum() <= 0:
        raise ValueError(f"{group_name}指标CRITIC信息量之和不为正。")
    weights_by_column = information / information.sum()

    rows: list[dict[str, object]] = []
    weights: dict[str, float] = {}
    reverse_names = {NORMALIZED_COLUMNS[name]: name for name in indicators}
    for indicator in indicators:
        column = NORMALIZED_COLUMNS[indicator]
        weights[indicator] = float(weights_by_column[column])
        correlation_text = "；".join(
            f"{reverse_names[other]}:{correlations.loc[column, other]:.6f}"
            for other in columns
            if other != column
        )
        rows.append(
            {
                "指标组": group_name,
                "指标": indicator,
                "方向": "越大表示成本越高" if group_name == "成本" else "越大表示效益越高",
                "预处理": PREPROCESS_DESCRIPTIONS[indicator],
                "标准差_sigma": float(standard_deviations[column]),
                "冲突性_sum_1_minus_r": float(conflict[column]),
                "与其他指标相关系数": correlation_text,
                "CRITIC信息量_Cj": float(information[column]),
                "CRITIC权重": float(weights_by_column[column]),
                "CRITIC公式": "C_j=sigma_j*sum_k(1-r_jk); w_j=C_j/sum(C_j)",
            }
        )
    return weights, pd.DataFrame(rows)


def weighted_score(
    normalized: pd.DataFrame,
    indicators: list[str],
    weights: dict[str, float],
) -> pd.Series:
    """计算同一指标组的加权和得分。"""
    score = pd.Series(0.0, index=normalized.index, dtype=float)
    for indicator in indicators:
        score = score + normalized[NORMALIZED_COLUMNS[indicator]] * weights[indicator]
    return score


def fit_1d_kmeans(
    values: pd.Series,
    random_state: int = RANDOM_STATE,
    n_init: int = N_INIT,
    max_iter: int = 300,
    tolerance: float = 1e-12,
) -> tuple[pd.Series, np.ndarray, float, float]:
    """可复现的一维KMeans(K=2)，使用k-means++随机初始化与多次重启。"""
    array = values.to_numpy(dtype=float)
    if len(array) < 2 or np.unique(array).size < 2 or (~np.isfinite(array)).any():
        raise ValueError("一维KMeans需要至少两个不同的有限得分。")
    rng = np.random.default_rng(random_state)
    best_inertia = np.inf
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None

    for _ in range(n_init):
        first_center = float(array[rng.integers(0, len(array))])
        squared_distance = (array - first_center) ** 2
        if squared_distance.sum() == 0:
            continue
        second_center = float(rng.choice(array, p=squared_distance / squared_distance.sum()))
        centers = np.asarray([first_center, second_center], dtype=float)

        for _ in range(max_iter):
            distances = np.abs(array[:, None] - centers[None, :])
            labels = np.argmin(distances, axis=1)
            if np.unique(labels).size < 2:
                farthest = int(np.argmax(np.min(distances, axis=1)))
                labels[farthest] = 1 - labels[farthest]
            new_centers = np.asarray(
                [array[labels == cluster].mean() for cluster in range(2)],
                dtype=float,
            )
            if np.max(np.abs(new_centers - centers)) <= tolerance:
                centers = new_centers
                break
            centers = new_centers

        labels = np.argmin(np.abs(array[:, None] - centers[None, :]), axis=1)
        inertia = float(np.sum((array - centers[labels]) ** 2))
        sorted_centers = np.sort(centers)
        is_better_tie = (
            np.isclose(inertia, best_inertia)
            and best_centers is not None
            and tuple(sorted_centers) < tuple(np.sort(best_centers))
        )
        if inertia < best_inertia - tolerance or is_better_tie:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    if best_labels is None or best_centers is None:
        raise RuntimeError("一维KMeans拟合失败。")
    low_cluster = int(np.argmin(best_centers))
    high_cluster = 1 - low_cluster
    human_labels = pd.Series(
        np.where(best_labels == low_cluster, "低", "高"),
        index=values.index,
        dtype="string",
    )
    low_center = float(best_centers[low_cluster])
    high_center = float(best_centers[high_cluster])
    threshold = (low_center + high_center) / 2.0
    return human_labels, np.asarray([low_center, high_center]), threshold, best_inertia


def map_category(cost_level: pd.Series, benefit_level: pd.Series) -> pd.Series:
    """把成本高低与效益高低映射为四种有效词类别。"""
    conditions = [
        cost_level.eq("低") & benefit_level.eq("高"),
        cost_level.eq("高") & benefit_level.eq("高"),
        cost_level.eq("低") & benefit_level.eq("低"),
        cost_level.eq("高") & benefit_level.eq("低"),
    ]
    values = ["黄金词", "重点词", "潜力词", "问题词"]
    return pd.Series(np.select(conditions, values, default="待人工复核"), index=cost_level.index)


def apply_special_categories(data: pd.DataFrame, category_column: str) -> None:
    """给zero_effect及两条0点击有浏览异常记录赋予透明的特殊分类。"""
    data.loc[data["zero_effect"], category_column] = "无效词"
    special_potential = (
        ~data["zero_effect"]
        & data["消费额"].eq(0)
        & data["点击量"].eq(0)
        & data["浏览量"].gt(0)
    )
    data.loc[special_potential, category_column] = "潜力词"


def build_threshold_row(
    model_name: str,
    dimension: str,
    centers: np.ndarray | None,
    threshold: float,
    inertia: float | None,
    boundary_distance: float | None = None,
) -> dict[str, object]:
    """构造统一阈值记录。"""
    return {
        "模型方案": model_name,
        "维度": dimension,
        "低簇中心": np.nan if centers is None else float(centers[0]),
        "高簇中心": np.nan if centers is None else float(centers[1]),
        "分类阈值": threshold,
        "阈值定义": "两个KMeans簇中心中点" if centers is not None else "得分中位数",
        "KMeans_K": 2 if centers is not None else np.nan,
        "random_state": RANDOM_STATE if centers is not None else np.nan,
        "n_init": N_INIT if centers is not None else np.nan,
        "簇内平方和": inertia,
        "boundary距离分位数": BOUNDARY_QUANTILE if boundary_distance is not None else np.nan,
        "boundary距离阈值": boundary_distance,
    }


def build_category_counts(data: pd.DataFrame) -> pd.DataFrame:
    """汇总主模型五类记录数量和比例。"""
    counts = data["最终类别"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    result = counts.rename_axis("最终类别").reset_index(name="数量")
    result["比例"] = result["数量"] / len(data)
    result["总记录数"] = len(data)
    return result


def build_stability_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """输出一致率、人数变化和常规评分记录的混淆矩阵。"""
    rows: list[dict[str, object]] = []
    scoreable = data["常规评分可用"]
    comparisons = [
        ("A_等权重+一维KMeans", "等权最终类别"),
        ("B_CRITIC权重+中位数阈值", "中位数最终类别"),
    ]
    for model_name, control_column in comparisons:
        for scope_name, mask in [("常规可评分记录", scoreable), ("全部记录", pd.Series(True, index=data.index))]:
            main = data.loc[mask, "最终类别"]
            control = data.loc[mask, control_column]
            same = main.eq(control)
            rows.append(
                {
                    "对照方案": model_name,
                    "记录类型": "一致率汇总",
                    "比较范围": scope_name,
                    "比较样本数": len(main),
                    "一致数量": int(same.sum()),
                    "完全一致比例": float(same.mean()),
                    "主模型类别": "",
                    "对照模型类别": "",
                    "主模型数量": np.nan,
                    "对照模型数量": np.nan,
                    "人数变化_对照减主模型": np.nan,
                    "交叉数量": np.nan,
                    "说明": "主模型与对照模型逐条比较",
                }
            )

        for category in CATEGORY_ORDER:
            main_count = int(data["最终类别"].eq(category).sum())
            control_count = int(data[control_column].eq(category).sum())
            rows.append(
                {
                    "对照方案": model_name,
                    "记录类型": "类别人数变化",
                    "比较范围": "全部记录",
                    "比较样本数": len(data),
                    "一致数量": np.nan,
                    "完全一致比例": np.nan,
                    "主模型类别": category,
                    "对照模型类别": category,
                    "主模型数量": main_count,
                    "对照模型数量": control_count,
                    "人数变化_对照减主模型": control_count - main_count,
                    "交叉数量": np.nan,
                    "说明": "无效词和异常特殊记录使用相同固定规则",
                }
            )

        confusion = pd.crosstab(
            data.loc[scoreable, "最终类别"],
            data.loc[scoreable, control_column],
        ).reindex(index=NORMAL_CATEGORY_ORDER, columns=NORMAL_CATEGORY_ORDER, fill_value=0)
        for main_category in NORMAL_CATEGORY_ORDER:
            for control_category in NORMAL_CATEGORY_ORDER:
                rows.append(
                    {
                        "对照方案": model_name,
                        "记录类型": "混淆矩阵",
                        "比较范围": "常规可评分记录",
                        "比较样本数": int(scoreable.sum()),
                        "一致数量": np.nan,
                        "完全一致比例": np.nan,
                        "主模型类别": main_category,
                        "对照模型类别": control_category,
                        "主模型数量": np.nan,
                        "对照模型数量": np.nan,
                        "人数变化_对照减主模型": np.nan,
                        "交叉数量": int(confusion.loc[main_category, control_category]),
                        "说明": "行=CRITIC+KMeans主模型，列=对照模型",
                    }
                )

    anomaly_count = int(data["anomaly_flag"].sum())
    rows.append(
        {
            "对照方案": "异常记录隔离检查",
            "记录类型": "拟合影响检查",
            "比较范围": "CRITIC与KMeans拟合样本",
            "比较样本数": int(scoreable.sum()),
            "一致数量": int(scoreable.sum()),
            "完全一致比例": 1.0,
            "主模型类别": "",
            "对照模型类别": "",
            "主模型数量": np.nan,
            "对照模型数量": np.nan,
            "人数变化_对照减主模型": np.nan,
            "交叉数量": np.nan,
            "说明": f"{anomaly_count}条异常记录未参与权重和阈值拟合，附加特殊分类不改变1337条常规记录",
        }
    )
    return pd.DataFrame(rows)


def build_data_quality_summary(data: pd.DataFrame) -> pd.DataFrame:
    """汇总结构性缺失与异常，不在终端展开原始数据。"""
    items = [
        ("Sheet3总记录数", pd.Series(True, index=data.index), ""),
        ("zero_effect记录", data["zero_effect"], "消费、点击、浏览均为0"),
        ("消费>0但点击=0", data["消费额"].gt(0) & data["点击量"].eq(0), "CPC不可定义"),
        ("点击=0但浏览>0", data["点击量"].eq(0) & data["浏览量"].gt(0), "不归无效词，暂归潜力词"),
        ("点击>0但跳出率缺失", data["点击量"].gt(0) & data["跳出率"].isna(), "不填补"),
        (
            "点击>0但平均访问时长缺失",
            data["点击量"].gt(0) & data["平均访问时长_秒"].isna(),
            "不填补",
        ),
        ("跳出率结构性不可定义", data["跳出率不可定义"], "原值为/或空，不填补"),
        ("平均访问时长结构性不可定义", data["平均访问时长不可定义"], "原值为/或空，不填补"),
        ("异常记录", data["anomaly_flag"], "详见q2_anomalies.csv"),
        ("常规评分可用记录", data["常规评分可用"], "参与CRITIC与KMeans拟合"),
    ]
    return pd.DataFrame(
        [
            {
                "检查项": name,
                "数量": int(mask.sum()),
                "占总记录比例": float(mask.mean()),
                "处理方式": note,
            }
            for name, mask, note in items
        ]
    )


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


def save_cost_benefit_scatter(
    data: pd.DataFrame,
    cost_threshold: float,
    benefit_threshold: float,
) -> None:
    """绘制主模型CostScore—BenefitScore四象限图。"""
    scoreable = data.loc[data["常规评分可用"]]
    colors = {
        "黄金词": "#E6B800",
        "重点词": "#D95F5F",
        "潜力词": "#55A868",
        "问题词": "#8172B3",
    }
    figure, axis = plt.subplots(figsize=(10, 8))
    for category in NORMAL_CATEGORY_ORDER:
        group = scoreable.loc[scoreable["最终类别"] == category]
        axis.scatter(
            group["CostScore"],
            group["BenefitScore"],
            s=20,
            alpha=0.55,
            color=colors[category],
            edgecolors="none",
            label=f"{category}（n={len(group)}）",
        )
    boundary = scoreable.loc[scoreable["boundary_sensitive"]]
    axis.scatter(
        boundary["CostScore"],
        boundary["BenefitScore"],
        s=42,
        facecolors="none",
        edgecolors="#222222",
        linewidths=0.65,
        alpha=0.65,
        label=f"边界敏感（n={len(boundary)}）",
    )
    axis.axvline(cost_threshold, color="#333333", linestyle="--", linewidth=1.3)
    axis.axhline(benefit_threshold, color="#333333", linestyle="--", linewidth=1.3)
    axis.text(0.02, 0.98, "黄金词\n低成本·高效益", transform=axis.transAxes, va="top", fontsize=11)
    axis.text(0.98, 0.98, "重点词\n高成本·高效益", transform=axis.transAxes, va="top", ha="right", fontsize=11)
    axis.text(0.02, 0.02, "潜力词\n低成本·低效益", transform=axis.transAxes, va="bottom", fontsize=11)
    axis.text(0.98, 0.02, "问题词\n高成本·低效益", transform=axis.transAxes, va="bottom", ha="right", fontsize=11)
    axis.set_xlabel("CostScore（越右成本越高）")
    axis.set_ylabel("BenefitScore（越上效益越高）")
    axis.set_title(
        "关键词投放实例成本—效益分类\n"
        f"无效词{int(data['zero_effect'].sum())}条和异常特殊记录"
        f"{int((~data['zero_effect'] & ~data['常规评分可用']).sum())}条未绘入散点"
    )
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)
    axis.grid(linestyle="--", linewidth=0.6, alpha=0.3)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.tight_layout()
    figure.savefig(SCATTER_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_cluster_distribution(
    data: pd.DataFrame,
    score_column: str,
    level_column: str,
    centers: np.ndarray,
    threshold: float,
    title: str,
    filename: Path,
) -> None:
    """绘制一维得分的低/高簇分布及中心、阈值。"""
    scoreable = data.loc[data["常规评分可用"]]
    figure, axis = plt.subplots(figsize=(9, 6))
    for level, color in [("低", "#55A868"), ("高", "#D95F5F")]:
        values = scoreable.loc[scoreable[level_column] == level, score_column]
        axis.hist(values, bins=28, alpha=0.58, color=color, label=f"{level}簇（n={len(values)}）")
    axis.axvline(centers[0], color="#2A7F45", linestyle=":", linewidth=2, label=f"低簇中心={centers[0]:.3f}")
    axis.axvline(centers[1], color="#A63D40", linestyle=":", linewidth=2, label=f"高簇中心={centers[1]:.3f}")
    axis.axvline(threshold, color="#222222", linestyle="--", linewidth=1.6, label=f"中心中点阈值={threshold:.3f}")
    axis.set_xlabel(score_column)
    axis.set_ylabel("投放实例数")
    axis.set_title(title)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """统一保存UTF-8 BOM CSV。"""
    dataframe.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def print_final_summary(
    data: pd.DataFrame,
    weights: pd.DataFrame,
    thresholds: pd.DataFrame,
    category_counts: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    """按要求只打印十类核心诊断信息。"""
    cost_weights = weights.loc[weights["指标组"] == "成本", ["指标", "CRITIC权重"]]
    benefit_weights = weights.loc[weights["指标组"] == "效益", ["指标", "CRITIC权重"]]
    main_thresholds = thresholds.loc[thresholds["模型方案"] == "主模型_CRITIC+一维KMeans"]
    stability_summary = stability.loc[
        stability["对照方案"].isin(["A_等权重+一维KMeans", "B_CRITIC权重+中位数阈值"])
        & stability["记录类型"].eq("一致率汇总")
        & stability["比较范围"].eq("常规可评分记录"),
        ["对照方案", "比较样本数", "一致数量", "完全一致比例"],
    ]

    print(f"1. 总记录数：{len(data)}")
    print(f"2. 无效词数量：{int(data['zero_effect'].sum())}")
    print(f"3. 异常记录数量：{int(data['anomaly_flag'].sum())}；异常不参与CRITIC和KMeans，暂归潜力词。")
    print("4. CRITIC成本权重")
    print(cost_weights.to_string(index=False))
    print("5. CRITIC效益权重")
    print(benefit_weights.to_string(index=False))
    cost_row = main_thresholds.loc[main_thresholds["维度"] == "成本"].iloc[0]
    benefit_row = main_thresholds.loc[main_thresholds["维度"] == "效益"].iloc[0]
    print(
        f"6. 成本KMeans中心与阈值：低中心={cost_row['低簇中心']:.6f}，"
        f"高中心={cost_row['高簇中心']:.6f}，阈值={cost_row['分类阈值']:.6f}"
    )
    print(
        f"7. 效益KMeans中心与阈值：低中心={benefit_row['低簇中心']:.6f}，"
        f"高中心={benefit_row['高簇中心']:.6f}，阈值={benefit_row['分类阈值']:.6f}"
    )
    print("8. 五类关键词数量和比例")
    print(category_counts[["最终类别", "数量", "比例"]].to_string(index=False))
    print("9. 两种稳健性方案与主分类的一致率（常规可评分记录）")
    print(stability_summary.to_string(index=False))
    print(f"10. boundary_sensitive数量：{int(data['boundary_sensitive'].sum())}")


def main() -> None:
    data = load_and_clean()
    normalized = build_normalized_indicators(data)
    for column in normalized.columns:
        data[column] = np.nan
        data.loc[normalized.index, column] = normalized[column]

    cost_weights, cost_diagnostics = critic_weights(normalized, COST_INDICATORS, "成本")
    benefit_weights, benefit_diagnostics = critic_weights(
        normalized, BENEFIT_INDICATORS, "效益"
    )
    critic_diagnostics = pd.concat(
        [cost_diagnostics, benefit_diagnostics], ignore_index=True
    )

    scoreable_index = normalized.index
    data["CostScore"] = np.nan
    data["BenefitScore"] = np.nan
    data.loc[scoreable_index, "CostScore"] = weighted_score(
        normalized, COST_INDICATORS, cost_weights
    )
    data.loc[scoreable_index, "BenefitScore"] = weighted_score(
        normalized, BENEFIT_INDICATORS, benefit_weights
    )

    cost_levels, cost_centers, cost_threshold, cost_inertia = fit_1d_kmeans(
        data.loc[scoreable_index, "CostScore"]
    )
    benefit_levels, benefit_centers, benefit_threshold, benefit_inertia = fit_1d_kmeans(
        data.loc[scoreable_index, "BenefitScore"]
    )
    data["成本高低"] = pd.Series(pd.NA, index=data.index, dtype="string")
    data["效益高低"] = pd.Series(pd.NA, index=data.index, dtype="string")
    data.loc[scoreable_index, "成本高低"] = cost_levels
    data.loc[scoreable_index, "效益高低"] = benefit_levels
    data["最终类别"] = pd.Series("待人工复核", index=data.index, dtype="string")
    data.loc[scoreable_index, "最终类别"] = map_category(cost_levels, benefit_levels)
    apply_special_categories(data, "最终类别")

    # 边界敏感记录：分别取距离成本/效益阈值最近的5%，两者取并集。
    data["成本阈值距离"] = np.nan
    data["效益阈值距离"] = np.nan
    data.loc[scoreable_index, "成本阈值距离"] = (
        data.loc[scoreable_index, "CostScore"] - cost_threshold
    ).abs()
    data.loc[scoreable_index, "效益阈值距离"] = (
        data.loc[scoreable_index, "BenefitScore"] - benefit_threshold
    ).abs()
    cost_boundary_distance = float(
        data.loc[scoreable_index, "成本阈值距离"].quantile(BOUNDARY_QUANTILE)
    )
    benefit_boundary_distance = float(
        data.loc[scoreable_index, "效益阈值距离"].quantile(BOUNDARY_QUANTILE)
    )
    data["成本边界敏感"] = False
    data["效益边界敏感"] = False
    data.loc[scoreable_index, "成本边界敏感"] = (
        data.loc[scoreable_index, "成本阈值距离"] <= cost_boundary_distance
    )
    data.loc[scoreable_index, "效益边界敏感"] = (
        data.loc[scoreable_index, "效益阈值距离"] <= benefit_boundary_distance
    )
    data["boundary_sensitive"] = data["成本边界敏感"] | data["效益边界敏感"]

    # 对照A：相同标准化指标等权求和，再分别做一维KMeans。
    equal_cost_weights = {indicator: 1 / len(COST_INDICATORS) for indicator in COST_INDICATORS}
    equal_benefit_weights = {
        indicator: 1 / len(BENEFIT_INDICATORS) for indicator in BENEFIT_INDICATORS
    }
    data["等权CostScore"] = np.nan
    data["等权BenefitScore"] = np.nan
    data.loc[scoreable_index, "等权CostScore"] = weighted_score(
        normalized, COST_INDICATORS, equal_cost_weights
    )
    data.loc[scoreable_index, "等权BenefitScore"] = weighted_score(
        normalized, BENEFIT_INDICATORS, equal_benefit_weights
    )
    equal_cost_levels, equal_cost_centers, equal_cost_threshold, equal_cost_inertia = fit_1d_kmeans(
        data.loc[scoreable_index, "等权CostScore"]
    )
    equal_benefit_levels, equal_benefit_centers, equal_benefit_threshold, equal_benefit_inertia = fit_1d_kmeans(
        data.loc[scoreable_index, "等权BenefitScore"]
    )
    data["等权最终类别"] = pd.Series("待人工复核", index=data.index, dtype="string")
    data.loc[scoreable_index, "等权最终类别"] = map_category(
        equal_cost_levels, equal_benefit_levels
    )
    apply_special_categories(data, "等权最终类别")

    # 对照B：保留CRITIC得分，以各自中位数划分高低。
    median_cost_threshold = float(data.loc[scoreable_index, "CostScore"].median())
    median_benefit_threshold = float(data.loc[scoreable_index, "BenefitScore"].median())
    median_cost_levels = pd.Series(
        np.where(data.loc[scoreable_index, "CostScore"] <= median_cost_threshold, "低", "高"),
        index=scoreable_index,
        dtype="string",
    )
    median_benefit_levels = pd.Series(
        np.where(
            data.loc[scoreable_index, "BenefitScore"] <= median_benefit_threshold,
            "低",
            "高",
        ),
        index=scoreable_index,
        dtype="string",
    )
    data["中位数最终类别"] = pd.Series("待人工复核", index=data.index, dtype="string")
    data.loc[scoreable_index, "中位数最终类别"] = map_category(
        median_cost_levels, median_benefit_levels
    )
    apply_special_categories(data, "中位数最终类别")

    unexpected = sorted(set(data["最终类别"].dropna()) - set(CATEGORY_ORDER))
    if unexpected:
        raise ValueError(f"存在无法进入正式五分类的记录：{unexpected}")

    thresholds = pd.DataFrame(
        [
            build_threshold_row(
                "主模型_CRITIC+一维KMeans",
                "成本",
                cost_centers,
                cost_threshold,
                cost_inertia,
                cost_boundary_distance,
            ),
            build_threshold_row(
                "主模型_CRITIC+一维KMeans",
                "效益",
                benefit_centers,
                benefit_threshold,
                benefit_inertia,
                benefit_boundary_distance,
            ),
            build_threshold_row(
                "A_等权重+一维KMeans",
                "成本",
                equal_cost_centers,
                equal_cost_threshold,
                equal_cost_inertia,
            ),
            build_threshold_row(
                "A_等权重+一维KMeans",
                "效益",
                equal_benefit_centers,
                equal_benefit_threshold,
                equal_benefit_inertia,
            ),
            build_threshold_row(
                "B_CRITIC权重+中位数阈值",
                "成本",
                None,
                median_cost_threshold,
                None,
            ),
            build_threshold_row(
                "B_CRITIC权重+中位数阈值",
                "效益",
                None,
                median_benefit_threshold,
                None,
            ),
        ]
    )
    category_counts = build_category_counts(data)
    stability = build_stability_comparison(data)
    quality_summary = build_data_quality_summary(data)

    special_unscoreable = ~data["zero_effect"] & ~data["常规评分可用"]
    data["分类处理说明"] = "常规CRITIC+一维KMeans"
    data.loc[data["zero_effect"], "分类处理说明"] = "结构性无成本无效益，先行归为无效词"
    data.loc[special_unscoreable, "分类处理说明"] = (
        "异常记录不参与拟合；依据0成本、0点击和少量浏览暂归潜力词"
    )
    anomaly_columns = [
        "序号",
        "关键词",
        "方案ID",
        "推广单元ID",
        "消费额",
        "点击量",
        "浏览量",
        "CPC",
        "浏览深度",
        "跳出率_原值",
        "跳出率",
        "非跳出率",
        "平均访问时长_原值",
        "平均访问时长_秒",
        "zero_effect",
        "anomaly_flag",
        "anomaly_reasons",
        "常规评分可用",
        "最终类别",
        "分类处理说明",
    ]
    anomalies = data.loc[data["anomaly_flag"], anomaly_columns].copy()

    score_columns = [
        "序号",
        "关键词",
        "方案ID",
        "推广单元ID",
        "消费额",
        "点击量",
        "浏览量",
        "CPC",
        "浏览深度",
        "跳出率",
        "非跳出率",
        "平均访问时长_秒",
        *NORMALIZED_COLUMNS.values(),
        "CostScore",
        "BenefitScore",
        "成本高低",
        "效益高低",
        "最终类别",
        "zero_effect",
        "anomaly_flag",
        "anomaly_reasons",
        "常规评分可用",
        "成本阈值距离",
        "效益阈值距离",
        "成本边界敏感",
        "效益边界敏感",
        "boundary_sensitive",
        "等权CostScore",
        "等权BenefitScore",
        "等权最终类别",
        "中位数最终类别",
        "分类处理说明",
    ]

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(data[score_columns], SCORES_CSV)
    save_csv(critic_diagnostics, WEIGHTS_CSV)
    save_csv(thresholds, THRESHOLDS_CSV)
    save_csv(category_counts, CATEGORY_COUNTS_CSV)
    save_csv(stability, STABILITY_CSV)
    save_csv(anomalies, ANOMALIES_CSV)
    save_csv(quality_summary, DATA_QUALITY_CSV)

    configure_chinese_font()
    save_cost_benefit_scatter(data, cost_threshold, benefit_threshold)
    save_cluster_distribution(
        data,
        "CostScore",
        "成本高低",
        cost_centers,
        cost_threshold,
        "CostScore分布与一维KMeans成本簇",
        COST_CLUSTER_FIGURE,
    )
    save_cluster_distribution(
        data,
        "BenefitScore",
        "效益高低",
        benefit_centers,
        benefit_threshold,
        "BenefitScore分布与一维KMeans效益簇",
        BENEFIT_CLUSTER_FIGURE,
    )

    print_final_summary(data, critic_diagnostics, thresholds, category_counts, stability)


if __name__ == "__main__":
    main()
