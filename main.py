
# -*- coding: utf-8 -*-
"""
PC28-DeepFusion-AI
多模型融合、独立测试、滚动回测、下一期研究性概率估计。
仅用于历史数据统计研究，不保证预测准确。
"""

import os
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT, "data", "historical.csv")
REPORT_DIR = os.path.join(ROOT, "reports")
REPORT_FILE = os.path.join(REPORT_DIR, "latest.json")

GROUPS = ["大单", "小单", "大双", "小双"]
RANDOM_STATE = 42


def log(message):
    print("[PC28-AI] " + str(message), flush=True)


def classify(total):
    size = "大" if total >= 14 else "小"
    parity = "单" if total % 2 else "双"
    return size + parity


def parse_number(value):
    parts = str(value).strip().replace(" ", "").split("+")
    if len(parts) != 3:
        raise ValueError("开奖号码格式错误：" + str(value))

    nums = [int(x) for x in parts]

    if any(x < 0 or x > 9 for x in nums):
        raise ValueError("开奖号码必须为0至9")

    return nums


def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("找不到 " + DATA_FILE)

    df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig",
        dtype={"draw_nbr": str}
    )

    required = [
        "draw_nbr", "draw_number", "draw_num",
        "combination_type"
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError("CSV缺少字段：" + ", ".join(missing))

    df["draw_nbr"] = pd.to_numeric(
        df["draw_nbr"], errors="raise"
    ).astype("int64")

    df["draw_num"] = pd.to_numeric(
        df["draw_num"], errors="raise"
    ).astype("int64")

    df = df.sort_values("draw_nbr").reset_index(drop=True)

    if df["draw_nbr"].duplicated().any():
        raise ValueError("历史数据有重复期号")

    parsed = df["draw_number"].apply(parse_number)

    df["n1"] = parsed.apply(lambda x: x[0])
    df["n2"] = parsed.apply(lambda x: x[1])
    df["n3"] = parsed.apply(lambda x: x[2])

    calculated = df["n1"] + df["n2"] + df["n3"]

    if not (calculated == df["draw_num"]).all():
        raise ValueError("开奖号码与总和不一致")

    df["group"] = calculated.apply(classify)

    if not (
        df["group"].astype(str).values
        == df["combination_type"].astype(str).str.strip().values
    ).all():
        raise ValueError("分类字段与计算结果不一致")

    if len(df) < 200:
        raise ValueError("历史数据不足200期")

    gaps = df["draw_nbr"].diff().dropna()

    info = {
        "rows": int(len(df)),
        "first_period": int(df["draw_nbr"].iloc[0]),
        "last_period": int(df["draw_nbr"].iloc[-1]),
        "period_gaps": int((gaps != 1).sum()),
        "validation": "passed"
    }

    return df, info


def build_features(df):
    """
    每一行的特征只使用此前已发生的开奖。
    shift(1) 防止把当前行结果泄漏到特征中。
    """
    f = pd.DataFrame(index=df.index)

    for col in ["draw_num", "n1", "n2", "n3"]:
        previous = df[col].shift(1)

        for lag in [1, 2, 3, 5, 10]:
            f[f"{col}_lag_{lag}"] = df[col].shift(lag)

        for window in [5, 10, 30]:
            f[f"{col}_mean_{window}"] = previous.rolling(
                window, min_periods=window
            ).mean()

        f[f"{col}_std_10"] = previous.rolling(
            10, min_periods=10
        ).std()

    previous_group = df["group"].shift(1)

    for group in GROUPS:
        indicator = (df["group"] == group).astype(float).shift(1)

        f["previous_" + group] = (
            previous_group == group
        ).astype(float)

        for window in [10, 30]:
            f[f"{group}_frequency_{window}"] = (
                indicator.rolling(window, min_periods=window).mean()
            )

    previous_sum = df["draw_num"].shift(1)

    for lag in [1, 2, 5]:
        f[f"sum_change_{lag}"] = previous_sum.diff(lag)

    big = (df["draw_num"] >= 14).astype(float).shift(1)
    odd = (df["draw_num"] % 2 == 1).astype(float).shift(1)

    for window in [10, 30]:
        f[f"big_rate_{window}"] = big.rolling(
            window, min_periods=window
        ).mean()

        f[f"odd_rate_{window}"] = odd.rolling(
            window, min_periods=window
        ).mean()

    return f.replace([np.inf, -np.inf], np.nan)


def make_models():
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=8,
                random_state=RANDOM_STATE + 1,
                n_jobs=-1
            ))
        ])
    }


def get_probabilities(model, X):
    raw = model.predict_proba(X)
    classes = list(model.named_steps["model"].classes_)
    result = np.zeros((len(X), len(GROUPS)))

    for j, group in enumerate(GROUPS):
        if group in classes:
            result[:, j] = raw[:, classes.index(group)]

    sums = result.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return result / sums


def evaluate(y_true, probabilities):
    pred = [
        GROUPS[i] for i in np.argmax(probabilities, axis=1)
    ]

    try:
        loss = float(log_loss(
            y_true,
            probabilities,
            labels=GROUPS
        ))
    except Exception:
        loss = None

    counts = pd.Series(y_true).value_counts(normalize=True)

    return {
        "samples": int(len(y_true)),
        "accuracy": round(
            float(accuracy_score(y_true, pred)), 6
        ),
        "log_loss": round(loss, 6) if loss is not None else None,
        "majority_baseline_accuracy": round(
            float(counts.max()), 6
        )
    }


def run_rolling_backtest(X, y, folds=4):
    """扩展窗口回测：每一折只用该测试块之前的数据。"""
    n = len(X)
    initial = int(n * 0.60)
    remaining = n - initial
    results = []

    if remaining < folds:
        return results

    boundaries = np.linspace(
        initial, n, folds + 1, dtype=int
    )
    models = make_models()

    for fold in range(folds):
        start = int(boundaries[fold])
        end = int(boundaries[fold + 1])

        if end <= start:
            continue

        train_y = y.iloc[:start]
        if train_y.nunique() < 4:
            continue

        predictions = []

        for model in models.values():
            fitted = clone(model)
            fitted.fit(X.iloc[:start], train_y)
            predictions.append(
                get_probabilities(fitted, X.iloc[start:end])
            )

        fused = np.mean(np.stack(predictions), axis=0)

        result = evaluate(
            y.iloc[start:end].tolist(), fused
        )
        result["fold"] = fold + 1
        result["train_samples"] = start
        results.append(result)

        log(
            "滚动回测第 %d 折：准确率 %.4f"
            % (fold + 1, result["accuracy"])
        )

    return results


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    log("读取历史数据")
    df, data_info = load_data()

    log("生成无未来泄漏的特征")
    features = build_features(df)
    valid = features.notna().all(axis=1)

    X = features.loc[valid].reset_index(drop=True)
    y = df.loc[valid, "group"].reset_index(drop=True)
    clean = df.loc[valid].reset_index(drop=True)

    if len(X) < 200:
        raise ValueError("有效特征样本不足")

    # 时间顺序：前80%训练，最后20%独立测试。
    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if y_train.nunique() < 4:
        raise ValueError("训练集没有覆盖全部四组")

    reports = {}
    test_probs = []
    models = make_models()

    log("训练样本：%d，独立测试样本：%d" % (
        len(X_train), len(X_test)
    ))

    for name, model in models.items():
        log("训练模型：" + name)
        fitted = clone(model)
        fitted.fit(X_train, y_train)

        probs = get_probabilities(fitted, X_test)
        reports[name] = evaluate(y_test.tolist(), probs)
        test_probs.append(probs)

    # 透明的等权融合，不根据最终测试集调权重。
    fused_test = np.mean(np.stack(test_probs), axis=0)
    fusion_report = evaluate(y_test.tolist(), fused_test)

    # 朴素基线：训练集各分类频率。
    train_freq = (
        y_train.value_counts(normalize=True)
        .reindex(GROUPS, fill_value=0)
        .to_numpy()
    )
    baseline_probs = np.tile(train_freq, (len(y_test), 1))
    baseline_report = evaluate(
        y_test.tolist(), baseline_probs
    )

    rolling = run_rolling_backtest(X, y)

    # 使用全部可用历史训练最终模型，仅用于生成下一期研究性估计。
    latest_models = {}

    for name, model in models.items():
        fitted = clone(model)
        fitted.fit(X, y)
        latest_models[name] = fitted

    # 在历史末尾追加占位行；shift特征保证只读取之前的记录。
    placeholder = df.iloc[[-1]].copy()
    placeholder["draw_nbr"] = int(df["draw_nbr"].iloc[-1]) + 1
    placeholder["draw_number"] = "0+0+0"
    placeholder["n1"] = 0
    placeholder["n2"] = 0
    placeholder["n3"] = 0
    placeholder["draw_num"] = 0
    placeholder["group"] = GROUPS[0]

    extended = pd.concat(
        [df, placeholder], ignore_index=True
    )

    next_features = build_features(extended).iloc[[-1]]
    next_probs = []

    for model in latest_models.values():
        next_probs.append(
            get_probabilities(model, next_features)[0]
        )

    fused_next = np.mean(np.stack(next_probs), axis=0)

    probabilities = {
        group: round(float(fused_next[i]), 6)
        for i, group in enumerate(GROUPS)
    }

    ranked = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # 两组概率之和仅表示模型给出的合计估计。
    # 不代表真实中奖概率，也不保证比单组更准确。
    top_two = [
        {"group": group, "probability": probability}
        for group, probability in ranked[:2]
    ]

    report = {
        "project": "PC28-DeepFusion-AI",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": data_info,
        "target": "combination_type",
        "groups": GROUPS,
        "method": {
            "split": "time_ordered_80_20",
            "fusion": "equal_weight",
            "features_use_only_prior_draws": True,
            "final_test_used_for_weight_tuning": False
        },
        "independent_test": {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "models": reports,
            "equal_weight_ensemble": fusion_report,
            "historical_frequency_baseline": baseline_report
        },
        "rolling_backtest": rolling,
        "next_draw_estimate": {
            "period_reference": int(df["draw_nbr"].max()) + 1,
            "probabilities": probabilities,
            "top_two_groups": top_two,
            "warning": (
                "研究性模型估计，不保证预测准确。"
                "若开奖近似独立随机，历史数据可能没有稳定预测力。"
            )
        }
    }

    temp_file = REPORT_FILE + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    os.replace(temp_file, REPORT_FILE)

    log("独立测试融合准确率：%.4f" % (
        fusion_report["accuracy"]
    ))
    log("历史频率基线准确率：%.4f" % (
        baseline_report["accuracy"]
    ))
    log("报告已保存：" + REPORT_FILE)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
