# -*- coding: utf-8 -*-
"""
PC28-DeepFusion-AI 数据采集模块

功能：
1. 获取最近100期开奖数据
2. 兼容常见字段格式
3. 校验开奖号码、总和与分类
4. 合并历史CSV并按期号去重
5. 输出接口最新期号、CSV最新期号及更新差距
6. 生成数据更新报告

注意：
此模块不会自动下注，也不预测开奖结果。
接口返回数据可能存在延迟，程序不会伪造缺失的开奖记录。
"""

import os
import json
import logging
from datetime import datetime, timezone

import requests
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

HISTORY_FILE = os.path.join(DATA_DIR, "historical.csv")
REPORT_FILE = os.path.join(REPORT_DIR, "data_update.json")

API_URL = "https://pc28.help/api/kj.json?nbr=100"

COLUMNS = [
    "draw_nbr",
    "draw_date",
    "draw_time",
    "draw_number",
    "draw_num",
    "size_type",
    "parity_type",
    "combination_type",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def classify(total):
    size = "大" if total >= 14 else "小"
    parity = "单" if total % 2 else "双"
    return size, parity, size + parity


def parse_numbers(value):
    """解析 1+2+3 格式的开奖号码。"""

    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).strip().split("+")

    if len(parts) != 3:
        raise ValueError("开奖号码格式错误")

    numbers = []

    for item in parts:
        try:
            number = int(item)
        except (TypeError, ValueError):
            raise ValueError("开奖号码包含非整数")

        if not 0 <= number <= 9:
            raise ValueError("单个开奖号码必须在0到9之间")

        numbers.append(number)

    return numbers


def normalize_record(item):
    """将接口记录转换为统一的数据结构。"""

    if not isinstance(item, dict):
        raise ValueError("开奖记录不是对象")

    period = item.get("nbr", item.get("draw_nbr"))

    if period is None:
        raise ValueError("开奖记录缺少期号")

    period = int(period)

    if period <= 0:
        raise ValueError("开奖期号无效")

    raw_number = item.get(
        "number",
        item.get("draw_number")
    )

    numbers = parse_numbers(raw_number)
    total = sum(numbers)

    supplied_total = item.get(
        "num",
        item.get("draw_num")
    )

    if supplied_total is not None and str(supplied_total).strip():
        if int(supplied_total) != total:
            raise ValueError("接口总和与开奖号码不一致")

    size, parity, combination = classify(total)

    supplied_combination = item.get(
        "combination",
        item.get("combination_type")
    )

    if supplied_combination:
        if str(supplied_combination).strip() != combination:
            raise ValueError("接口分类与计算结果不一致")

    date = str(item.get("date", item.get("draw_date", "")))
    draw_time = str(item.get("time", item.get("draw_time", "")))

    return {
        "draw_nbr": period,
        "draw_date": date,
        "draw_time": draw_time,
        "draw_number": "+".join(map(str, numbers)),
        "draw_num": total,
        "size_type": size,
        "parity_type": parity,
        "combination_type": combination,
    }


def fetch_latest():
    """请求接口并返回校验通过的开奖记录。"""

    logging.info("正在请求开奖接口：%s", API_URL)

    response = requests.get(
        API_URL,
        timeout=30,
        headers={
            "User-Agent": "PC28-DeepFusion-AI/1.0"
        },
    )

    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError:
        payload = json.loads(
            response.content.decode("utf-8-sig")
        )

    if isinstance(payload, dict):
        records = payload.get("data")

        if records is None:
            records = payload.get("result")

        if records is None:
            raise ValueError(
                "接口返回格式不符合预期：没有 data/result 字段"
            )

    elif isinstance(payload, list):
        records = payload

    else:
        raise ValueError("接口返回的数据类型不正确")

    if not isinstance(records, list) or not records:
        raise ValueError("接口没有返回有效开奖记录")

    normalized = []
    rejected = []

    for index, item in enumerate(records):
        try:
            normalized.append(normalize_record(item))
        except Exception as exc:
            rejected.append({
                "index": index,
                "error": str(exc),
            })

    if not normalized:
        raise ValueError(
            "所有接口记录均未通过校验，已停止更新"
        )

    if rejected:
        logging.warning(
            "有 %s 条接口记录未通过校验",
            len(rejected),
        )
        for item in rejected[:10]:
            logging.warning(
                "无效记录 index=%s，原因=%s",
                item["index"],
                item["error"],
            )

    df = pd.DataFrame(normalized, columns=COLUMNS)

    if df["draw_nbr"].duplicated().any():
        duplicated = (
            df.loc[
                df["draw_nbr"].duplicated(keep=False),
                "draw_nbr",
            ]
            .astype(int)
            .tolist()
        )
        raise ValueError(
            f"接口返回重复期号：{duplicated[:10]}"
        )

    api_min = int(df["draw_nbr"].min())
    api_max = int(df["draw_nbr"].max())

    logging.info("接口原始记录数：%s", len(records))
    logging.info("接口有效记录数：%s", len(df))
    logging.info("接口最小期号：%s", api_min)
    logging.info("接口最大期号：%s", api_max)

    return df, rejected


def load_history():
    """读取已有历史库。"""

    if not os.path.exists(HISTORY_FILE):
        logging.warning("历史CSV不存在，将创建新文件")
        return pd.DataFrame(columns=COLUMNS)

    history = pd.read_csv(
        HISTORY_FILE,
        encoding="utf-8-sig",
        dtype={"draw_nbr": str},
    )

    missing = [
        col for col in COLUMNS
        if col not in history.columns
    ]

    if missing:
        raise ValueError(
            "历史CSV缺少必要字段：" + ", ".join(missing)
        )

    history = history[COLUMNS].copy()

    history["draw_nbr"] = pd.to_numeric(
        history["draw_nbr"],
        errors="raise",
    ).astype("int64")

    if history["draw_nbr"].duplicated().any():
        raise ValueError(
            "现有历史库存在重复期号；为防止误覆盖，已停止更新"
        )

    logging.info("历史CSV记录数：%s", len(history))

    if not history.empty:
        logging.info(
            "CSV更新前最小期号：%s",
            int(history["draw_nbr"].min()),
        )
        logging.info(
            "CSV更新前最大期号：%s",
            int(history["draw_nbr"].max()),
        )

    return history


def merge_history(history, latest):
    """合并历史与接口记录，不静默覆盖已有开奖号码。"""

    history = history.copy()
    latest = latest.copy()

    if not history.empty:
        history["draw_nbr"] = history["draw_nbr"].astype("int64")

    latest["draw_nbr"] = latest["draw_nbr"].astype("int64")

    old_ids = set(
        history["draw_nbr"].astype(int).tolist()
    )

    old_lookup = {
        int(row["draw_nbr"]): row
        for _, row in history.iterrows()
    }

    for _, row in latest.iterrows():
        period = int(row["draw_nbr"])

        if period not in old_lookup:
            continue

        old = old_lookup[period]

        for col in [
            "draw_number",
            "draw_num",
            "combination_type",
        ]:
            if str(old[col]) != str(row[col]):
                raise ValueError(
                    f"期号 {period} 的历史记录与接口不一致，"
                    "已停止写入，防止静默覆盖"
                )

    new_rows = latest[
        ~latest["draw_nbr"].isin(old_ids)
    ].copy()

    merged = pd.concat(
        [history, new_rows],
        ignore_index=True,
    )

    merged = merged.drop_duplicates(
        subset=["draw_nbr"],
        keep="first",
    )

    merged = merged.sort_values(
        "draw_nbr",
        ascending=True,
    ).reset_index(drop=True)

    logging.info("本次识别到新增期数：%s", len(new_rows))

    return merged, new_rows


def atomic_save_csv(df, path):
    """先写临时文件，成功后再替换正式文件。"""

    temp_path = path + ".tmp"

    df.to_csv(
        temp_path,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(temp_path, path)


def write_report(report):
    os.makedirs(REPORT_DIR, exist_ok=True)

    temp_path = REPORT_FILE + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temp_path, REPORT_FILE)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()

    history = load_history()
    latest, rejected = fetch_latest()

    history_max = (
        int(history["draw_nbr"].max())
        if not history.empty
        else None
    )

    api_max = int(latest["draw_nbr"].max())
    api_min = int(latest["draw_nbr"].min())

    logging.info("CSV更新前最大期号：%s", history_max)
    logging.info("接口最大期号：%s", api_max)

    if history_max is not None:
        if api_max <= history_max:
            logging.warning(
                "接口没有提供比CSV更新的期号："
                "CSV=%s，接口=%s",
                history_max,
                api_max,
            )
        elif api_max > history_max + 1:
            logging.warning(
                "检测到期号跨度：CSV=%s，接口=%s，"
                "请核对中间期号是否都存在",
                history_max,
                api_max,
            )

    merged, new_rows = merge_history(
        history,
        latest,
    )

    # 仅在校验和合并成功后写入正式文件。
    atomic_save_csv(merged, HISTORY_FILE)

    merged_max = (
        int(merged["draw_nbr"].max())
        if not merged.empty
        else None
    )

    missing_periods = []

    if history_max is not None and api_max > history_max + 1:
        present = set(
            latest["draw_nbr"].astype(int).tolist()
        )
        missing_periods = [
            period
            for period in range(history_max + 1, api_max)
            if period not in present
        ]

    report = {
        "project": "PC28-DeepFusion-AI",
        "status": "success",
        "updated_at": started_at,
        "api_url": API_URL,
        "history_before": int(len(history)),
        "api_records_validated": int(len(latest)),
        "api_min_period": api_min,
        "api_max_period": api_max,
        "history_max_before": history_max,
        "new_records": int(len(new_rows)),
        "history_after": int(len(merged)),
        "history_max_after": merged_max,
        "missing_periods_between_old_and_api": missing_periods,
        "rejected_count": len(rejected),
        "rejected_records": rejected,
        "api_ahead_of_history": (
            api_max > history_max
            if history_max is not None
            else None
        ),
    }

    if history_max is not None and api_max <= history_max:
        report["status"] = "no_new_periods_from_api"

    write_report(report)

    logging.info("更新后CSV记录数：%s", len(merged))
    logging.info("更新后CSV最大期号：%s", merged_max)
    logging.info("接口最大期号：%s", api_max)
    logging.info("本次新增期数：%s", len(new_rows))

    if missing_periods:
        logging.warning(
            "接口返回范围内发现缺失期号：%s",
            missing_periods[:30],
        )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("数据更新失败")
        raise SystemExit(1)
