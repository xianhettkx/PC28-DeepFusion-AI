# -*- coding: utf-8 -*-
"""
PC28-DeepFusion-AI 数据采集模块

功能：
1. 获取最近100期开奖数据
2. 兼容常见字段格式
3. 校验开奖号码、总和与分类
4. 合并历史CSV并按期号去重
5. 生成数据更新报告

注意：
此模块不会自动下注，也不预测开奖结果。
"""

import os
import json
import time
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
    """请求开奖接口并转换数据。"""

    logging.info("正在请求开奖接口")

    response = requests.get(
        API_URL,
        timeout=20,
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
            "有 %s 条记录未通过校验，未写入历史库",
            len(rejected)
        )

    df = pd.DataFrame(normalized, columns=COLUMNS)

    if df["draw_nbr"].duplicated().any():
        raise ValueError(
            "接口返回了重复期号，请检查数据源后再更新"
        )

    return df, rejected


def load_history():
    """读取已有历史库，不存在时创建空表。"""

    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=COLUMNS)

    history = pd.read_csv(
        HISTORY_FILE,
        encoding="utf-8-sig",
        dtype={"draw_nbr": str},
    )

    missing = [c for c in COLUMNS if c not in history.columns]

    if missing:
        raise ValueError(
            "历史CSV缺少必要字段：" + ", ".join(missing)
        )

    history = history[COLUMNS].copy()

    history["draw_nbr"] = pd.to_numeric(
        history["draw_nbr"], errors="raise"
    ).astype("int64")

    if history["draw_nbr"].duplicated().any():
        raise ValueError(
            "现有历史库存在重复期号；为防止误覆盖，已停止更新"
        )

    return history


def merge_history(history, latest):
    """保留历史记录，并按期号合并新记录。"""

    old_ids = set(history["draw_nbr"].astype(int).tolist())

    latest = latest.copy()
    latest["draw_nbr"] = latest["draw_nbr"].astype("int64")

    # 对相同期号，如果内容不同，不允许静默覆盖。
    old_lookup = {
        int(row["draw_nbr"]): row
        for _, row in history.iterrows()
    }

    for _, row in latest.iterrows():
        period = int(row["draw_nbr"])

        if period in old_lookup:
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

    logging.info("现有历史记录：%s 期", len(history))

    latest, rejected = fetch_latest()

    merged, new_rows = merge_history(
        history,
        latest,
    )

    # 仅在所有校验通过后写入正式历史库。
    atomic_save_csv(merged, HISTORY_FILE)

    report = {
        "project": "PC28-DeepFusion-AI",
        "status": "success",
        "updated_at": started_at,
        "api_url": API_URL,
        "history_before": int(len(history)),
        "api_records_validated": int(len(latest)),
        "new_records": int(len(new_rows)),
        "history_after": int(len(merged)),
        "rejected_records": rejected,
        "latest_period": (
            int(merged["draw_nbr"].max())
            if len(merged)
            else None
        ),
        "earliest_period": (
            int(merged["draw_nbr"].min())
            if len(merged)
            else None
        ),
    }

    write_report(report)

    logging.info(
        "更新完成：新增 %s 期，总计 %s 期",
        len(new_rows),
        len(merged),
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("数据更新失败")
        raise SystemExit(1) from exc
