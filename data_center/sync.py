#!/usr/bin/env python3
"""PC28 data ingestion: immutable archive + latest 10,000 rolling window.

The parser intentionally fails closed when the upstream API schema is unknown.
No credentials are required for the public data endpoint.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = Path(os.getenv("PC28_INITIAL_CSV", ROOT / "data" / "raw" / "initial_history.csv"))
ARCHIVE = ROOT / "data" / "archive" / "history.csv"
ROLLING = ROOT / "data" / "processed" / "rolling_10000.csv"
MANIFEST = ROOT / "data" / "processed" / "manifest.json"
API_URL = os.getenv("PC28_API_URL", "https://pc28.help/api/kj.json?nbr=100")
WINDOW = 10_000

PERIOD_KEYS = ("期号", "期數", "period", "issue", "issue_no", "expect", "draw_id", "qihao", "no")
CODE_KEYS = ("开奖号", "开奖号码", "开奖结果", "开奖号碼", "open_code", "opencode", "code", "result", "number", "numbers")
N1_KEYS = ("n1", "号码1", "第一位", "ball1")
N2_KEYS = ("n2", "号码2", "第二位", "ball2")
N3_KEYS = ("n3", "号码3", "第三位", "ball3")
TIME_KEYS = ("开奖时间", "开奖日期", "draw_time", "open_time", "date", "time")


def first_value(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for key in aliases:
        if key in row and row[key] not in (None, ""):
            return row[key]
        if key.lower() in lowered and lowered[key.lower()] not in (None, ""):
            return lowered[key.lower()]
    return None


def parse_digits(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        parts = list(value[:3])
    elif isinstance(value, dict):
        parts = [value.get(k) for k in ("n1", "n2", "n3")]
    else:
        text = str(value).strip()
        # Accept 1,2,3 / 1+2+3 / 1 2 3 / 123, but never guess from long IDs.
        nums = re.findall(r"(?<!\d)\d(?!\d)", text)
        if len(nums) == 3:
            parts = nums
        elif re.fullmatch(r"\d{3}", text):
            parts = list(text)
        else:
            return None
    try:
        digits = tuple(int(x) for x in parts)
    except (TypeError, ValueError):
        return None
    if len(digits) != 3 or any(x < 0 or x > 9 for x in digits):
        return None
    return digits  # type: ignore[return-value]


def normalize(row: dict[str, Any], source: str) -> dict[str, Any]:
    period = first_value(row, PERIOD_KEYS)
    if period is None:
        raise ValueError(f"Missing period field. Available fields: {list(row)[:30]}")
    period = str(period).strip()
    if not period:
        raise ValueError("Empty period")

    digits = parse_digits(first_value(row, CODE_KEYS))
    if digits is None:
        digits = parse_digits([
            first_value(row, N1_KEYS),
            first_value(row, N2_KEYS),
            first_value(row, N3_KEYS),
        ])
    if digits is None:
        raise ValueError(f"Cannot parse three 0-9 draw digits for period {period}; fields={list(row)[:30]}")

    total = sum(digits)
    size = "大" if total >= 14 else "小"
    parity = "单" if total % 2 else "双"
    return {
        "period": period,
        "n1": digits[0], "n2": digits[1], "n3": digits[2],
        "sum": total,
        "combination_type": size + parity,
        "draw_time": str(first_value(row, TIME_KEYS) or ""),
        "source": source,
    }


def find_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if payload and all(isinstance(x, dict) for x in payload):
            return payload
        for item in payload:
            found = find_records(item)
            if found:
                return found
    elif isinstance(payload, dict):
        if first_value(payload, PERIOD_KEYS) is not None:
            return [payload]
        # Common API wrappers; recurse without assuming a specific response envelope.
        for value in payload.values():
            if isinstance(value, (list, dict)):
                found = find_records(value)
                if found:
                    return found
    return []


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        return [normalize(row, "csv") for row in reader if any(v not in (None, "") for v in row.values())]


def read_archive() -> dict[str, dict[str, Any]]:
    if not ARCHIVE.exists():
        return {}
    with ARCHIVE.open("r", encoding="utf-8", newline="") as f:
        return {row["period"]: row for row in csv.DictReader(f)}


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["period", "n1", "n2", "n3", "sum", "combination_type", "draw_time", "source"]
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> int:
    archive = read_archive()
    imported = 0

    # Bootstrap only from the user's supplied CSV. Never fabricate seed data.
    if not archive and RAW_CSV.exists():
        for row in read_csv(RAW_CSV):
            if row["period"] in archive:
                raise ValueError(f"Duplicate period in initial CSV: {row['period']}")
            archive[row["period"]] = row
            imported += 1

    if not archive:
        print(f"WAITING_FOR_CSV: upload the 10,000-row file to {RAW_CSV.relative_to(ROOT)}")
        return 0

    response = requests.get(API_URL, timeout=30, headers={"User-Agent": "PC28-DeepFusion-AI/1.0"})
    response.raise_for_status()
    payload = response.json()
    records = find_records(payload)
    if not records:
        raise ValueError("API response schema not recognized; archive was not changed.")

    added = 0
    for item in records:
        row = normalize(item, "api")
        old = archive.get(row["period"])
        if old is not None:
            # Existing periods are immutable unless the API explicitly contradicts them.
            old_digits = (int(old["n1"]), int(old["n2"]), int(old["n3"]))
            new_digits = (row["n1"], row["n2"], row["n3"])
            if old_digits != new_digits:
                raise ValueError(f"Conflicting result for period {row['period']}; refusing overwrite.")
            continue
        archive[row["period"]] = row
        added += 1

    # Numeric period ordering is preferred; lexical fallback preserves opaque period IDs.
    def order_key(row: dict[str, Any]):
        p = str(row["period"])
        m = re.search(r"\d+", p)
        return (int(m.group()) if m else -1, p)

    all_rows = sorted(archive.values(), key=order_key)
    rolling = all_rows[-WINDOW:]
    atomic_csv(ARCHIVE, all_rows)
    atomic_csv(ROLLING, rolling)

    manifest = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "archive_rows": len(all_rows),
        "rolling_rows": len(rolling),
        "first_period": rolling[0]["period"] if rolling else None,
        "latest_period": rolling[-1]["period"] if rolling else None,
        "csv_imported": imported,
        "api_added": added,
        "api_url": API_URL,
        "status": "ready" if len(rolling) == WINDOW else "awaiting_10000_rows",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DATA_SYNC_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
