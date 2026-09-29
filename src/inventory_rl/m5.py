"""Time-safe ingestion of M5 daily item/store unit sales (external dataset)."""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class M5Series:
    store_id: str
    item_ids: tuple[str, ...]
    sales: np.ndarray  # [sku, day], nonnegative integer sales proxy
    train_end: int  # exclusive, zero-based
    validation_end: int  # exclusive
    source_sha256: str

    @property
    def n_sku(self) -> int:
        return len(self.item_ids)


def load_m5(path: Path, *, store_id: str = "CA_1", sku_count: int = 64,
            train_end: int = 1700, validation_end: int = 1800) -> M5Series:
    """Select active SKUs using training history only; reserve later days untouched.

    Expects the original `sales_train_validation.csv` schema. The source file
    stays outside Git. Selection by historical units is deterministic and does
    not inspect validation/test sales.
    """
    if sku_count < 2:
        raise ValueError("sku_count must be at least 2")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    candidates = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        required = {"id", "item_id", "store_id", "d_1"}
        if not required.issubset(header):
            raise ValueError("not an M5 sales_train_validation.csv file")
        days = [int(col[2:]) for col in header if col.startswith("d_") and col[2:].isdigit()]
        if days != list(range(1, len(days) + 1)):
            raise ValueError("M5 day columns must be consecutive from d_1")
        if not 28 < train_end < validation_end < len(days):
            raise ValueError("invalid chronological train/validation/test split")
        store_col = header.index("store_id")
        item_col = header.index("item_id")
        day_start = header.index("d_1")
        if header[day_start:] != [f"d_{i}" for i in days]:
            raise ValueError("M5 day columns must be at the end in order")
        for row in reader:
            if row[store_col] != store_id:
                continue
            values = np.asarray(row[day_start:], dtype=np.int32)
            if len(values) != len(days) or (values < 0).any():
                raise ValueError("invalid M5 sales row")
            recent = values[max(0, train_end - 365):train_end]
            active_days = int(np.count_nonzero(recent))
            if active_days >= 90:
                candidates.append((int(recent.sum()), row[item_col], values))
    if len(candidates) < sku_count:
        raise ValueError(f"store has only {len(candidates)} active items, need {sku_count}")
    candidates.sort(key=lambda row: (-row[0], row[1]))
    chosen = candidates[:sku_count]
    return M5Series(store_id, tuple(row[1] for row in chosen),
                    np.stack([row[2] for row in chosen]), train_end,
                    validation_end, digest)
