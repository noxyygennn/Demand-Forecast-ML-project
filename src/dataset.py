from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# sales должен идти первым: эта колонка есть и в истории, и в будущем окне
# в будущем окне sales зануляем, чтобы не было утечки
FEATURE_COLS = [
    "sales",
    "price",
    "promo_flag",
    "discount_pct",
    "is_weekend",
    "is_holiday",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


@dataclass
class TSConfig:
    lookback: int = 28
    horizon: int = 14


def add_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["dow"] = out["date"].dt.dayofweek.astype(int)
    out["month"] = out["date"].dt.month.astype(int)

    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    if "is_weekend" not in out.columns:
        out["is_weekend"] = (out["dow"] >= 5).astype(int)
    if "is_holiday" not in out.columns:
        out["is_holiday"] = 0

    # нормализуем названия входных колонок
    if "promo" in out.columns and "promo_flag" not in out.columns:
        out = out.rename(columns={"promo": "promo_flag"})
    if "promo_flag" not in out.columns:
        out["promo_flag"] = 0
    if "discount_pct" not in out.columns:
        out["discount_pct"] = 0.0
    if "price" not in out.columns:
        out["price"] = 1.0
    if "sales" not in out.columns:
        out["sales"] = 0.0

    return out


def build_sequences_with_future_exog(df: pd.DataFrame, cfg: TSConfig):
    """
    Строит последовательности по каждому SKU.

    X: (N, lookback + horizon, F)
       - первые lookback шагов: история
       - последние horizon шагов: будущие известные признаки
    y: (N, horizon)
       - реальные будущие продажи в абсолютном масштабе
    """
    df = add_calendar_feats(df).sort_values(["sku", "date"]).reset_index(drop=True)
    out = {}
    lookback, horizon = cfg.lookback, cfg.horizon

    for sku, g in df.groupby("sku", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        arr = g[FEATURE_COLS].to_numpy(dtype=float)
        sales = g["sales"].to_numpy(dtype=float)
        dates = g["date"].to_numpy()

        Xs, ys, ds = [], [], []
        for i in range(lookback, len(g) - horizon):
            past = arr[i - lookback : i, :].copy()
            fut = arr[i : i + horizon, :].copy()

            # future sales неизвестны, поэтому зануляем только эту колонку
            fut[:, 0] = 0.0

            Xseq = np.vstack([past, fut])
            yseq = sales[i : i + horizon].copy()

            Xs.append(Xseq)
            ys.append(yseq)
            ds.append(dates[i])

        if Xs:
            out[sku] = (np.stack(Xs), np.stack(ys), np.array(ds))

    return out
