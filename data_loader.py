"""
data_loader.py
Loads and preprocesses the PAPA Excel dataset.
"""

import pandas as pd
import numpy as np


def load_excel(path):
    """
    Load Excel with raw PAPA items.
    Handles categorical columns and whitespace-only strings.
    """
    df = pd.read_excel(path)
    print(f"  Loaded: {df.shape}")

    exclude_cols = ["Subject", "Sampling Weight", "SAD", "GAD"]
    papa_items   = [col for col in df.columns if col not in exclude_cols]
    print(f"  Features: {len(papa_items)}")

    for col in papa_items:
        df[col] = df[col].replace(r'^\s*\.?\s*$', np.nan, regex=True)
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted.fillna(0)
        else:
            df[col] = df[col].fillna("missing")
            df[col] = df[col].astype("category").cat.codes

    if "GAD" in df.columns and "SAD" in df.columns:
        df = df.dropna(subset=["GAD", "SAD"])
        df[["GAD", "SAD"]] = df[["GAD", "SAD"]].astype(int)

    non_numeric = [c for c in papa_items if df[c].dtype == object]
    if non_numeric:
        print(f"  WARNING: Still non-numeric after encoding: {non_numeric}")
    else:
        print(f"  All {len(papa_items)} feature columns are numeric.")

    rename_map = {}
    for col in df.columns:
        clean = col
        for ch in ['[', ']', '{', '}', ':', ',', '"', "'", '\n', '\r']:
            clean = clean.replace(ch, '_')
        if clean != col:
            rename_map[col] = clean
    if rename_map:
        df = df.rename(columns=rename_map)
        papa_items = [rename_map.get(c, c) for c in papa_items]
        print(f"  Renamed {len(rename_map)} columns for LightGBM compatibility.")

    return df, papa_items


def check_compatibility(train_items, test_items):
    """Return common features between train and test datasets."""
    common = set(train_items) & set(test_items)
    return list(common), len(common) > 0
