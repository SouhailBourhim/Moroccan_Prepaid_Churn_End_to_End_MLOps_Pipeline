#!/usr/bin/env python3
"""Synthesize a 2M-row Moroccan prepaid telecom churn dataset.

Steps
-----
1. Load a stratified sample of Expresso Train.csv as a structural seed (learns
   realistic feature correlations: regularity vs churn, call volumes, etc.).
2. Train CTGAN (sdv library) on the sample.
3. Generate ~2.4M synthetic rows (20% headroom for class-ratio resampling).
4. Post-process to enforce Moroccan market constraints:
   - REGION        → 12 Moroccan cities at population weights. Expresso regions
                     are relabelled by a churn-rate-ordered (comonotone) mapping,
                     so the region–churn association learned from Expresso is
                     kept; which city inherits which Senegalese region's churn
                     level is arbitrary.
   - OPERATOR      → Inwi / Maroc_Telecom / Orange_Maroc at market shares
                     (random metadata: carries no churn signal; dropped before
                     modelling)
   - ARPU_SEGMENT  → calibrated to 25–30 MAD monthly ARPU (Moroccan target)
   - REVENUE / MONTANT → proportionally rescaled with ARPU
   - DATA_VOLUME   → raised to 74 % data users for Moroccan 4G penetration; new
                     data users are drawn preferentially from active (high
                     REGULARITY) subscribers rather than uniformly at random
   - TOP_PACK      → Moroccan-style pack names (MT Jaweb/Hissab, Inwi Jibi,
                     Orange), mapped from Expresso packs the same comonotone way
   - MRG           → allow YES values (~12 %); Expresso is always NO, so there
                     is no signal to transfer and MRG is random (no churn signal)
   - CHURN         → resample classes to 2.7 % monthly churn rate
5. Subsample to exactly 2 M rows, save to data/raw/moroccan_telecom_churn.csv.
6. Print a validation report comparing key statistics between datasets.

What this dataset is
--------------------
Expresso's joint feature/label distribution (via CTGAN), shifted to a 2.7 %
churn prior and relabelled with Moroccan names and units. It is useful for
exercising the pipeline on Moroccan-shaped data; it is not evidence about how
Moroccan subscribers actually churn.

--no-ctgan draws rows *without replacement* from the full Expresso Train.csv.
Bootstrapping a small sample with replacement would create many identical rows
that land on both sides of the train/holdout split and inflate holdout scores;
if the requested size is not reachable without replacement, the output is
capped and a warning is printed.

Usage
-----
    python generate_moroccan_dataset.py [options]
    python generate_moroccan_dataset.py --no-ctgan   # fast parametric sampler

Options
-------
    --seed INT           Random seed (default 42)
    --sample-size INT    Expresso rows for CTGAN training (default 50 000)
    --n-rows INT         Target output rows (default 2 000 000)
    --epochs INT         CTGAN training epochs (default 300)
    --batch-size INT     CTGAN batch size (default 500)
    --no-ctgan           Skip CTGAN; resample full Expresso without replacement
    --output PATH        Output CSV path (default data/raw/moroccan_telecom_churn.csv)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# ── Moroccan market parameters ─────────────────────────────────────────────────

# City populations approximate (2024 estimates). Weights normalised in usage.
MOROCCAN_REGIONS: dict[str, float] = {
    "Casablanca":   0.320,
    "Rabat":        0.154,
    "Fès":          0.098,
    "Marrakech":    0.091,
    "Tanger":       0.081,
    "Agadir":       0.064,
    "Meknès":       0.056,
    "Oujda":        0.043,
    "Béni_Mellal":  0.034,
    "Tétouan":      0.029,
    "Settat":       0.020,
    "Nador":        0.010,
}

# ANRT 2023 mobile market share report
OPERATORS: dict[str, float] = {
    "Inwi":          0.353,
    "Maroc_Telecom": 0.325,
    "Orange_Maroc":  0.322,
}

TARGET_CHURN_RATE: float = 0.027   # 2.7 % monthly
TARGET_ARPU_MAD:   float = 27.5    # mid-point of 25–30 MAD
DATA_USER_RATE:    float = 0.74    # fraction with DATA_VOLUME > 0 (74 % 4G penetration)
DATA_VOL_SCALE:    float = 1.80    # scale factor for non-zero DATA_VOLUME
MRG_VAS_RATE:      float = 0.12    # 12 % VAS subscribers

# Moroccan pack catalogue (name, relative frequency weight)
MOROCCAN_PACKS: list[tuple[str, float]] = [
    ("Hissab 3 GO_7j_MT",         0.180),
    ("Jaweb 10 MAD_24H_MT",       0.120),
    ("Hissab 5 GO_30j_MT",        0.100),
    ("Jibi 1 GO_24H_Inwi",        0.090),
    ("Jibi 5 GO_7j_Inwi",         0.070),
    ("Orange Data 500MB_24H",     0.055),
    ("Orange Forfait 5 GO_30j",   0.050),
    ("Pass Social_24H_MT",        0.040),
    ("Jaweb 20 MAD_3j_MT",        0.035),
    ("Jibi 10 GO_30j_Inwi",       0.035),
    ("Orange Data 2 GO_7j",       0.030),
    ("Hissab 10 GO_30j_MT",       0.025),
    ("Pass YouTube_24H_MT",       0.025),
    ("Jibi Nuit illimite_Inwi",   0.020),
    ("Orange Appels_30j",         0.020),
    ("Pass WhatsApp_7j_MT",       0.015),
    ("Jibi Pack Famille_Inwi",    0.015),
    ("Orange Max Social_30j",     0.015),
    ("Hissab 1 GO_3j_MT",         0.015),
    ("OTHER",                     0.041),
]

# Expresso column dtypes used when loading the seed data
_EXPRESSO_COLS = [
    "REGION", "TENURE", "MONTANT", "FREQUENCE_RECH", "REVENUE",
    "ARPU_SEGMENT", "FREQUENCE", "DATA_VOLUME", "ON_NET", "ORANGE",
    "TIGO", "ZONE1", "ZONE2", "MRG", "REGULARITY", "TOP_PACK",
    "FREQ_TOP_PACK", "CHURN",
]

# Final column order in the output CSV (Expresso schema + OPERATOR)
OUTPUT_COLS = [
    "user_id", "REGION", "TENURE", "MONTANT", "FREQUENCE_RECH",
    "REVENUE", "ARPU_SEGMENT", "FREQUENCE", "DATA_VOLUME",
    "ON_NET", "ORANGE", "TIGO", "ZONE1", "ZONE2",
    "MRG", "REGULARITY", "TOP_PACK", "FREQ_TOP_PACK", "CHURN", "OPERATOR",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_expresso_sample(sample_size: int | None, seed: int) -> pd.DataFrame:
    """Load Expresso Train.csv; stratified-sample `sample_size` rows (None = all rows)."""
    train_path = ROOT / "data" / "raw" / "expresso" / "Train.csv"
    if not train_path.exists():
        sys.exit(f"ERROR: Expresso Train.csv not found at {train_path}")

    print(f"[1/6] Loading Expresso seed from {train_path} …")
    df = pd.read_csv(train_path, usecols=_EXPRESSO_COLS, low_memory=False)
    df["CHURN"] = df["CHURN"].astype(int)
    if sample_size is None or sample_size >= len(df):
        print(f"    Expresso rows: {len(df):,}  churn rate={df['CHURN'].mean():.3%}")
        return df.reset_index(drop=True)

    # Stratified sample — preserve Expresso churn ratio
    pos = df[df["CHURN"] == 1]
    neg = df[df["CHURN"] == 0]
    n_pos = int(sample_size * df["CHURN"].mean())
    n_neg = sample_size - n_pos
    sample = pd.concat([
        pos.sample(min(n_pos, len(pos)),  random_state=seed),
        neg.sample(min(n_neg, len(neg)), random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"    Expresso sample: {len(sample):,} rows  "
          f"churn rate={sample['CHURN'].mean():.3%}")
    return sample


# ── CTGAN training and generation ─────────────────────────────────────────────

def _train_ctgan(
    sample: pd.DataFrame,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Any:
    try:
        from sdv.single_table import CTGANSynthesizer
    except ImportError:
        sys.exit(
            "ERROR: sdv is required.\n"
            "Install with:  pip install sdv\n"
            "Or run with:   --no-ctgan  to resample real Expresso rows instead."
        )

    print(f"[2/6] Training CTGAN  epochs={epochs}  batch={batch_size}  "
          f"seed={seed}  rows={len(sample):,} …")
    t0 = time.perf_counter()

    # CTGAN draws from torch's and numpy's global RNGs; seed both so a run is repeatable.
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    np.random.seed(seed)

    try:
        # SDV >= 1.15: one Metadata class for single- and multi-table data.
        from sdv.metadata import Metadata

        metadata = Metadata.detect_from_dataframes({"expresso": sample})
    except ImportError:
        # Older SDV (removed in 2.0).
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(sample)

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
    )
    try:
        synthesizer.fit({"expresso": sample})   # SDV >= 2.0 takes {table_name: DataFrame}
    except Exception as exc:
        if "dictionary" not in str(exc):
            raise
        synthesizer.fit(sample)                 # older SDV takes the DataFrame
    print(f"    CTGAN training complete  ({time.perf_counter() - t0:.0f}s)")
    return synthesizer


def _generate_ctgan(model: Any, n_rows: int) -> pd.DataFrame:
    print(f"[3/6] Sampling {n_rows:,} rows from CTGAN …")
    t0 = time.perf_counter()
    sampled = model.sample(n_rows)
    # SDV >= 2.0 returns {table_name: DataFrame}; older versions return the DataFrame.
    df: pd.DataFrame = next(iter(sampled.values())) if isinstance(sampled, dict) else sampled
    print(f"    Sampling done  ({time.perf_counter() - t0:.1f}s)")
    return df


# ── Resampling fallback (--no-ctgan) ──────────────────────────────────────────

def _generate_statistical(expresso: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle real Expresso rows — fast, non-parametric, preserves joint distributions.

    No replacement: every output row is a distinct subscriber, so no row can be
    duplicated across the downstream train/holdout split.
    """
    print(f"[3/6] Shuffling {len(expresso):,} Expresso rows (--no-ctgan, no replacement) …")
    t0 = time.perf_counter()
    synth = expresso.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"    Done  ({time.perf_counter() - t0:.1f}s)")
    return synth


# ── Post-processing helpers ────────────────────────────────────────────────────

def _clean_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce CTGAN output to correct dtypes and clip to physically valid ranges."""
    float_cols = [
        "MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE",
        "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO", "ZONE1", "ZONE2",
        "REGULARITY", "FREQ_TOP_PACK",
    ]
    for col in float_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # Domain clipping
    if "REGULARITY" in df.columns:
        df["REGULARITY"] = df["REGULARITY"].clip(1, 90)
    if "FREQUENCE_RECH" in df.columns:
        df["FREQUENCE_RECH"] = df["FREQUENCE_RECH"].clip(0, 200)
    if "FREQUENCE" in df.columns:
        df["FREQUENCE"] = df["FREQUENCE"].clip(0, 150)
    for col in ["ON_NET", "ORANGE", "TIGO", "ZONE1", "ZONE2"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 80_000)
    if "DATA_VOLUME" in df.columns:
        df["DATA_VOLUME"] = df["DATA_VOLUME"].clip(0, 2_000_000)
    if "FREQ_TOP_PACK" in df.columns:
        df["FREQ_TOP_PACK"] = df["FREQ_TOP_PACK"].clip(0, 800)

    return df


def _assign_user_ids(n: int) -> list[str]:
    return [hashlib.sha1(f"maroc_{i}".encode()).hexdigest() for i in range(n)]


def _comonotone_assign(
    source: pd.Series,
    churn: pd.Series | None,
    targets: dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Relabel `source` categories as `targets` while keeping their link to churn.

    Rows are ordered by the churn rate of their source category (ties broken at
    random) and cut into consecutive blocks sized by the target weights. The
    output has exactly the target marginal distribution, and source categories
    with similar churn rates end up in the same target category — so a feature
    that predicted churn in Expresso still predicts it after relabelling.
    Missing source values are treated as their own category. Without CHURN the
    assignment is uniformly random.
    """
    n = len(source)
    names = list(targets)
    weights = np.array(list(targets.values()), dtype=float)
    weights /= weights.sum()

    src = source.astype("object").where(source.notna(), "__MISSING__").astype(str)
    if churn is not None:
        y = pd.to_numeric(churn, errors="coerce").fillna(0).clip(0, 1)
        rate = y.groupby(src).transform("mean").to_numpy(dtype=float)
    else:
        rate = np.zeros(n)

    order = np.lexsort((rng.random(n), rate))   # sort by rate, random tiebreak
    counts = np.floor(weights * n).astype(int)
    counts[0] += n - counts.sum()
    out = np.empty(n, dtype=object)
    out[order] = np.repeat(np.array(names, dtype=object), counts)
    return out


def _assign_region(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    source = (
        df["REGION"] if "REGION" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=object)
    )
    churn = df["CHURN"] if "CHURN" in df.columns else None
    df["REGION"] = _comonotone_assign(source, churn, MOROCCAN_REGIONS, rng)
    return df


def _assign_operator(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    ops   = list(OPERATORS.keys())
    probs = np.array(list(OPERATORS.values()), dtype=float)
    probs /= probs.sum()
    df["OPERATOR"] = rng.choice(ops, size=len(df), p=probs)
    return df


def _calibrate_monetary(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Rescale ARPU_SEGMENT to 25–30 MAD; apply same scale to REVENUE and MONTANT.

    A single linear scale preserves the Expresso-learned correlations between
    monetary columns while moving the distribution to Moroccan MAD units.
    NaN values are preserved so that the downstream MissingIndicatorAdder can
    still add {col}_missing flags (MAR missing pattern).
    Small multiplicative noise (±5 %) prevents a perfectly scaled artefact.
    """
    if "ARPU_SEGMENT" not in df.columns:
        return df

    arpu = pd.to_numeric(df["ARPU_SEGMENT"], errors="coerce")
    pos_mask = arpu.notna() & (arpu > 0)
    nonzero_mean = float(arpu[pos_mask].mean()) if pos_mask.any() else 1.0
    scale = TARGET_ARPU_MAD / nonzero_mean

    noise = rng.normal(1.0, 0.05, size=len(df)).clip(0.85, 1.15)

    for col in ["ARPU_SEGMENT", "REVENUE", "MONTANT"]:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")  # NaN preserved
        pos = vals.notna() & (vals > 0)
        scaled = vals.astype("float64")  # work in float64 to avoid dtype mismatch
        scaled[pos] = (vals[pos].astype("float64") * scale * noise[pos]).clip(0.5, 5_000.0)
        df[col] = scaled.astype("float32")

    return df


def _calibrate_data_volume(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Two-step DATA_VOLUME calibration for Moroccan 4G penetration.

    1. Convert enough zero-data rows to data users until the non-zero fraction
       reaches DATA_USER_RATE (74 %). Converts are drawn with probability
       proportional to REGULARITY + 1: active subscribers are the likeliest
       data adopters, and choosing uniformly at random would inject label noise
       (turning inactive, likely-churning subscribers into data users).
    2. Scale all non-zero DATA_VOLUME by DATA_VOL_SCALE (1.8×) — Morocco's
       higher 4G bandwidth means heavier per-session usage.
    """
    if "DATA_VOLUME" not in df.columns:
        return df

    dv = pd.to_numeric(df["DATA_VOLUME"], errors="coerce").fillna(0.0).to_numpy(float)
    is_user = dv > 0

    current_rate = is_user.mean()
    if current_rate < DATA_USER_RATE:
        non_user_idx = np.where(~is_user)[0]
        n_convert = int((DATA_USER_RATE - current_rate) * len(df))
        n_convert = min(n_convert, len(non_user_idx))
        if n_convert > 0:
            if "REGULARITY" in df.columns:
                reg = pd.to_numeric(df["REGULARITY"], errors="coerce").fillna(0.0)
                activity = reg.to_numpy(float)[non_user_idx] + 1.0
            else:
                activity = np.ones(len(non_user_idx))
            to_convert = rng.choice(
                non_user_idx, size=n_convert, replace=False, p=activity / activity.sum()
            )
            nz = dv[is_user]
            if len(nz) > 0:
                mu  = float(np.log(nz + 1).mean())
                sig = float(np.log(nz + 1).std())
            else:
                mu, sig = 6.0, 1.5
            new_vals = rng.lognormal(mu, sig, n_convert) - 1
            dv[to_convert] = new_vals.clip(1.0, 2_000_000.0)
            is_user = dv > 0

    dv = np.where(is_user, dv * DATA_VOL_SCALE, dv)
    df["DATA_VOLUME"] = dv.clip(0.0, 2_000_000.0).astype("float32")
    return df


def _assign_top_pack(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace Expresso pack names with Moroccan-style pack names.

    Preserves the NaN structure from CTGAN: subscribers with no pack stay NaN.
    Packs are mapped with _comonotone_assign, so the pack–churn association
    learned from Expresso survives the renaming.
    """
    if "TOP_PACK" in df.columns:
        has_pack = df["TOP_PACK"].notna()
        source = df.loc[has_pack, "TOP_PACK"]
    else:
        has_pack = pd.Series(True, index=df.index)
        source = pd.Series(np.nan, index=df.index, dtype=object)
    churn = df.loc[has_pack, "CHURN"] if "CHURN" in df.columns else None

    packs = pd.Series(np.nan, index=df.index, dtype=object)
    if has_pack.any():
        packs.loc[has_pack] = _comonotone_assign(source, churn, dict(MOROCCAN_PACKS), rng)
    df["TOP_PACK"] = packs
    return df


def _assign_mrg(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """MRG (VAS flag): allow ~12 % YES in Moroccan market (Expresso = 0 %)."""
    df["MRG"] = np.where(rng.random(len(df)) < MRG_VAS_RATE, "YES", "NO")
    return df


def _calibrate_churn(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_target: int,
    allow_replacement: bool = True,
) -> pd.DataFrame:
    """Subsample to n_target rows at TARGET_CHURN_RATE (2.7 %).

    Strategy:
      - Take n_churn churners from the pool and fill the remaining slots with
        non-churners.
      - With allow_replacement (CTGAN output, where every row is newly sampled)
        a too-small pool is topped up with replacement — a rare edge case.
      - Without it (real rows, --no-ctgan) n_target is capped at what the pools
        support, so no row is ever duplicated.
    This preserves feature distributions within each class as learned by CTGAN.
    """
    if "CHURN" not in df.columns:
        df["CHURN"] = (rng.random(len(df)) < TARGET_CHURN_RATE).astype("int8")
        return df.sample(n_target, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

    df["CHURN"] = (
        pd.to_numeric(df["CHURN"], errors="coerce")
        .fillna(0).round().clip(0, 1).astype(int)
    )

    churners     = df[df["CHURN"] == 1]
    non_churners = df[df["CHURN"] == 0]

    if not allow_replacement:
        feasible = min(
            n_target,
            int(len(churners) / TARGET_CHURN_RATE),
            int(len(non_churners) / (1 - TARGET_CHURN_RATE)),
        )
        if feasible < n_target:
            print(f"    WARNING: only {feasible:,} rows reachable at "
                  f"{TARGET_CHURN_RATE:.1%} churn without duplicating rows "
                  f"(requested {n_target:,}); output capped.")
            n_target = feasible

    n_churn     = int(n_target * TARGET_CHURN_RATE)
    n_non_churn = n_target - n_churn

    seed_int = int(rng.integers(0, 2**31))
    sampled_pos = churners.sample(
        n_churn, replace=(len(churners) < n_churn), random_state=seed_int
    )
    sampled_neg = non_churners.sample(
        n_non_churn, replace=(len(non_churners) < n_non_churn), random_state=seed_int + 1
    )

    combined = (
        pd.concat([sampled_pos, sampled_neg])
        .sample(frac=1, random_state=seed_int + 2)
        .reset_index(drop=True)
    )
    combined["CHURN"] = combined["CHURN"].astype("int8")
    return combined


def _post_process(
    synth: pd.DataFrame,
    rng: np.random.Generator,
    n_target: int,
    allow_replacement: bool = True,
) -> pd.DataFrame:
    print("[4/6] Applying Moroccan market calibrations …")
    synth = _clean_numerics(synth)
    synth = _calibrate_monetary(synth, rng)
    synth = _calibrate_data_volume(synth, rng)
    synth = _calibrate_churn(synth, rng, n_target, allow_replacement)
    # Relabel after resampling to the target churn rate, so the final file has
    # exactly the target region / operator / pack / MRG shares.
    synth = _assign_region(synth, rng)
    synth = _assign_operator(synth, rng)
    synth = _assign_top_pack(synth, rng)
    synth = _assign_mrg(synth, rng)
    return synth


# ── Validation report ─────────────────────────────────────────────────────────

def _fmt(val: float, pct: bool = False) -> str:
    return f"{val:.3%}" if pct else f"{val:.3f}"


def _validation_report(
    expresso: pd.DataFrame, moroccan: pd.DataFrame
) -> None:
    SEP = "─" * 72
    print(f"\n{SEP}")
    print("  VALIDATION REPORT — Expresso (seed)  vs  Moroccan (synthetic)")
    print(SEP)

    def row(label: str, exp_val: str, mar_val: str, note: str = "") -> None:
        print(f"  {label:<32}  {exp_val:>14}  {mar_val:>14}  {note}")

    row("Metric", "Expresso", "Moroccan", "Check")
    print(f"  {'─'*32}  {'─'*14}  {'─'*14}  {'─'*18}")
    row("Rows", f"{len(expresso):,}", f"{len(moroccan):,}")

    # Churn rate
    e_churn = expresso["CHURN"].mean() if "CHURN" in expresso.columns else float("nan")
    m_churn = moroccan["CHURN"].mean()  if "CHURN" in moroccan.columns  else float("nan")
    churn_ok = abs(m_churn - TARGET_CHURN_RATE) < 0.003
    row("Churn rate", _fmt(e_churn, True), _fmt(m_churn, True),
        "PASS ✓" if churn_ok else f"WARN (target {TARGET_CHURN_RATE:.1%})")

    # Monetary columns — use non-NaN mean (ARPU of active subscribers)
    print()
    for col, target_label, lo, hi in [
        ("ARPU_SEGMENT", "25–30 MAD", 25.0, 30.0),
        ("REVENUE",      "–",         None,  None),
        ("MONTANT",      "–",         None,  None),
    ]:
        if col in moroccan.columns:
            e_m = expresso[col].dropna().mean() if col in expresso.columns else float("nan")
            m_m = moroccan[col].dropna().mean()
            m_s = moroccan[col].dropna().std()
            note = ""
            if lo is not None:
                note = "PASS ✓" if lo <= m_m <= hi else f"WARN (target {target_label}, got {m_m:.1f})"
            row(f"{col} non-null mean (std)",
                f"{e_m:.1f}",
                f"{m_m:.1f} (±{m_s:.1f})",
                note)

    # DATA_VOLUME usage rate
    if "DATA_VOLUME" in moroccan.columns:
        print()
        e_dv = (expresso["DATA_VOLUME"] > 0).mean() if "DATA_VOLUME" in expresso.columns else float("nan")
        m_dv = (moroccan["DATA_VOLUME"] > 0).mean()
        dv_ok = m_dv >= 0.70
        row("DATA_VOLUME > 0 rate",
            _fmt(e_dv, True), _fmt(m_dv, True),
            "PASS ✓" if dv_ok else "WARN (target ≥ 70 %)")

    # Operator distribution
    if "OPERATOR" in moroccan.columns:
        print()
        print("  Operator distribution (Moroccan):")
        op_dist = moroccan["OPERATOR"].value_counts(normalize=True).sort_index()
        for op in sorted(OPERATORS):
            got = op_dist.get(op, 0.0)
            tgt = OPERATORS[op]
            note = "PASS ✓" if abs(got - tgt) < 0.015 else "WARN"
            print(f"    {op:<20} got={got:.3%}  target={tgt:.3%}  {note}")

    # Region top-5
    if "REGION" in moroccan.columns:
        print()
        print("  Top Moroccan regions:")
        for reg, share in moroccan["REGION"].value_counts(normalize=True).head(5).items():
            tgt = MOROCCAN_REGIONS.get(str(reg), 0.0)
            print(f"    {str(reg):<20} {share:.3%}  (target {tgt:.3%})")

    # TOP_PACK top-5
    if "TOP_PACK" in moroccan.columns:
        print()
        print("  Top 5 packs (Moroccan):")
        for pack, cnt in moroccan["TOP_PACK"].value_counts().head(5).items():
            print(f"    {str(pack):<42} {cnt:>10,}")

    # MRG distribution
    if "MRG" in moroccan.columns:
        print()
        print("  MRG distribution:")
        for val, share in moroccan["MRG"].value_counts(normalize=True).items():
            note = "PASS ✓" if abs(share - (1 - MRG_VAS_RATE if val == "NO" else MRG_VAS_RATE)) < 0.02 else ""
            print(f"    {str(val):<8} {share:.3%}  {note}")

    print(f"{SEP}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate synthetic Moroccan prepaid telecom churn dataset"
    )
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--sample-size", type=int,  default=50_000,
                   help="Expresso rows used to train CTGAN (default 50 000)")
    p.add_argument("--n-rows",      type=int,  default=2_000_000,
                   help="Target output rows (default 2 000 000)")
    p.add_argument("--epochs",      type=int,  default=300,
                   help="CTGAN training epochs (default 300)")
    p.add_argument("--batch-size",  type=int,  default=500,
                   help="CTGAN mini-batch size (default 500)")
    p.add_argument("--no-ctgan",    action="store_true",
                   help="Skip CTGAN; resample full Expresso without replacement (fast)")
    p.add_argument("--output",      type=Path,
                   default=ROOT / "data" / "raw" / "moroccan_telecom_churn.csv")
    args = p.parse_args()

    t_total = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    n_target = args.n_rows

    # Generate 20 % extra rows so churn subsampling never runs out of positives/negatives
    n_generate = int(n_target * 1.25)

    # ── 1–3. Seed data and synthetic rows ─────────────────────────────────────
    if args.no_ctgan:
        expresso = _load_expresso_sample(None, args.seed)
        print("[2/6] Skipping CTGAN (--no-ctgan)")
        synth = _generate_statistical(expresso, args.seed)
    else:
        expresso_sample = _load_expresso_sample(args.sample_size, args.seed)
        model = _train_ctgan(
            expresso_sample, args.epochs, args.batch_size, args.seed
        )
        synth = _generate_ctgan(model, n_generate)

    print(f"    Raw synthetic rows: {len(synth):,}  "
          f"columns: {list(synth.columns)}")

    # ── 4. Post-process ────────────────────────────────────────────────────────
    synth = _post_process(synth, rng, n_target, allow_replacement=not args.no_ctgan)

    # ── 5. Add user IDs and reorder columns ───────────────────────────────────
    synth.insert(0, "user_id", _assign_user_ids(len(synth)))
    present = [c for c in OUTPUT_COLS if c in synth.columns]
    extras  = [c for c in synth.columns if c not in OUTPUT_COLS]
    synth   = synth[present + extras]

    # ── 6. Save ────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[5/6] Saving {len(synth):,} rows → {args.output} …")
    t_save = time.perf_counter()
    synth.to_csv(args.output, index=False)
    size_mb = args.output.stat().st_size / 1e6
    print(f"    Saved  {size_mb:.1f} MB  ({time.perf_counter() - t_save:.1f}s)")

    # ── 7. Validation report ───────────────────────────────────────────────────
    print("[6/6] Generating validation report …")
    report_cols = ["ARPU_SEGMENT", "REVENUE", "MONTANT", "DATA_VOLUME", "CHURN"]
    expresso_ref = pd.read_csv(
        ROOT / "data" / "raw" / "expresso" / "Train.csv",
        usecols=[c for c in report_cols if c in _EXPRESSO_COLS],
        low_memory=False,
        nrows=300_000,  # sample for speed
    )
    _validation_report(expresso_ref, synth)

    elapsed = time.perf_counter() - t_total
    print(f"Done in {elapsed:.0f}s.  Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
