"""
Kaggle Sentinel — Feature Engineering Pipeline (Optimized)
Vectorized for 1M+ row datasets.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Behavioral DNA feature engineering — fully vectorized."""
    df = df.copy()

    fc = df['FOLLOWER_COUNT'].values.astype(np.float64)
    foc = df['FOLLOWING_COUNT'].values.astype(np.float64)
    dc = df['DISCUSSION_COUNT'].values.astype(np.float64)
    dsc = df['DATASET_COUNT'].values.astype(np.float64)
    cc = df['CODE_COUNT'].values.astype(np.float64)
    rt = df['AVG_NB_READ_TIME_MIN'].values.astype(np.float64)
    vn = df['TOTAL_VOTES_GAVE_NB'].values.astype(np.float64)
    vd = df['TOTAL_VOTES_GAVE_DS'].values.astype(np.float64)
    vdc = df['TOTAL_VOTES_GAVE_DC'].values.astype(np.float64)

    # Engagement
    df['ENGAGEMENT_RATIO'] = (fc + 1) / (foc + 1)
    df['FOLLOW_RECIPROCITY'] = np.minimum(fc, foc) / (np.maximum(fc, foc) + 1)
    df['SOCIAL_REACH'] = np.sqrt(fc * foc)

    # Content
    df['TOTAL_CONTENT'] = dsc + cc + dc
    df['CONTENT_PER_DISCUSSION'] = (dsc + cc) / (dc + 1)
    activity = fc + foc + dsc * 3 + cc * 3 + dc * 2 + rt
    df['ACTIVITY_SCORE'] = activity
    df['IS_DORMANT'] = (activity < 2).astype(np.int8)

    # Reading
    df['READ_PER_DISCUSSION'] = rt / (dc + 1)
    df['READ_ENGAGEMENT'] = rt * dc
    df['HAS_READ_TIME'] = (rt > 0).astype(np.int8)

    # Voting — fully vectorized entropy (no Python loop)
    vote_array = np.column_stack([vn, vd, vdc])
    vote_total = vote_array.sum(axis=1)
    df['VOTE_TOTAL'] = vote_total

    safe_total = np.where(vote_total > 0, vote_total, 1.0)
    probs = vote_array / safe_total[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_probs = np.where(probs > 0, np.log(probs), 0.0)
    ent = -np.sum(probs * log_probs, axis=1)
    ent[vote_total == 0] = 0.0
    df['VOTE_ENTROPY'] = ent

    df['VOTE_NB_RATIO'] = vn / (vote_total + 1)
    df['VOTE_CONCENTRATION'] = vote_array.max(axis=1) / (vote_total + 1)

    # Composite
    df['PHANTOM_SCORE'] = vote_total / (activity + 1)
    df['AUTHENTICITY_INDEX'] = (df['SOCIAL_REACH'].values * (dsc + cc + dc) * rt) / (vote_total + 1)

    return df


def benfords_expected():
    return {d: np.log10(1 + 1 / d) for d in range(1, 10)}


def first_digit_distribution(series):
    digits = series.dropna().astype(int).abs()
    digits = digits[digits > 0]
    first_digits = digits.astype(str).str[0].astype(int)
    counts = first_digits.value_counts().sort_index()
    total = len(first_digits)
    return {d: counts.get(d, 0) / total for d in range(1, 10)}, total


def benford_analysis(series, label=""):
    from scipy.stats import chisquare
    from scipy.spatial.distance import jensenshannon

    dist, n = first_digit_distribution(series)
    if n == 0:
        return {'label': label, 'dist': dist, 'n': 0, 'chi2': 0, 'pval': 1,
                'js_divergence': 0, 'verdict': 'NO DATA'}

    expected = benfords_expected()
    observed_counts = np.array([dist.get(d, 0) * n for d in range(1, 10)])
    expected_counts = np.array([expected[d] * n for d in range(1, 10)])
    chi2, pval = chisquare(observed_counts, expected_counts)

    p = np.array([dist.get(d, 0) for d in range(1, 10)])
    q = np.array([expected[d] for d in range(1, 10)])
    p = p / (p.sum() + 1e-10)
    q = q / q.sum()
    js_div = jensenshannon(p, q)

    return {
        'label': label, 'dist': dist, 'n': n, 'chi2': chi2, 'pval': pval,
        'js_divergence': js_div,
        'verdict': 'EXTREME' if js_div > 0.3 else 'MODERATE' if js_div > 0.1 else 'NORMAL'
    }
