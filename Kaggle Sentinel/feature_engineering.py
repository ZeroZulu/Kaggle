"""
Kaggle Sentinel — Feature Engineering Pipeline
Reusable module for both notebook and Streamlit dashboard.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Behavioral DNA feature engineering to the dataset."""
    df = df.copy()

    # Engagement metrics
    df['ENGAGEMENT_RATIO'] = (df['FOLLOWER_COUNT'] + 1) / (df['FOLLOWING_COUNT'] + 1)
    df['FOLLOW_RECIPROCITY'] = df[['FOLLOWER_COUNT', 'FOLLOWING_COUNT']].min(axis=1) / \
                                (df[['FOLLOWER_COUNT', 'FOLLOWING_COUNT']].max(axis=1) + 1)
    df['SOCIAL_REACH'] = np.sqrt(df['FOLLOWER_COUNT'] * df['FOLLOWING_COUNT'])

    # Content production
    df['TOTAL_CONTENT'] = df['DATASET_COUNT'] + df['CODE_COUNT'] + df['DISCUSSION_COUNT']
    df['CONTENT_PER_DISCUSSION'] = df[['DATASET_COUNT', 'CODE_COUNT']].sum(axis=1) / (df['DISCUSSION_COUNT'] + 1)
    df['ACTIVITY_SCORE'] = (df['FOLLOWER_COUNT'] + df['FOLLOWING_COUNT'] +
                            df['DATASET_COUNT'] * 3 + df['CODE_COUNT'] * 3 +
                            df['DISCUSSION_COUNT'] * 2 + df['AVG_NB_READ_TIME_MIN'])
    df['IS_DORMANT'] = (df['ACTIVITY_SCORE'] < 2).astype(int)

    # Reading behavior
    df['READ_PER_DISCUSSION'] = df['AVG_NB_READ_TIME_MIN'] / (df['DISCUSSION_COUNT'] + 1)
    df['READ_ENGAGEMENT'] = df['AVG_NB_READ_TIME_MIN'] * df['DISCUSSION_COUNT']
    df['HAS_READ_TIME'] = (df['AVG_NB_READ_TIME_MIN'] > 0).astype(int)

    # Voting behavior
    df['VOTE_TOTAL'] = df['TOTAL_VOTES_GAVE_NB'] + df['TOTAL_VOTES_GAVE_DS'] + df['TOTAL_VOTES_GAVE_DC']
    vote_cols = ['TOTAL_VOTES_GAVE_NB', 'TOTAL_VOTES_GAVE_DS', 'TOTAL_VOTES_GAVE_DC']
    vote_array = df[vote_cols].values.astype(float)
    vote_sums = vote_array.sum(axis=1, keepdims=True)
    vote_probs = np.divide(vote_array, vote_sums, where=vote_sums > 0, out=np.zeros_like(vote_array))
    df['VOTE_ENTROPY'] = np.array([sp_entropy(row) if row.sum() > 0 else 0 for row in vote_probs])
    df['VOTE_NB_RATIO'] = df['TOTAL_VOTES_GAVE_NB'] / (df['VOTE_TOTAL'] + 1)
    df['VOTE_CONCENTRATION'] = df[vote_cols].max(axis=1) / (df['VOTE_TOTAL'] + 1)

    # Composite
    df['PHANTOM_SCORE'] = df['VOTE_TOTAL'] / (df['ACTIVITY_SCORE'] + 1)
    df['AUTHENTICITY_INDEX'] = (df['SOCIAL_REACH'] * df['TOTAL_CONTENT'] *
                                df['AVG_NB_READ_TIME_MIN']) / (df['VOTE_TOTAL'] + 1)

    return df


def benfords_expected():
    """Return expected Benford distribution for digits 1-9."""
    return {d: np.log10(1 + 1 / d) for d in range(1, 10)}


def first_digit_distribution(series):
    """Extract first-digit frequency distribution."""
    digits = series.dropna().astype(int).abs()
    digits = digits[digits > 0]
    first_digits = digits.apply(lambda x: int(str(x)[0]))
    counts = first_digits.value_counts().sort_index()
    total = len(first_digits)
    return {d: counts.get(d, 0) / total for d in range(1, 10)}, total


def benford_analysis(series, label=""):
    """Run full Benford's Law analysis on a series."""
    from scipy.stats import chisquare
    from scipy.spatial.distance import jensenshannon

    dist, n = first_digit_distribution(series)
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
        'label': label,
        'dist': dist,
        'n': n,
        'chi2': chi2,
        'pval': pval,
        'js_divergence': js_div,
        'verdict': 'EXTREME' if js_div > 0.3 else 'MODERATE' if js_div > 0.1 else 'NORMAL'
    }
