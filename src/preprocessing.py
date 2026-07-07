"""
preprocessing.py — Data loading and feature engineering helpers for the ACR'25 study.

Functions
---------
find_duplicate_columns : Find pairs of value-identical columns in a DataFrame.
build_kdd_preprocessor : Return a fitted ColumnTransformer for the KDD dataset.
build_cc_preprocessor  : Return a fitted ColumnTransformer for the Credit Card dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def find_duplicate_columns(df: pd.DataFrame) -> list:
    """
    Return pairs of columns whose values are identical across all rows.

    Uses a two-stage filter (same dtype + same nunique) before the expensive
    .equals() check to keep runtime manageable on wide DataFrames.

    Parameters
    ----------
    df : DataFrame to inspect.

    Returns
    -------
    List of (col_a, col_b) tuples.
    """
    cols = df.columns.tolist()
    candidates = [
        (c1, c2)
        for i, c1 in enumerate(cols)
        for c2 in cols[i + 1:]
        if df[c1].dtype == df[c2].dtype and df[c1].nunique() == df[c2].nunique()
    ]
    return [(c1, c2) for c1, c2 in candidates if df[c1].equals(df[c2])]


def build_kdd_preprocessor(X_train: pd.DataFrame,
                            num_cols: list,
                            cat_cols: list) -> ColumnTransformer:
    """
    Build and fit a ColumnTransformer for the KDD dataset on training data only.

    Numerical columns are standardised with StandardScaler.
    Categorical columns are one-hot encoded (drop='first', handle_unknown='ignore').

    Parameters
    ----------
    X_train   : Training feature DataFrame (after split).
    num_cols  : List of numerical column names.
    cat_cols  : List of categorical column names.

    Returns
    -------
    Fitted ColumnTransformer. Apply .transform() to test data.
    """
    prep = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), num_cols),
            (
                "ohe",
                OneHotEncoder(
                    sparse_output=False,
                    drop="first",
                    handle_unknown="ignore",
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    prep.fit(X_train)
    return prep


def build_cc_preprocessor(X_train: pd.DataFrame,
                           feature_cols: list) -> ColumnTransformer:
    """
    Build and fit a ColumnTransformer for the Credit Card dataset on training data only.

    All features are standardised with StandardScaler (V1–V28 are already PCA
    components; Time and Amount are scaled for model compatibility).

    Parameters
    ----------
    X_train      : Training feature DataFrame (after split).
    feature_cols : All feature column names (everything except 'Class').

    Returns
    -------
    Fitted ColumnTransformer. Apply .transform() to test data.
    """
    prep = ColumnTransformer(
        transformers=[("scale", StandardScaler(), feature_cols)]
    )
    prep.fit(X_train)
    return prep
