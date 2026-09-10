from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Diagnosis"
ID_COL = "ID"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).strip().replace(" ", "_").replace("(", "").replace(")", "")
        for c in out.columns
    ]
    return out


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_column_names(df)
    return df


def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Normalize text labels while keeping the original meaning.
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip()

    # Convert obvious numeric-looking columns when possible.
    for col in out.columns:
        if col not in ["Diagnosis"]:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() >= 0.9 * len(out):
                out[col] = converted

    return out


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features to training-set IQR bounds.

    Bounds are learned only from the training split, avoiding test-set leakage.
    """

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        self.feature_names_in_ = X_df.columns.to_numpy()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        return X_df.clip(self.lower_, self.upper_, axis=1).to_numpy()


def prepare_xy(df: pd.DataFrame):
    df = normalize_values(df)

    # Remove exact duplicate rows.
    df = df.drop_duplicates().copy()

    # ID is an identifier and should not be used as a predictor.
    feature_cols = [c for c in df.columns if c not in [TARGET, ID_COL]]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    # Treat Gender and Smoking_Status as categorical variables.
    categorical = [c for c in feature_cols if c.lower() in {"gender", "smoking_status"}]
    numeric = [c for c in feature_cols if c not in categorical]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("outlier_clip", IQRClipper()),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )

    return X, y, preprocessor
