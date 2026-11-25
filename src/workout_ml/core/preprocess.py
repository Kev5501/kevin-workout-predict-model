"""Column selection and the three preprocessing recipes."""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class ColumnSelector(TransformerMixin, BaseEstimator):
    """Keep only the model's feature columns (drops 'date' and everything unselected)."""

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f'Missing feature columns: {missing}')
        return self

    def transform(self, X):
        return X[list(self.columns)].copy()


class NativeCategorical(TransformerMixin, BaseEstimator):
    """Categories fixed at fit time so train and predict match.

    ``mode='category'``: pandas category dtype, unseen/missing -> NaN (LightGBM, XGBoost).
    ``mode='string'``: plain strings with 'NA' for missing/unseen (CatBoost).
    """

    def __init__(self, cat_cols=(), mode='category'):
        self.cat_cols = cat_cols
        self.mode = mode

    def fit(self, X, y=None):
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in self.cat_cols}
        return self

    def transform(self, X):
        X = X.copy()
        for c in X.columns:
            if c in self.categories_:
                vals = X[c].astype(object).where(X[c].notna(), None)
                known = set(self.categories_[c])
                vals = vals.map(lambda v, known=known: str(v) if v is not None and str(v) in known else None)
                if self.mode == 'category':
                    X[c] = pd.Categorical(vals, categories=self.categories_[c])   # unseen -> NaN
                else:
                    X[c] = vals.fillna('NA').astype(object)
            else:
                X[c] = pd.to_numeric(X[c], errors='coerce').astype(float)
        return X


def _num_cat(features, categorical):
    cats = [c for c in features if c in categorical]
    nums = [c for c in features if c not in cats]
    return nums, cats


def build_preprocessor(kind: str, features: list[str], categorical: list[str]):
    nums, cats = _num_cat(features, categorical)
    if kind == 'native':
        return NativeCategorical(cat_cols=cats, mode='category')
    if kind == 'native_string':
        return NativeCategorical(cat_cols=cats, mode='string')
    if kind == 'ordinal':
        return ColumnTransformer(
            [('cat', Pipeline([
                ('str', NativeCategorical(cat_cols=cats, mode='string')),
                ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                       encoded_missing_value=-1)),
            ]), cats)] if cats else [],
            remainder='passthrough', verbose_feature_names_out=False)
    if kind == 'linear':
        transformers = [('num', Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=True)),
            ('scale', StandardScaler()),
        ]), nums)]
        if cats:
            transformers.append(('cat', Pipeline([
                ('str', NativeCategorical(cat_cols=cats, mode='string')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), cats))
        return ColumnTransformer(transformers, verbose_feature_names_out=False)
    if kind == 'none':
        return 'passthrough'
    raise ValueError(f'Unknown preprocessor kind {kind!r}')


def make_pipeline(features: list[str], categorical: list[str], kind: str, estimator) -> Pipeline:
    return Pipeline([
        ('select', ColumnSelector(columns=list(features))),
        ('prep', build_preprocessor(kind, features, categorical)),
        ('model', estimator),
    ])
