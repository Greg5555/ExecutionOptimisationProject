CUSTOM_TRANSFO
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from collections import Counter


# custom_utils.py
import pandas as pd
import numpy as np

def cast_float32(X):
    if isinstance(X, pd.DataFrame):
        num_cols = X.select_dtypes(include=[np.number]).columns
        X = X.copy()
        X[num_cols] = X[num_cols].astype(np.float32)
        return X
    else:
        return X.astype(np.float32)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Réduit la cardinalité par colonne :
    - garde les catégories fréquentes (>= min_freq),
    - si > (max_categories - 1), tronque aux top (max_categories - 1),
    - mappe le reste vers other_token.
    """
    def __init__(self, max_categories=255, min_freq=20, other_token="__OTHER__"):
        self.max_categories = max_categories
        self.min_freq = min_freq
        self.other_token = other_token
        self.kept_per_col_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        n_cols = X.shape[1]
        kept = []
        for j in range(n_cols):
            col = X[:, j]
            cnt = Counter(col)
            keep = [v for v, f in cnt.most_common() if f >= self.min_freq]
            if len(keep) > self.max_categories - 1:
                keep = keep[: self.max_categories - 1]
            kept.append(set(keep))
        self.kept_per_col_ = kept

        self.category_maps_ = self.kept_per_col_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object).copy()
        kept = getattr(self, "kept_per_col_", None)
        if kept is None:
            kept = getattr(self, "category_maps_", None)
        if kept is None:
            raise AttributeError(
                "RareCategoryGrouper has no kept_per_col_ nor category_maps_. "
            )

        for j, keep in enumerate(kept):
            col = X[:, j]
            mask = ~np.isin(col, list(keep))
            if mask.any():
                col = col.copy()
                col[mask] = self.other_token
                X[:, j] = col
        return X
