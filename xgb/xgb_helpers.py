from typing import List, Tuple

import numpy as np, pandas as pd, xgboost as xgb
from rich.progress import Progress, BarColumn, TimeElapsedColumn
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost.callback import TrainingCallback

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    return df

def split_numeric_categorical(
    df: pd.DataFrame, features: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Split a list of columns into numeric vs. categorical based on dtype.

    Parameters
    ----------
    df
        Your full DataFrame.
    features
        List of column names to split.

    Returns
    -------
    numeric_feats
        Columns whose dtype is a subclass of np.number (int, float, etc.).
    categorical_feats
        The remaining columns (object, category, datetime, etc.).
    """
    numeric_feats = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = [col for col in features if col not in numeric_feats]

    return numeric_feats, categorical_feats


def build_dmatrix(
    df: pd.DataFrame, y_col, base_margin=None, weight=None
) -> xgb.DMatrix:
    """
    Builds an XGBoost DMatrix
    """
    # features
    X = df

    # build DMatrix
    dmatrix = xgb.DMatrix(
        data=X,
        label=y_col,
        base_margin=base_margin,
        weight=weight,
    )
    return dmatrix


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for ordinal encoding with rare and missing value handling.
    """

    def __init__(
        self,
        columns: list[str],
        min_support: int = 5,
        other_category: bool = True,
        random_scale: bool = True,
        seed: int = 1234,
        offset: int = 0,
        method: str = "freq",
        verbose: bool = False,
    ):
        self.columns = columns
        self.min_support = min_support
        self.other_category = other_category
        self.random_scale = random_scale
        self.seed = seed
        self.offset = offset
        self.method = method
        self.verbose = verbose

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self._columns
        return np.array(input_features, dtype=object)

    def fit(self, X: pd.DataFrame, y=None):
        rng = np.random.RandomState(self.seed)
        self.mappings_: dict[str, dict] = {}

        for col in self.columns:
            s = X[col]
            nonnull = s.dropna()
            counts = nonnull.value_counts()
            eligible = counts[counts >= self.min_support].index.tolist()

            # Determine ordering
            if self.method == "freq":
                levels = counts.loc[eligible].sort_values().index.tolist()
            elif self.method == "lex":
                levels = sorted(eligible)
            elif self.method in ("random", "None") and self.random_scale:
                levels = eligible.copy()
                rng.shuffle(levels)
            elif self.method in ("random", "None") and not self.random_scale:
                levels = sorted(eligible)
            else:
                raise ValueError(f"Unsupported method='{self.method}'")

            lvl_to_code = {lvl: i + self.offset for i, lvl in enumerate(levels)}
            self.mappings_[col] = lvl_to_code

            if self.verbose:
                print(f"[OrdinalEncoder2] {col}: {lvl_to_code}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.columns:
            mapping = self.mappings_[col]
            s = X_out[col]
            is_na = s.isna()

            codes = s.map(mapping).astype(float)  # unknowns become NaN
            codes[is_na] = -2  # missing

            unknown_mask = (~s.isin(mapping)) & (~is_na)
            codes[unknown_mask] = -1 if self.other_category else -2

            X_out[col] = codes.astype(int)

        return X_out


class NumericImputerArbitrary(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer for numeric imputation with some optional monotonicity stuff

    Features:
    - Imputes missing values using either an arbitrary constant or median.
    - Adds 'missingness' indicators when median is used.
    - Scales variables in [0, 1] if their range is between 0 and 0.1.
    - Skips scaling or median for monotonic features.
    """

    def __init__(
        self,
        threshold: int = 10,
        min_count_na: int = 5,
        arbimp: float = -9999.0,
        scale_small: bool = True,
        mono_up=None,  # Avoid mutable default
        mono_down=None,  # Avoid mutable default
        verbose: bool = True,
    ):
        # Store arguments exactly as passed, to keep sklearn.clone() happy
        self.threshold = threshold
        self.min_count_na = min_count_na
        self.arbimp = arbimp
        self.scale_small = scale_small
        self.mono_up = mono_up
        self.mono_down = mono_down
        self.verbose = verbose

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self._columns
        return np.array(input_features, dtype=object)

    def fit(self, X: pd.DataFrame, y=None):
        # Convert mono lists to empty lists if None (done here, not in __init__)
        mono_up = self.mono_up if self.mono_up is not None else []
        mono_down = self.mono_down if self.mono_down is not None else []

        self.impute_values_ = {}
        self.scaling_params_ = {}

        for col in X.select_dtypes(include=[np.number]):
            arr = X[col]
            finite = arr.dropna()[np.isfinite(arr.dropna())]
            n_finite = finite.shape[0]
            n_na = arr.isna().sum()

            # Arbitrary imputation is used if enough data + missingness + not monotonic
            use_arb = (
                n_finite > self.threshold
                and n_na >= self.min_count_na
                and col not in mono_up
                and col not in mono_down
            )

            if use_arb:
                self.impute_values_[col] = self.arbimp
                self.scaling_params_[col] = (
                    None  # skip scaling for arbitrarily imputed values
                )
            else:
                median = finite.median()
                self.impute_values_[col] = median

                if self.scale_small:
                    if not finite.empty and finite.min() >= 0 and finite.max() <= 0.1:
                        self.scaling_params_[col] = (finite.min(), finite.max())
                    else:
                        self.scaling_params_[col] = None

            if self.verbose:
                strategy = "arb" if use_arb else "median"
                print(
                    f"[NumericImputerArbitrary] {col}: {strategy}, scale={self.scaling_params_[col]}"
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy().astype(float)

        for col, impval in self.impute_values_.items():
            arr = X_out[col]
            was_na = arr.isna()
            X_out[col] = arr.fillna(impval)

            # Add indicator if median-imputed and NA existed
            if impval != self.arbimp and was_na.any() and arr.nunique(dropna=True) >= 1:
                X_out[f"{col}-mi"] = was_na.astype(int)

            # Scale small values between 0 and 1, only for median-imputed
            params = self.scaling_params_.get(col)
            if self.scale_small and params is not None:
                mn, mx = params
                mask = X_out[col] != self.arbimp
                if mx > mn:
                    X_out.loc[mask, col] = (X_out.loc[mask, col] - mn) / (mx - mn)

        return X_out


class RichProgressBarCallback(TrainingCallback):
    def __init__(self, total_rounds: int):
        self.total_rounds = total_rounds
        self.progress = Progress(
            "[progress.percentage]{task.percentage:>3.0f}%",
            BarColumn(bar_width=None),
            "•",
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
            transient=True,
        )
        self.task_id = None

    def before_training(self, model):
        self.progress.start()
        self.task_id = self.progress.add_task("Training", total=self.total_rounds)
        return model

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        self.progress.update(self.task_id, advance=1)
        return False  # return True to stop training early

    def after_training(self, model):
        self.progress.stop()
        return model


# def patch_skopt_for_numpy():
#     import numpy as np
#     # import skopt.space.transformers

#     class PatchedNormalize(skopt.space.transformers.Normalize):
#         def transform(self, X):
#             if self.is_int:
#                 return (np.round(X).astype(int) - self.low) / (self.high - self.low)
#             return (X - self.low) / (self.high - self.low)

#         def inverse_transform(self, X):
#             X_orig = X * (self.high - self.low) + self.low
#             if self.is_int:
#                 return np.round(X_orig).astype(int)
#             return X_orig

#     skopt.space.transformers.Normalize = PatchedNormalize
