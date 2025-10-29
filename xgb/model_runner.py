"""Utilities to run XGBoost market or frequency models from the shared configuration."""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

import modin.pandas as md  # type: ignore
import snowflake.snowpark.modin.plugin  # noqa: F401
from snowflake.snowpark.context import get_active_session

from xgb_helpers import (
    NumericImputerArbitrary,
    OrdinalEncoder,
    RichProgressBarCallback,
    build_dmatrix,
    build_monotone_constraints,
    poisson_deviance,
    split_numeric_categorical,
    weighted_gini_norm,
)

CONFIG_FILE = "config.ini"


def _normalize_config_entry(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in {"", "none"}:
        return None
    return stripped


def _parse_list(value: Optional[str]) -> List[str]:
    normalized = _normalize_config_entry(value)
    if not normalized:
        return []
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _parse_params(section: configparser.SectionProxy) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for key, value in section.items():
        normalized = value.strip()
        if normalized.lower() in {"true", "false"}:
            params[key] = normalized.lower() == "true"
            continue
        try:
            params[key] = int(normalized)
            continue
        except ValueError:
            pass
        try:
            params[key] = float(normalized)
            continue
        except ValueError:
            pass
        params[key] = normalized
    return params


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c and c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y: pd.Series
    feature_names: List[str]
    identifiers: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class FrequencyExtras:
    exposure: str
    weight: Optional[str]
    offset: Optional[str]
    partition: str
    train_partition_values: Tuple[int, ...]
    valid_partition_values: Tuple[int, ...]


def _read_data(config: configparser.ConfigParser) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = config["PATHS"]
    train_path = _normalize_config_entry(paths.get("train"))
    test_path = _normalize_config_entry(paths.get("test"))
    if not train_path or not test_path:
        raise ValueError("Both train and test Snowflake objects must be defined in config.ini")

    df_train = md.read_snowflake(name_or_query=train_path)
    df_test = md.read_snowflake(name_or_query=test_path)
    return df_train.to_pandas(), df_test.to_pandas()


def _prepare_features(
    config: configparser.ConfigParser,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> PreparedData:
    variables = config["VARIABLES"]
    target_column = _normalize_config_entry(variables.get("y_col"))
    if not target_column:
        raise ValueError("A target column must be provided via [VARIABLES] y_col in config.ini")

    identifier_cols = _parse_list(variables.get("identifier_cols"))
    drop_cols = set(_parse_list(variables.get("drop_cols")))
    modelling_features = _parse_list(variables.get("modelling_features"))
    if not modelling_features:
        raise ValueError(
            "No modelling features were provided via [VARIABLES] modelling_features in config.ini"
        )

    df_train = df_train.copy()
    df_test = df_test.copy()

    for col in drop_cols:
        if col in df_train.columns:
            df_train.drop(columns=col, inplace=True)
        if col in df_test.columns:
            df_test.drop(columns=col, inplace=True)

    if target_column not in df_train.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the training dataset")

    valid_features = [feature for feature in modelling_features if feature in df_train.columns]
    if target_column in valid_features:
        valid_features.remove(target_column)
    if not valid_features:
        raise ValueError("None of the configured modelling features are present in the training set")

    identifier_cols_in_test = [c for c in identifier_cols if c in df_test.columns]
    identifiers = df_test.loc[:, identifier_cols_in_test].copy()

    X_train_raw = df_train.loc[:, valid_features].copy()
    X_test_raw = df_test.loc[:, [feature for feature in valid_features if feature in df_test.columns]].copy()

    missing_test_features = [feature for feature in valid_features if feature not in X_test_raw.columns]
    for column in missing_test_features:
        X_test_raw[column] = np.nan
    X_test_raw = X_test_raw.reindex(columns=valid_features)

    num_feats, cat_feats = split_numeric_categorical(X_train_raw, valid_features)

    preprocessor = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(columns=cat_feats, method="freq"), cat_feats),
            ("num", NumericImputerArbitrary(), num_feats),
        ],
        remainder="drop",
    )

    X_train_transformed = preprocessor.fit_transform(X_train_raw)
    X_test_transformed = preprocessor.transform(X_test_raw)

    feature_names = num_feats + cat_feats
    X_train = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test_raw.index)

    y = df_train[target_column].copy()

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y=y,
        feature_names=feature_names,
        identifiers=identifiers,
        train_df=df_train,
        test_df=df_test,
    )


def _parse_frequency_extras(config: configparser.ConfigParser) -> FrequencyExtras:
    variables = config["VARIABLES"]
    exposure = _normalize_config_entry(variables.get("exposure_col"))
    partition = _normalize_config_entry(variables.get("partition_col"))
    if not exposure:
        raise ValueError("[VARIABLES] exposure_col must be set for frequency models")
    if not partition:
        raise ValueError("[VARIABLES] partition_col must be set for frequency models")

    weight = _normalize_config_entry(variables.get("weight_col"))
    offset = _normalize_config_entry(variables.get("offset_col"))

    train_partitions = _parse_list(variables.get("train_partition_values"))
    valid_partitions = _parse_list(variables.get("valid_partition_values"))

    def _convert(parts: List[str], default: Tuple[int, ...]) -> Tuple[int, ...]:
        if not parts:
            return default
        try:
            return tuple(int(p) for p in parts)
        except ValueError as exc:
            raise ValueError("Partition values must be integers") from exc

    return FrequencyExtras(
        exposure=exposure,
        weight=weight,
        offset=offset,
        partition=partition,
        train_partition_values=_convert(train_partitions, (1,)),
        valid_partition_values=_convert(valid_partitions, (2,)),
    )


def _save_predictions(
    session,
    predictions: pd.DataFrame,
    config: configparser.ConfigParser,
) -> str:
    output_path = config["PATHS"].get("output_predictions_csv", "xgb_test_predictions.csv")
    predictions.to_csv(output_path, index=False)
    if session is not None:
        session.write_pandas(
            predictions,
            "snowpark_jc",
            auto_create_table=True,
            table_type="transient",
        )
    return output_path


def _run_market_model(
    prepared: PreparedData,
    config: configparser.ConfigParser,
    session,
) -> Dict[str, object]:
    params_section = config["PARAMS"]
    default_params = _parse_params(params_section)

    search_space = [
        Integer(4, 7, name="max_depth"),
        Real(0.1, 0.2, prior="log-uniform", name="eta"),
        Real(0.5, 1.0, prior="uniform", name="subsample"),
        Real(0.0, 10.0, prior="uniform", name="reg_alpha"),
        Real(0.0, 10.0, prior="uniform", name="reg_lambda"),
        Integer(500, 3000, name="n_estimators"),
        Real(0.0, 1.0, prior="uniform", name="colsample_bytree"),
        Categorical(["hist", "approx", "gpu_hist"], name="tree_method"),
    ]

    dtrain_cv = build_dmatrix(prepared.X_train, prepared.y)

    @use_named_args(search_space)
    def objective(**params_opt):
        xgb_params = default_params.copy()
        xgb_params.update(
            {
                "max_depth": params_opt["max_depth"],
                "eta": params_opt["eta"],
                "subsample": params_opt["subsample"],
                "reg_alpha": params_opt["reg_alpha"],
                "reg_lambda": params_opt["reg_lambda"],
                "colsample_bytree": params_opt["colsample_bytree"],
                "tree_method": params_opt["tree_method"],
            }
        )

        cv_results = xgb.cv(
            params=xgb_params,
            dtrain=dtrain_cv,
            num_boost_round=params_opt["n_estimators"],
            nfold=3,
            metrics="rmse",
            early_stopping_rounds=20,
            verbose_eval=False,
            seed=42,
        )
        return cv_results["test-rmse-mean"].min()

    result = gp_minimize(objective, search_space, n_calls=30, random_state=42)

    trained_params = default_params.copy()
    for dim, val in zip(search_space, result.x):
        if dim.name in trained_params:
            trained_params[dim.name] = int(val) if isinstance(dim, Integer) else val

    n_estimators_for_cv = int(
        trained_params.get("n_estimators", default_params.get("n_estimators", 1000))
    )

    cv_results = xgb.cv(
        params=trained_params,
        dtrain=dtrain_cv,
        num_boost_round=n_estimators_for_cv,
        nfold=3,
        metrics="rmse",
        early_stopping_rounds=20,
        verbose_eval=False,
        seed=42,
    )
    best_iteration = int(cv_results["test-rmse-mean"].idxmin())
    final_boost_rounds = best_iteration + 1

    trained_params["n_estimators"] = final_boost_rounds

    dtrain = build_dmatrix(prepared.X_train, prepared.y)
    rich_callback = RichProgressBarCallback(total_rounds=final_boost_rounds)

    model = xgb.train(
        trained_params,
        dtrain,
        num_boost_round=final_boost_rounds,
        callbacks=[rich_callback],
    )

    dtest = build_dmatrix(prepared.X_test, None)
    preds = model.predict(dtest)

    results = prepared.identifiers.copy()
    results["prediction"] = preds

    output_path = _save_predictions(session, results, config)

    return {
        "model_type": "market",
        "best_params": trained_params,
        "best_iteration": best_iteration,
        "output_path": output_path,
    }


def _compute_base_margin(df: pd.DataFrame, exposure_col: str, offset_col: Optional[str]) -> np.ndarray:
    exposure = np.clip(df[exposure_col].to_numpy(dtype=float), 1e-12, None)
    offset = df[offset_col].to_numpy(dtype=float) if offset_col else np.zeros_like(exposure)
    return np.log(exposure) + offset


def _extract_weights(df: pd.DataFrame, weight_col: Optional[str]) -> Optional[np.ndarray]:
    if not weight_col:
        return None
    return df[weight_col].to_numpy(dtype=float)


def _run_frequency_model(
    prepared: PreparedData,
    config: configparser.ConfigParser,
    session,
) -> Dict[str, object]:
    extras = _parse_frequency_extras(config)
    _ensure_columns(
        prepared.train_df,
        [extras.exposure, extras.partition] + [extras.weight, extras.offset],
    )
    _ensure_columns(
        prepared.test_df,
        [extras.exposure] + [extras.weight, extras.offset],
    )

    params = _parse_params(config["PARAMS"])
    num_boost_round = int(params.pop("num_boost_round", params.pop("n_estimators", 5000)))
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 200))

    mono_inc = _parse_list(config["VARIABLES"].get("mono_inc"))
    mono_dec = _parse_list(config["VARIABLES"].get("mono_dec"))
    monotone = build_monotone_constraints(prepared.feature_names, mono_inc, mono_dec)
    params.setdefault("monotone_constraints", monotone)

    if "eval_metric" not in params:
        params["eval_metric"] = "poisson-nloglik"
    if "objective" not in params:
        params["objective"] = "count:poisson"
    params.setdefault("tree_method", "hist")

    train_mask = prepared.train_df[extras.partition].isin(extras.train_partition_values)
    valid_mask = prepared.train_df[extras.partition].isin(extras.valid_partition_values)

    if not train_mask.any():
        raise ValueError("No rows matched the configured training partition values")
    if not valid_mask.any():
        raise ValueError("No rows matched the configured validation partition values")

    X_tr = prepared.X_train.loc[train_mask]
    X_va = prepared.X_train.loc[valid_mask]
    y_tr = prepared.y.loc[train_mask].to_numpy(dtype=float)
    y_va = prepared.y.loc[valid_mask].to_numpy(dtype=float)

    bm_tr = _compute_base_margin(prepared.train_df.loc[train_mask], extras.exposure, extras.offset)
    bm_va = _compute_base_margin(prepared.train_df.loc[valid_mask], extras.exposure, extras.offset)

    w_tr = _extract_weights(prepared.train_df.loc[train_mask], extras.weight)
    w_va = _extract_weights(prepared.train_df.loc[valid_mask], extras.weight)

    dtr = xgb.DMatrix(
        X_tr,
        label=y_tr,
        weight=w_tr,
        base_margin=bm_tr,
        feature_names=prepared.feature_names,
    )
    dva = xgb.DMatrix(
        X_va,
        label=y_va,
        weight=w_va,
        base_margin=bm_va,
        feature_names=prepared.feature_names,
    )

    evals = [(dtr, "train"), (dva, "valid")]
    model = xgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    best_iteration = model.best_iteration if model.best_iteration is not None else num_boost_round - 1
    pred_va = model.predict(dva, iteration_range=(0, best_iteration + 1))
    poiss_dev_valid = poisson_deviance(y_va, pred_va, sample_weight=w_va)
    wgini_norm_valid = weighted_gini_norm(y_va, pred_va, weight=w_va)

    # Prepare test predictions
    test_base_margin = _compute_base_margin(prepared.test_df, extras.exposure, extras.offset)
    test_weights = _extract_weights(prepared.test_df, extras.weight)
    dtest = xgb.DMatrix(
        prepared.X_test,
        base_margin=test_base_margin,
        weight=test_weights,
        feature_names=prepared.feature_names,
    )
    preds = model.predict(dtest, iteration_range=(0, best_iteration + 1))
    results = prepared.identifiers.copy()
    results["prediction"] = preds
    output_path = _save_predictions(session, results, config)

    return {
        "model_type": "frequency",
        "best_iteration": best_iteration,
        "valid_poisson_deviance": poiss_dev_valid,
        "valid_weighted_gini_norm": wgini_norm_valid,
        "output_path": output_path,
    }


def run_from_config(config_file: str = CONFIG_FILE) -> Dict[str, object]:
    config = configparser.ConfigParser()
    config.read(config_file)
    df_train, df_test = _read_data(config)
    prepared = _prepare_features(config, df_train, df_test)

    session = None
    try:
        session = get_active_session()
    except Exception:
        session = None

    model_type = _normalize_config_entry(config["VARIABLES"].get("model_type")) or "market"
    model_type = model_type.lower()
    if model_type not in {"market", "frequency"}:
        raise ValueError("[VARIABLES] model_type must be either 'market' or 'frequency'")

    if model_type == "frequency":
        return _run_frequency_model(prepared, config, session)
    return _run_market_model(prepared, config, session)
