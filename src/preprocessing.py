"""NSL-KDD loading and preprocessing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
NORMAL_VAL_FRACTION = 0.2
CORRELATION_THRESHOLD = 0.95

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

ATTACK_CATEGORIES = {
    "normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
META_COLUMNS = ["label", "difficulty_level", "category", "is_attack"]


@dataclass
class PreprocessBundle:
    """Fitted preprocessing artifacts for reproducible transforms."""

    encoded_columns: list[str]
    feature_cols: list[str]
    removed_constant: list[str] = field(default_factory=list)
    removed_correlated: list[str] = field(default_factory=list)
    scaler_anomaly: StandardScaler | None = None
    scaler_supervised: StandardScaler | None = None

    def active_features(self) -> list[str]:
        return self.feature_cols


def load_nslkdd(path: str | Path) -> pd.DataFrame:
    """Load an NSL-KDD split and derive attack metadata."""
    df = pd.read_csv(path, names=COLUMNS, header=None)
    df["label"] = df["label"].str.strip()
    df["category"] = df["label"].map(ATTACK_CATEGORIES).fillna("unknown")
    df["is_attack"] = (df["category"] != "normal").astype(int)
    return df


def split_normal_rows(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split normal rows from KDDTrain+ into train/validation (anomaly calibration)."""
    normal_df = train_df[train_df["is_attack"] == 0].copy()
    return train_test_split(
        normal_df,
        test_size=NORMAL_VAL_FRACTION,
        random_state=RANDOM_STATE,
    )


def fit_encoding_columns(train_df: pd.DataFrame) -> list[str]:
    """Learn one-hot column layout from official training data only."""
    encoded = pd.get_dummies(train_df, columns=CATEGORICAL_FEATURES, dtype=float)
    return list(encoded.columns)


def encode_dataframe(df: pd.DataFrame, encoded_columns: list[str]) -> pd.DataFrame:
    """One-hot encode and align to training column vocabulary."""
    encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dtype=float)
    return encoded.reindex(columns=encoded_columns, fill_value=0.0)


def base_feature_columns(encoded_columns: list[str]) -> list[str]:
    """Return model feature names excluding label metadata."""
    return [col for col in encoded_columns if col not in META_COLUMNS]


def detect_constant_columns(encoded_train: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Identify features with a single unique value in training data."""
    return [col for col in feature_cols if encoded_train[col].nunique(dropna=False) <= 1]


def detect_redundant_columns(
    encoded_train: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = CORRELATION_THRESHOLD,
) -> list[str]:
    """Drop one feature from each highly correlated pair (train only)."""
    if len(feature_cols) < 2:
        return []
    corr = encoded_train[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: set[str] = set()
    for col_a, col_b in upper.stack().index:
        if upper.loc[col_a, col_b] >= threshold:
            to_drop.add(col_b)
    return sorted(to_drop)


def apply_feature_selection(
    feature_cols: list[str],
    remove_constant: list[str] | None = None,
    remove_correlated: list[str] | None = None,
) -> list[str]:
    """Return feature list after optional removals."""
    remove = set(remove_constant or []) | set(remove_correlated or [])
    return [col for col in feature_cols if col not in remove]


def matrix_from_frame(
    encoded_df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> np.ndarray:
    """Extract numpy matrix; optionally fit/transform scaler."""
    values = encoded_df[feature_cols].to_numpy(dtype=np.float64)
    if scaler is None:
        return values
    if fit_scaler:
        return scaler.fit_transform(values)
    return scaler.transform(values)


def build_preprocess_bundle(
    train_df: pd.DataFrame,
    normal_train_df: pd.DataFrame,
    remove_constants: bool = True,
    remove_correlated: bool = False,
) -> PreprocessBundle:
    """
    Build strict preprocessing:
    - encoding vocabulary from KDDTrain+
    - optional constant / redundant removal using train statistics
    - anomaly scaler fit on normal_train only
    - supervised scaler fit on all KDDTrain+ rows
    """
    encoded_columns = fit_encoding_columns(train_df)
    train_enc = encode_dataframe(train_df, encoded_columns)
    normal_train_enc = encode_dataframe(normal_train_df, encoded_columns)

    base_cols = base_feature_columns(encoded_columns)
    constant_cols = detect_constant_columns(train_enc, base_cols) if remove_constants else []
    feature_cols = apply_feature_selection(base_cols, remove_constant=constant_cols)
    redundant_cols: list[str] = []
    if remove_correlated:
        redundant_cols = detect_redundant_columns(train_enc, feature_cols)
        feature_cols = apply_feature_selection(feature_cols, remove_correlated=redundant_cols)

    scaler_anomaly = StandardScaler()
    scaler_supervised = StandardScaler()
    matrix_from_frame(normal_train_enc, feature_cols, scaler_anomaly, fit_scaler=True)
    matrix_from_frame(train_enc, feature_cols, scaler_supervised, fit_scaler=True)

    return PreprocessBundle(
        encoded_columns=encoded_columns,
        feature_cols=feature_cols,
        removed_constant=constant_cols,
        removed_correlated=redundant_cols,
        scaler_anomaly=scaler_anomaly,
        scaler_supervised=scaler_supervised,
    )


def transform_splits(
    bundle: PreprocessBundle,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    normal_train_df: pd.DataFrame,
    normal_val_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Transform all splits with the fitted bundle."""
    train_enc = encode_dataframe(train_df, bundle.encoded_columns)
    test_enc = encode_dataframe(test_df, bundle.encoded_columns)
    normal_train_enc = encode_dataframe(normal_train_df, bundle.encoded_columns)
    normal_val_enc = encode_dataframe(normal_val_df, bundle.encoded_columns)

    cols = bundle.active_features()
    return {
        "x_train_supervised": matrix_from_frame(train_enc, cols, bundle.scaler_supervised),
        "x_test_supervised": matrix_from_frame(test_enc, cols, bundle.scaler_supervised),
        "x_normal_train": matrix_from_frame(normal_train_enc, cols, bundle.scaler_anomaly),
        "x_normal_val": matrix_from_frame(normal_val_enc, cols, bundle.scaler_anomaly),
        "x_test_anomaly": matrix_from_frame(test_enc, cols, bundle.scaler_anomaly),
    }


def save_preprocessing_summary(bundle: PreprocessBundle, path: Path) -> None:
    """Persist preprocessing decisions for the report."""
    payload = {
        "num_features_after_encoding": len(bundle.encoded_columns),
        "num_model_features": len(bundle.feature_cols),
        "removed_constant_columns": bundle.removed_constant,
        "removed_correlated_columns": bundle.removed_correlated,
        "anomaly_scaler_fit_on": "normal_train_only",
        "supervised_scaler_fit_on": "all_kddtrain_rows",
        "encoding_fit_on": "kddtrain_only",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
