"""NSL-KDD loading and preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

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
DROP_COLUMNS = ["label", "difficulty_level", "category", "is_attack"]


def load_nslkdd(path: str) -> pd.DataFrame:
    """Load an NSL-KDD split and derive attack metadata."""
    df = pd.read_csv(path, names=COLUMNS, header=None)
    df["label"] = df["label"].str.strip()
    df["category"] = df["label"].map(ATTACK_CATEGORIES).fillna("unknown")
    df["is_attack"] = (df["category"] != "normal").astype(int)
    return df


def encode_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categoricals without leaking test-only categories into training."""
    train_encoded = pd.get_dummies(train_df, columns=CATEGORICAL_FEATURES, dtype=float)
    test_encoded = pd.get_dummies(test_df, columns=CATEGORICAL_FEATURES, dtype=float)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0.0)
    return train_encoded, test_encoded


def build_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
    """Return scaled feature matrices and fitted scaler."""
    train_encoded, test_encoded = encode_features(train_df, test_df)
    feature_cols = [col for col in train_encoded.columns if col not in DROP_COLUMNS]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_encoded[feature_cols])
    x_test = scaler.transform(test_encoded[feature_cols])
    return x_train, x_test, feature_cols, scaler
