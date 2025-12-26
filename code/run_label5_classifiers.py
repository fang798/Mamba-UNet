"""
Train/evaluate classic classifiers on extracted CRANet_dense features.

Requirements:
- Keep all epochs of the same image in the same split (group by image_name).
- Skip samples with label == -1.
- Use class weights (if supported) based on train-set label counts.

Example:
python /home/hit/fx/Mamba-UNet/code/run_label5_classifiers.py \
  --in_file /home/hit/fx/Mamba-UNet/run_result/cranet_dense_features.csv \
  --label_col label5 \
  --out_dir /home/hit/fx/Mamba-UNet/run_result
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedBaggingClassifier


def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _build_group_split(
    df: pd.DataFrame, label_col: str, test_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df["image_name"].values
    labels = df[label_col].values
    train_idx, test_idx = next(gss.split(df, labels, groups=groups))
    return train_idx, test_idx


def _compute_class_weight(y: np.ndarray) -> Dict[int, float]:
    classes = np.array([0, 1], dtype=int)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _evaluate(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds, labels=[0, 1])

    roc_auc = None
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, scores)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        roc_auc = roc_auc_score(y, scores)

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
    if roc_auc is not None:
        metrics["roc_auc"] = float(roc_auc)
    return metrics, cm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_file",
        type=str,
        default="/home/hit/fx/Mamba-UNet/run_result/cranet_dense_features.csv",
    )
    parser.add_argument("--label_col", type=str, default="label5")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/hit/fx/Mamba-UNet/run_result",
    )
    args = parser.parse_args()

    print("Stage: loading data")
    if not os.path.exists(args.in_file):
        raise FileNotFoundError(f"input file not found: {args.in_file}")

    df = _read_table(args.in_file)
    if "image_name" not in df.columns:
        raise ValueError("input file must contain column: image_name")
    if args.label_col not in df.columns:
        raise ValueError(f"label column not found: {args.label_col}")

    print("Stage: filtering labels")
    df = df[df[args.label_col].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("no valid samples (label 0/1) after filtering")

    print("Stage: preparing features")
    feature_cols = [
        c
        for c in df.columns
        if c.startswith("mask_f") or c.startswith("image_f") or c.startswith("unclear_f")
    ]
    if not feature_cols:
        raise ValueError("no feature columns found (mask_f*, image_f*, unclear_f*)")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    print("Stage: splitting train/test")
    train_idx, test_idx = _build_group_split(
        df, args.label_col, args.test_size, args.seed
    )
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[args.label_col].to_numpy(dtype=int)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df[args.label_col].to_numpy(dtype=int)

    class_weight = _compute_class_weight(y_train)

    models = {
        # "svm_rbf": make_pipeline(
        #     StandardScaler(),
        #     SVC(
        #         kernel="rbf",
        #         class_weight=class_weight,
        #         probability=True,
        #         random_state=args.seed,
        #     ),
        # ),
        "decision_tree": DecisionTreeClassifier(
            class_weight=class_weight, random_state=args.seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        ),
        # "logreg": make_pipeline(
        #     StandardScaler(),
        #     LogisticRegression(
        #         max_iter=2000,
        #         class_weight=class_weight,
        #         random_state=args.seed,
        #     ),
        # ),
    }

    if BalancedBaggingClassifier is not None:
        bbc_kwargs = {
            "sampling_strategy": "auto",
            "n_estimators": 50,
            "replacement": True,
            "random_state": args.seed,
            "n_jobs": -1,
        }
        base_tree = DecisionTreeClassifier(
            class_weight=class_weight, random_state=args.seed
        )
        try:
            models["bbc"] = BalancedBaggingClassifier(
                estimator=base_tree, **bbc_kwargs
            )
        except TypeError:
            models["bbc"] = BalancedBaggingClassifier(
                base_estimator=base_tree, **bbc_kwargs
            )

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "label5_classification_results.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"input: {args.in_file}\n")
        f.write(f"label_col: {args.label_col}\n")
        f.write(f"train_size: {len(train_df)}  test_size: {len(test_df)}\n")
        f.write(f"unique_train_images: {train_df['image_name'].nunique()}\n")
        f.write(f"unique_test_images: {test_df['image_name'].nunique()}\n")
        f.write(f"class_weight: {class_weight}\n")
        f.write(f"feature_dim: {len(feature_cols)}\n\n")

        print("Stage: training models")
        for name, model in tqdm(list(models.items()), desc="Models"):
            model.fit(X_train, y_train)

            train_metrics, train_cm = _evaluate(model, X_train, y_train)
            test_metrics, test_cm = _evaluate(model, X_test, y_test)

            f.write(f"[{name}]\n")
            f.write(f"train_metrics: {train_metrics}\n")
            f.write(f"train_confusion: {train_cm.tolist()}\n")
            f.write(f"test_metrics: {test_metrics}\n")
            f.write(f"test_confusion: {test_cm.tolist()}\n\n")

    print(f"Saved results to: {report_path}")


if __name__ == "__main__":
    main()
