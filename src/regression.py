"""
Trains a logistic regression model on entropy data to classify pages as good or bad based on CER thresholds,
computes performance metrics including AUC, ROC curves, and evaluates thresholds using Youden's J or Minimum Error statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from .utils import save_figures

from .utils import YOUDEN_J, MIN_ERROR


def assign_page_labels(df):
    df["good_page_primary"] = (df["cer"] <= 0.01).astype("uint8")
    df["good_page_secondary"] = (df["cer"] <= 0.02).astype("uint8")
    return df


# Answers the question: will this entropy level have a CER <= 1% or 2%?
def train_logistic_regression_model(df, primary: bool = False, random_state=50):
    x = np.asarray(df["avg_bits_per_token"]).reshape(-1, 1)

    if primary:
        y = np.asarray(df["good_page_primary"])
    else:
        y = np.asarray(df["good_page_secondary"])

    # Implicitly shuffles data
    X_train, X_val, Y_train, Y_val, _, val_indices = train_test_split(
        x, y, df.index, test_size=0.20, random_state=random_state
    )

    try:
        clf = LogisticRegression(random_state=random_state).fit(X_train, Y_train)
    except Exception as e:
        raise e

    P_val = clf.predict_proba(X_val)[:, 1]
    return P_val, Y_val, val_indices


def compute_auc(true_class, predicted_class):
    return roc_auc_score(true_class, predicted_class)


def compute_roc_curve(true_class, predicted_class):
    fpr, tpr, thresholds = roc_curve(true_class, predicted_class)
    return fpr, tpr, thresholds


def visualize_roc_curve(output, use_primary, fpr, tpr):
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, marker="o", markersize=3, alpha=0.5)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    if use_primary:
        plt.title(f"ROC Curve for a 1% CER Threshold")
    else:
        plt.title(f"ROC Curve for a 2% CER Threshold")

    save_figures(plt.gcf(), f"{output}/figures/figure_06_roc_entropy")


def use_youden_j_statistic(thresholds, fpr, tpr) -> float:
    j = tpr - fpr
    # Finds where j is maximized in thresholds
    youden_j_threshold = thresholds[np.argmax(j)]
    return youden_j_threshold


def use_min_error_statistic(thresholds, fpr, tpr, Y_val) -> float:
    misclassification = 1 - (tpr * sum(Y_val) + (1 - fpr) * sum(1 - Y_val)) / len(Y_val)
    min_error_threshold = thresholds[np.argmin(misclassification[1:])]
    return min_error_threshold


def compute_threshold(thresholds, fpr, tpr, Y_val, statistic):
    if statistic == YOUDEN_J:
        return use_youden_j_statistic(thresholds, fpr, tpr)

    if statistic == MIN_ERROR:
        return use_min_error_statistic(thresholds, fpr, tpr, Y_val)


def compute_sensitivity(tp, fn):
    return tp / (tp + fn)


def compute_specificity(tn, fp):
    return tn / (tn + fp)


def get_misclassified_triage_decisions(df, use_primary, threshold_type):
    df = assign_page_labels(df)

    P_val, Y_val, val_indices = train_logistic_regression_model(df, use_primary)
    fpr, tpr, thresholds = compute_roc_curve(Y_val, P_val)
    threshold = compute_threshold(thresholds, fpr, tpr, Y_val, threshold_type)

    Y_pred = P_val >= threshold

    return (Y_pred == Y_val), val_indices


def create_roc_table(output, use_primary, table_data, headers):
    fig, ax = plt.subplots(figsize=(10, 2.2))

    tbl = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    if use_primary:
        title = "ROC Threshold Summary (CER ≤ 1%)"
    else:
        title = "ROC Threshold Summary (CER ≤ 2%)"

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    ax.set_title(title, pad=12)
    ax.axis("off")

    fig.tight_layout()
    save_figures(fig, f"{output}/tables/table_01_roc_table")


def main(df, output, use_primary=False):
    df = assign_page_labels(df)

    P_val, Y_val, _ = train_logistic_regression_model(df, use_primary)
    auc = compute_auc(Y_val, P_val)
    fpr, tpr, thresholds = compute_roc_curve(Y_val, P_val)
    visualize_roc_curve(output, use_primary, fpr, tpr)

    table_data = []
    threshold_statistics = [YOUDEN_J, MIN_ERROR]
    for statistic in threshold_statistics:
        threshold = compute_threshold(thresholds, fpr, tpr, Y_val, statistic)

        Y_pred = P_val >= threshold

        tp, fp, fn, tn = confusion_matrix(Y_val, Y_pred).ravel()

        if (tp == 0 and fn == 0) or (tn == 0 and fp == 0):
            print(
                f"Encountered either no positive or negative class samples with statistic {statistic}. Skipping sensitivity and specificity calculation. {statistic} will not be included in the ROC table."
            )
            continue

        sensitivity = compute_sensitivity(tp, fn)
        specificity = compute_specificity(tn, fp)
        table_data.append(
            [
                statistic,
                f"{round(threshold, 4):.4f}",
                f"{round(sensitivity, 4):.4f}",
                f"{round(specificity, 4):.4f}",
                f"{round(auc, 4):.4f}",
            ]
        )

    headers = ["Threshold Type", "Threshold", "Sensitivity", "Specificity", "AUROC"]
    create_roc_table(output, use_primary, table_data, headers)


if __name__ == "__main__":
    pass
