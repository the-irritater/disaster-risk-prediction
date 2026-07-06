import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, auc, brier_score_loss
)
from sklearn.calibration import calibration_curve

def calculate_classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes a comprehensive dictionary of classification metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Precision and Recall
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Area under curves
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    
    brier = brier_score_loss(y_true, y_prob)
    
    # Balanced accuracy & Matthews Correlation Coefficient
    balanced_acc = (recall + specificity) / 2.0
    
    # MCC
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    
    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "balanced_accuracy": float(balanced_acc),
        "mcc": float(mcc)
    }

def calculate_regression_metrics(y_true, y_pred):
    """
    Computes standard regression metrics.
    MdAPE (Median APE) is used instead of MAPE because economic loss values
    include near-zero observations that cause arithmetic explosion in MAPE.
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # MdAPE — Median Absolute Percentage Error (robust to near-zero y values)
    mask = y_true > 0.1  # exclude effectively-zero loss events
    mdape = float(
        np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    ) if mask.sum() > 0 else 0.0

    mean_y = float(np.mean(y_true))
    std_y = float(np.std(y_true))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r_squared": round(r2, 4),
        "mdape": round(mdape, 4),   # Median APE (replaces MAPE)
        "mean_y": round(mean_y, 4),
        "std_y": round(std_y, 4)
    }

def find_optimal_threshold(y_true, y_prob, target_recall=0.75):
    """
    Finds the largest probability threshold that meets a minimum target recall constraint.
    This balances the disaster-risk focus (high recall is critical to avoid false negatives).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Filter thresholds where recall is >= target_recall
    valid_idx = np.where(recalls >= target_recall)[0]
    if len(valid_idx) == 0:
        # If no threshold reaches target, return the one with highest recall
        best_idx = np.argmax(recalls)
    else:
        # From the valid indices, choose the one with the highest precision
        best_idx = valid_idx[np.argmax(precisions[valid_idx])]
        
    # Return chosen threshold
    if best_idx >= len(thresholds):
        return 0.50
    return float(thresholds[best_idx])

def plot_and_save_curves(y_true, y_prob, output_dir="images"):
    """
    Plots ROC curve, PR curve, and Calibration curve, saving them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ROC & Precision-Recall curves combined plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Receiver Operating Characteristic (ROC)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    axes[1].plot(rec, prec, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall (PR) Curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_pr_curves.png"), dpi=150)
    plt.close(fig)
    
    # 2. Calibration Curve plot
    fig_cal, ax_cal = plt.subplots(figsize=(7, 7))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)
    
    ax_cal.plot(prob_pred, prob_true, marker="s", color="darkorange", label=f"Model (Brier = {brier:.4f})")
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.set_title("Probability Calibration Curve")
    ax_cal.legend(loc="lower right")
    ax_cal.grid(True, linestyle=":", alpha=0.6)
    
    fig_cal.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=150)
    plt.close(fig_cal)
    
    print(f"Evaluation plots saved to {output_dir}/")
