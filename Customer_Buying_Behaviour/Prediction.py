import argparse
import json
import sys
import warnings
from pathlib import Path


import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

warnings.filterwarnings("ignore")

DATA = Path(r"C:\Users\B01846982\OneDrive - University of the West of Scotland\Desktop\Customer_Buying_Behaviour\customer_booking.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
TARGET = "booking_complete"

# Colour palette (British Airways-ish)
NAVY, RED, GREY = "#1b3a6b", "#c8102e", "#9aa5b1"


# ----------------------------------------------------------------------------
# 1. Load & explore
# ----------------------------------------------------------------------------
def load_data(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    dupes = int(df.duplicated().sum())
    print(f"Exact duplicate rows: {dupes:,} (kept - repeated search patterns are legitimate)")
    rate = df[TARGET].mean()
    print(f"Target balance: {rate:.2%} positive / {1 - rate:.2%} negative  -> imbalanced\n")
    return df


# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Extras: individual flags plus a combined "engagement" score -------
    extras = ["wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals"]
    df["num_extras"] = df[extras].sum(axis=1)

    # --- Day of week: ordered + weekend flag ------------------------------
    day_map = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    df["flight_day_num"] = df["flight_day"].map(day_map)
    df["is_weekend_flight"] = df["flight_day_num"].isin([6, 7]).astype(int)

    # --- Departure time of day buckets ------------------------------------
    df["is_red_eye"] = df["flight_hour"].between(0, 5).astype(int)

    # --- Trip shape --------------------------------------------------------
    df["stay_per_lead"] = df["length_of_stay"] / (df["purchase_lead"] + 1)
    df["is_long_haul"] = (df["flight_duration"] >= 7).astype(int)
    df["is_last_minute"] = (df["purchase_lead"] <= 7).astype(int)
    df["is_solo"] = (df["num_passengers"] == 1).astype(int)

    # --- Route / origin: target-free frequency encoding --------------------
    # 799 routes and 104 origins would explode into huge sparse one-hot blocks
    # and starve the trees of signal, so encode by how common each value is.
    df["route_freq"] = df["route"].map(df["route"].value_counts(normalize=True))
    df["origin_freq"] = df["booking_origin"].map(
        df["booking_origin"].value_counts(normalize=True)
    )
    # Geographic signal: origin airport (first 3 chars) and destination
    df["origin_airport"] = df["route"].str[:3]
    df["dest_airport"] = df["route"].str[3:]
    df["origin_airport_freq"] = df["origin_airport"].map(
        df["origin_airport"].value_counts(normalize=True)
    )
    df["dest_airport_freq"] = df["dest_airport"].map(
        df["dest_airport"].value_counts(normalize=True)
    )

    # --- Low-cardinality categoricals: one-hot ----------------------------
    df = pd.get_dummies(df, columns=["sales_channel", "trip_type"], drop_first=True)

    df = df.drop(columns=["flight_day", "route", "booking_origin",
                          "origin_airport", "dest_airport"])
    return df


# ----------------------------------------------------------------------------
# 3. Cross-validation
# ----------------------------------------------------------------------------
def cross_validate_model(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
                         return_train_score=True)

    print("=" * 66)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("=" * 66)
    print(f"{'Metric':<12}{'Test mean':>12}{'Std':>9}{'Train mean':>13}")
    print("-" * 66)
    summary = {}
    for name in scoring:
        te, tr = res[f"test_{name}"], res[f"train_{name}"]
        summary[name] = {"mean": float(te.mean()), "std": float(te.std()),
                         "train_mean": float(tr.mean()),
                         "folds": [round(float(v), 4) for v in te]}
        print(f"{name:<12}{te.mean():>12.4f}{te.std():>9.4f}{tr.mean():>13.4f}")
    print("-" * 66)
    baseline = float(y.mean())
    print(f"Baseline (always predict 'no booking') accuracy = {1 - baseline:.4f}")
    print(f"Baseline PR-AUC (random)                       = {baseline:.4f}\n")
    return summary


# ----------------------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------------------
PRETTY = {
    "purchase_lead": "Purchase lead time (days)",
    "length_of_stay": "Length of stay (nights)",
    "flight_hour": "Departure hour",
    "flight_duration": "Flight duration (hrs)",
    "route_freq": "Route popularity",
    "origin_freq": "Booking-origin popularity",
    "origin_airport_freq": "Origin airport popularity",
    "dest_airport_freq": "Destination airport popularity",
    "num_passengers": "Number of passengers",
    "num_extras": "Number of extras requested",
    "flight_day_num": "Day of week",
    "stay_per_lead": "Stay / lead-time ratio",
    "wants_extra_baggage": "Wants extra baggage",
    "wants_preferred_seat": "Wants preferred seat",
    "wants_in_flight_meals": "Wants in-flight meals",
    "is_weekend_flight": "Weekend flight",
    "is_red_eye": "Red-eye departure",
    "is_long_haul": "Long-haul flight",
    "is_last_minute": "Last-minute booking",
    "is_solo": "Solo traveller",
    "sales_channel_Mobile": "Sales channel: Mobile",
    "trip_type_OneWay": "Trip type: One way",
    "trip_type_RoundTrip": "Trip type: Round trip",
}
label = lambda c: PRETTY.get(c, c.replace("_", " ").capitalize())


def plot_importances(model, X, y_test, X_test, feats):
    gini = pd.Series(model.feature_importances_, index=feats).sort_values()

    print("Computing permutation importance (this takes a moment)...")
    perm = permutation_importance(model, X_test, y_test, n_repeats=10,
                                  random_state=RANDOM_STATE, n_jobs=-1,
                                  scoring="roc_auc")
    perm_s = pd.Series(perm.importances_mean, index=feats)
    perm_std = pd.Series(perm.importances_std, index=feats)

    # --- Main deliverable: Gini importance --------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    colors = [RED if v >= gini.quantile(0.8) else NAVY for v in gini]
    ax.barh([label(c) for c in gini.index], gini.values, color=colors)
    for y_, v in enumerate(gini.values):
        ax.text(v + gini.max() * 0.01, y_, f"{v:.3f}", va="center", fontsize=9, color="#333")
    ax.set_xlabel("Mean decrease in impurity (relative contribution)", fontsize=11)
    ax.set_title("What drives a completed booking?\nRandom Forest feature importance",
                 fontsize=15, fontweight="bold", loc="left", color=NAVY)
    ax.set_xlim(0, gini.max() * 1.13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.01, 0.01, "Red = top 20% of features by contribution", fontsize=9, color=GREY)
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)

    # --- Cross-check: permutation importance ------------------------------
    p = perm_s.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([label(c) for c in p.index], p.values,
            xerr=perm_std[p.index].values, color=NAVY, ecolor=GREY, capsize=3)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled", fontsize=11)
    ax.set_title("Permutation importance (top 15)\nModel-agnostic cross-check on held-out data",
                 fontsize=14, fontweight="bold", loc="left", color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "permutation_importance.png", dpi=150)
    plt.close(fig)

    return gini.sort_values(ascending=False), perm_s.sort_values(ascending=False)


def plot_evaluation(y_test, y_pred, y_proba, cv_summary):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No booking", "Booking"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format=",d")
    axes[0].set_title("Confusion matrix (hold-out)", fontweight="bold", color=NAVY)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[1].plot(fpr, tpr, color=NAVY, lw=2, label=f"RandomForest (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "--", color=GREY, label="Random (AUC = 0.500)")
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC curve", fontweight="bold", color=NAVY)
    axes[1].legend(loc="lower right"); axes[1].grid(alpha=0.25)

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    base = y_test.mean()
    axes[2].plot(rec, prec, color=RED, lw=2, label=f"RandomForest (PR-AUC = {ap:.3f})")
    axes[2].axhline(base, ls="--", color=GREY, label=f"Baseline ({base:.3f})")
    axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
    axes[2].set_title("Precision-Recall curve", fontweight="bold", color=NAVY)
    axes[2].legend(); axes[2].grid(alpha=0.25)

    for ax in axes[1:]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model evaluation - predicting completed bookings",
                 fontsize=15, fontweight="bold", color=NAVY, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "model_evaluation.png", dpi=150)
    plt.close(fig)

    # CV metric spread
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(cv_summary)
    means = [cv_summary[n]["mean"] for n in names]
    stds = [cv_summary[n]["std"] for n in names]
    ax.bar(names, means, yerr=stds, capsize=5, color=NAVY, ecolor=RED)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
    ax.set_title("5-fold cross-validation scores (mean ± std)",
                 fontweight="bold", color=NAVY, loc="left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "cv_scores.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    df = load_data()
    df = engineer(df)

    X = df.drop(columns=[TARGET]).astype(float)
    y = df[TARGET]
    feats = list(X.columns)
    print(f"Modelling matrix: {X.shape[0]:,} rows x {X.shape[1]} features\n")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt",
        class_weight="balanced_subsample",  # counteract the 85/15 imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv_summary = cross_validate_model(model, X, y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("=" * 66)
    print("HOLD-OUT TEST SET (20%, n = {:,})".format(len(y_te)))
    print("=" * 66)
    print(classification_report(y_te, y_pred,
                                target_names=["No booking", "Booking"], digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_te, y_proba):.4f}   "
          f"PR-AUC: {average_precision_score(y_te, y_proba):.4f}\n")

    gini, perm = plot_importances(model, X, y_te, X_te, feats)
    plot_evaluation(y_te, y_pred, y_proba, cv_summary)

    print("=" * 66)
    print("TOP 10 FEATURES (Gini importance)")
    print("=" * 66)
    for i, (f, v) in enumerate(gini.head(10).items(), 1):
        print(f"{i:>2}. {label(f):<34} {v:.4f}")
    print("\nTOP 10 FEATURES (Permutation importance, ROC-AUC drop)")
    for i, (f, v) in enumerate(perm.head(10).items(), 1):
        print(f"{i:>2}. {label(f):<34} {v:.4f}")

    json.dump(
        {"cv": cv_summary,
         "holdout": {"roc_auc": float(roc_auc_score(y_te, y_proba)),
                     "pr_auc": float(average_precision_score(y_te, y_proba))},
         "gini_importance": {k: float(v) for k, v in gini.items()},
         "permutation_importance": {k: float(v) for k, v in perm.items()}},
        open(OUT / "results.json", "w"), indent=2)
    print(f"\nSaved charts + results.json to {OUT.resolve()}")


if __name__ == "__main__":
    main()
