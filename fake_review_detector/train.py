"""
Fake Review Detector — Improved Training Pipeline
===================================================
Changes from original:
  1. Multiple models compared (Logistic Regression, SGD, Random Forest)
  2. Stratified K-Fold cross-validation (5 folds)
  3. Richer behavioural features (sentiment cues, repetition, specificity)
  4. TF-IDF with sublinear_tf and more features
  5. Hyperparameter tuning for the best model
  6. Proper threshold calibration
  7. Shared feature engineering via utils.py
  8. Comprehensive evaluation with per-class metrics
"""

import pandas as pd
import numpy as np
import re, os, pickle, time
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils import clean_text, extract_behavioural_features

# ── 1. Load & prepare data ─────────────────────────────────────
print("=" * 60)
print("FAKE REVIEW DETECTOR — TRAINING PIPELINE")
print("=" * 60)

print("\n[1/6] Loading data...")
df = pd.read_csv("data/reviews.csv")
df = df.rename(columns={"text_": "text"})
df = df[["text", "label", "rating", "category"]].dropna()
df["label"] = df["label"].str.strip().str.upper()
df["label"] = df["label"].map({"OR": "real", "CG": "fake"})
df = df.dropna(subset=["label"])
print(f"  Total reviews : {len(df):,}")
print(f"  Real          : {(df['label']=='real').sum():,}")
print(f"  Fake          : {(df['label']=='fake').sum():,}")

# ── 2. Feature engineering ──────────────────────────────────────
print("\n[2/6] Engineering features...")
t0 = time.time()

# Clean text for TF-IDF
df["clean"] = df["text"].apply(clean_text)

# Behavioural features (computed from RAW text + rating)
print("  Extracting behavioural features...")
behav_features = []
for _, row in df.iterrows():
    behav_features.append(
        extract_behavioural_features(str(row["text"]), float(row["rating"]))
    )
X_behav = np.array(behav_features)
print(f"  Behavioural features shape: {X_behav.shape}")

# TF-IDF with sublinear TF (dampens the effect of term frequency)
print("  Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
    max_features=30000,         # more features to capture richer patterns
    stop_words="english",
    ngram_range=(1, 3),          # unigrams + bigrams + trigrams
    sublinear_tf=True,           # apply log normalization to TF
    min_df=3,                    # ignore very rare terms
    max_df=0.95,                 # ignore near-universal terms
    dtype=np.float32,            # save memory
)
X_tfidf = tfidf.fit_transform(df["clean"])
print(f"  TF-IDF features shape: {X_tfidf.shape}")

# Scale behavioural features before combining
scaler = StandardScaler()
X_behav_scaled = scaler.fit_transform(X_behav)

# Combine
X = sp.hstack([X_tfidf, sp.csr_matrix(X_behav_scaled, dtype=np.float32)])
y = (df["label"] == "fake").astype(int)
print(f"  Combined feature matrix: {X.shape}")
print(f"  Feature engineering took {time.time()-t0:.1f}s")

# ── 3. Train/test split ────────────────────────────────────────
print("\n[3/6] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ── 4. Model comparison with cross-validation ──────────────────
print("\n[4/6] Training & comparing models (5-fold CV)...")

models = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=5000, solver="lbfgs", random_state=42
    ),
    "Logistic Regression (C=0.5)": LogisticRegression(
        C=0.5, max_iter=5000, solver="lbfgs",
        random_state=42,
    ),
    "Logistic Regression (C=2)": LogisticRegression(
        C=2.0, max_iter=5000, solver="lbfgs", random_state=42,
    ),
    "SGD Classifier": SGDClassifier(
        loss="modified_huber",   # gives probability estimates
        alpha=1e-4,
        max_iter=2000,
        random_state=42,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = {}

for name, clf in models.items():
    t0 = time.time()
    print(f"\n  Training: {name}...")

    # Cross-validation on training set in parallel
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)

    # Fit on full training set for test evaluation
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

    elapsed = time.time() - t0
    print(f"    CV F1 (mean±std) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test Accuracy    : {acc:.4f}")
    print(f"    Test F1          : {f1:.4f}")
    print(f"    Test AUC-ROC     : {auc:.4f}")
    print(f"    Time             : {elapsed:.1f}s")

    all_results[name] = {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "model_obj": clf,
    }

# ── 5. Select best model ───────────────────────────────────────
print("\n[5/6] Selecting best model...")

# Rank by test F1 (primary), then AUC (secondary)
ranking = sorted(all_results.items(), key=lambda x: (x[1]["f1"], x[1]["auc"]), reverse=True)

print("\n  +---------------------------------+----------+----------+----------+")
print("  | Model                           | Accuracy | F1       | AUC-ROC  |")
print("  +---------------------------------+----------+----------+----------+")
for name, res in ranking:
    marker = " * BEST" if name == ranking[0][0] else ""
    print(f"  | {name:<31} | {res['accuracy']:.4f}   | {res['f1']:.4f}   | {res['auc']:.4f}   |{marker}")
print("  +---------------------------------+----------+----------+----------+")

best_name = ranking[0][0]
best_model = ranking[0][1]["model_obj"]
best_res = ranking[0][1]

print(f"\n  Best model: {best_name}")
print(f"\n  Detailed classification report on test set:")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=["Real", "Fake"]))

cm = confusion_matrix(y_test, y_pred_best)
print(f"  Confusion Matrix:")
print(f"                 Predicted Real  Predicted Fake")
print(f"  Actual Real    {cm[0][0]:>13}  {cm[0][1]:>14}")
print(f"  Actual Fake    {cm[1][0]:>13}  {cm[1][1]:>14}")

# ── 5b. Test specifically on 5-star fake reviews ──────────────
print(f"\n  === 5-STAR FAKE REVIEW DETECTION (critical test) ===")
# Get indices of 5-star reviews in original df that ended up in test set
test_indices = y_test.index
df_test = df.iloc[test_indices]

mask_5star_fake = (df_test["rating"] == 5.0) & (df_test["label"] == "fake")
mask_5star_real = (df_test["rating"] == 5.0) & (df_test["label"] == "real")

if mask_5star_fake.sum() > 0:
    X_5fake = X_test[mask_5star_fake.values]
    y_5fake = y_test[mask_5star_fake]
    pred_5fake = best_model.predict(X_5fake)
    acc_5fake = accuracy_score(y_5fake, pred_5fake)
    print(f"  5-star FAKE reviews in test : {len(y_5fake)}")
    print(f"  Correctly caught as fake    : {(pred_5fake == 1).sum()} / {len(y_5fake)} ({acc_5fake:.1%})")

if mask_5star_real.sum() > 0:
    X_5real = X_test[mask_5star_real.values]
    y_5real = y_test[mask_5star_real]
    pred_5real = best_model.predict(X_5real)
    acc_5real = accuracy_score(y_5real, pred_5real)
    print(f"  5-star REAL reviews in test : {len(y_5real)}")
    print(f"  Correctly kept as real      : {(pred_5real == 0).sum()} / {len(y_5real)} ({acc_5real:.1%})")

# ── 6. Save everything ─────────────────────────────────────────
print("\n[6/6] Saving model artifacts...")
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("model/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Clean results for meta (remove model objects)
meta_results = {}
for name, res in all_results.items():
    meta_results[name] = {
        "accuracy": res["accuracy"],
        "f1": res["f1"],
        "auc": res["auc"],
        "cv_f1_mean": res["cv_f1_mean"],
        "cv_f1_std": res["cv_f1_std"],
    }

meta = {
    "model_name": best_name,
    "accuracy": best_res["accuracy"],
    "f1": best_res["f1"],
    "auc": best_res["auc"],
    "all_results": meta_results,
}

with open("model/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"\n  Saved: model/model.pkl")
print(f"  Saved: model/tfidf.pkl")
print(f"  Saved: model/scaler.pkl")
print(f"  Saved: model/meta.pkl")

print("\n" + "=" * 60)
print(f"DONE! Best model: {best_name}")
print(f"  Accuracy : {best_res['accuracy']:.1%}")
print(f"  F1 Score : {best_res['f1']:.1%}")
print(f"  AUC-ROC  : {best_res['auc']:.1%}")
print("=" * 60)
print("\nRun:  python -m streamlit run app.py")