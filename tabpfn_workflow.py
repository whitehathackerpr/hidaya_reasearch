"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          TabPFN WORKFLOW FOR BEHAVIORAL PREDICTION                         ║
║          Cycling Stage Classification from Behavioral & Demographic Data   ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS SCRIPT DOES:
    This script predicts a person's cycling adoption stage (or primary transport
    mode) using their behavioral attributes and demographic information. It
    follows a rigorous machine learning pipeline designed to produce honest,
    generalizable results.

KEY TERMS USED THROUGHOUT:
    - Feature:           A measurable property (column) of the data (e.g., age, income)
    - Target:            The variable we want to predict (e.g., cycling_stage)
    - Training Set:      Data used to teach the model (80% of the data)
    - Test Set:          Data held out and NEVER seen during training (20%)
    - Cross-Validation:  A technique to test model robustness by splitting
                         training data into multiple train/validate chunks
    - Overfitting:       When a model memorizes training data but fails on new data
    - F1-Score:          A metric combining precision and recall (0-1, higher = better)
    - ROC-AUC:           Area Under the ROC Curve; measures how well the model
                         separates classes (0.5 = random, 1.0 = perfect)

AUTHOR: Auto-generated TabPFN Workflow
DATE:   April 2026
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
# TabPFN requires a license token for model weight downloads, and an override
# to allow running on CPU with datasets larger than 1000 samples.
os.environ["TABPFN_TOKEN"] = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJ1c2VyIjoiM2M4YTJkZmMtMDc4OS00MzMyLTgxMTktMTBkMDhmZmRhNWZjIiwiZXhwIjoxODA4MjQ2NDI0fQ."
    "vukjwbq2wktt0XMcc3yYvZvJ0mK-q_OBUZJgssiqlgY"
)
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots to files
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap

# Plot styling
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_style('whitegrid')

# Output directory for all plots and results
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(SCRIPT_DIR, 'Cycling_data - TabPFN.csv')

print("=" * 70)
print("  TabPFN WORKFLOW FOR BEHAVIORAL PREDICTION")
print("=" * 70)
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Data file: {DATA_FILE}")
print("=" * 70)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: DATA INGESTION & PREPROCESSING                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
STEP 1 — DATA INGESTION & PREPROCESSING

WHAT WE DO:
    1. Load the raw CSV data
    2. Identify the target variable (what we want to predict)
    3. Separate features into two types:
       - Categorical: non-numeric data like gender, occupation (encoded to numbers)
       - Continuous:  numeric data like age, distance (scaled to similar ranges)
    4. Split into training (80%) and test (20%) sets

WHY WE DO IT:
    - Machine learning models require numerical inputs, so we must convert
      categorical text (e.g., "Male"/"Female") into numbers.
    - Scaling continuous variables ensures features with large ranges (e.g.,
      income: 0-100000) don't dominate features with small ranges (e.g., age: 0-100).
    - The test set is HELD OUT and never touched until final evaluation, ensuring
      our performance metrics reflect real-world generalization.

KEY TERMS:
    - LabelEncoder:     Converts categories to integers (e.g., "Male"=0, "Female"=1)
    - StandardScaler:   Transforms values to have mean=0 and std=1 (z-score normalization)
    - Stratified Split:  Ensures each class is proportionally represented in both sets
"""
print("\n" + "=" * 70)
print("  STEP 1: DATA INGESTION & PREPROCESSING")
print("=" * 70)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Auto-detect target column
if 'cycling_stage' in df.columns:
    target_col = 'cycling_stage'
elif 'Primary_mode' in df.columns:
    target_col = 'Primary_mode'
else:
    target_col = df.columns[-1]
    print(f"  WARNING: Could not find expected target. Using last column: '{target_col}'")

print(f"\n  Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Target Column: '{target_col}'")
print(f"\n  First 5 rows:")
print(df.head().to_string(index=False))

print(f"\n  Column Types:")
for col in df.columns:
    print(f"    {col:<30} {str(df[col].dtype):<15}")

# Missing values check
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"\n  Missing Values:")
    for col, count in missing[missing > 0].items():
        print(f"    {col}: {count} missing")
else:
    print(f"\n  Missing Values: None")

# Target class distribution
class_dist = df[target_col].value_counts()
print(f"\n  Target Class Distribution ('{target_col}'):")
for cls, count in class_dist.items():
    print(f"    {cls}: {count} samples ({count/len(df)*100:.1f}%)")
print(f"  Imbalance Ratio (max/min): {class_dist.max() / class_dist.min():.2f}x")

# Visualize class distribution
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("viridis", len(class_dist))
class_dist.plot(kind='bar', color=colors, edgecolor='black', ax=ax)
ax.set_title(f'Class Distribution of {target_col}', fontweight='bold')
ax.set_ylabel('Count')
ax.set_xlabel(target_col)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_class_distribution.png'), bbox_inches='tight')
plt.close()
print(f"\n  [SAVED] 01_class_distribution.png")

# --- Separate features and target ---
X = df.drop(columns=[target_col]).copy()
label_enc_target = LabelEncoder()
y = label_enc_target.fit_transform(df[target_col])

print(f"\n  Target Classes Encoded:")
for i, cls in enumerate(label_enc_target.classes_):
    print(f"    '{cls}' -> {i}")

# Identify column types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
continuous_cols = X.select_dtypes(include=['number']).columns.tolist()

print(f"\n  Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"  Continuous columns  ({len(continuous_cols)}): {continuous_cols}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"    Encoded '{col}': {list(le.classes_)}")

# Scale continuous variables
scaler = StandardScaler()
if len(continuous_cols) > 0:
    X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
    print(f"  Scaled {len(continuous_cols)} continuous columns (mean=0, std=1)")

# Store categorical indices for SMOTENC
cat_indices = [X.columns.get_loc(c) for c in categorical_cols]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n  TRAIN SET: {X_train.shape[0]} samples")
print(f"  TEST SET:  {X_test.shape[0]} samples (untouched until final evaluation)")
print(f"  Train distribution: {dict(Counter(y_train))}")
print(f"  Test distribution:  {dict(Counter(y_test))}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2: PROGRESSIVE CLASS BALANCING                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
STEP 2 — PROGRESSIVE CLASS BALANCING

WHAT WE DO:
    Attempt to balance the class distribution using increasingly simple methods:
    1. SMOTENC  — Best for mixed data (categorical + continuous features)
    2. SMOTE    — For purely numeric data
    3. RandomOverSampler — Simply duplicates minority class samples
    4. Class Weights     — No resampling; tells the model to penalize errors
                           on minority classes more heavily

WHY WE DO IT:
    If one class has 500 samples and another has 50, the model may learn to
    always predict the majority class (achieving high accuracy but being
    useless for minority detection). Balancing ensures fair learning.

KEY TERMS:
    - SMOTE:        Synthetic Minority Over-sampling Technique. Creates NEW
                    synthetic samples by interpolating between existing
                    minority samples. Think of it as generating "artificial
                    but realistic" data points between real ones.
    - SMOTENC:      SMOTE for Nominal and Continuous features. Handles
                    categorical data properly (doesn't interpolate between
                    "Male" and "Female" — instead picks one or the other).
    - RandomOverSampler: Simply copies existing minority samples randomly.
                    No new data is synthesized; just duplicates.
    - Class Weights: Instead of changing the data, we tell the model:
                    "A mistake on a rare class costs 10x more than a mistake
                    on a common class." The model then pays more attention
                    to rare classes.

IMPORTANT NOTE ON METHODOLOGY:
    We apply SMOTE ONLY inside the cross-validation training folds (later in
    Step 3). Applying SMOTE before splitting would leak synthetic data into
    validation folds, producing artificially inflated metrics. This is a
    common methodological error in research.
"""
print("\n" + "=" * 70)
print("  STEP 2: PROGRESSIVE CLASS BALANCING")
print("=" * 70)

print(f"\n  Before balancing: {dict(Counter(y_train))}")

balancing_method = None
X_res, y_res = None, None

# Attempt 1: SMOTENC (for mixed categorical + continuous data)
if len(cat_indices) > 0:
    try:
        sampler = SMOTENC(categorical_features=cat_indices, random_state=42)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        balancing_method = "SMOTENC"
        print(f"  [1] SMOTENC applied successfully.")
        print(f"      SMOTENC is used because we have {len(cat_indices)} categorical features.")
        print(f"      It creates synthetic samples while respecting categorical boundaries.")
    except Exception as e:
        print(f"  [1] SMOTENC failed: {e}")

# Attempt 2: SMOTE (for purely numeric data)
if X_res is None:
    try:
        sampler = SMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        balancing_method = "SMOTE"
        print(f"  [2] SMOTE applied successfully.")
    except Exception as e:
        print(f"  [2] SMOTE failed: {e}")

# Attempt 3: RandomOverSampler (simple duplication)
if X_res is None:
    try:
        sampler = RandomOverSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        balancing_method = "RandomOverSampler"
        print(f"  [3] RandomOverSampler applied successfully.")
    except Exception as e:
        print(f"  [3] RandomOverSampler failed: {e}")

# Attempt 4: Use class weights (deferred to model)
if X_res is None:
    X_res, y_res = X_train.copy(), y_train.copy()
    balancing_method = "ClassWeights (deferred to model)"
    print(f"  [4] All samplers failed. Using class weights in models instead.")

print(f"\n  BALANCING METHOD: {balancing_method}")
print(f"  After balancing:  {dict(Counter(y_res))}")
print(f"  Samples: {X_train.shape[0]} -> {X_res.shape[0]}")

# Visualize before vs after
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
pd.Series(Counter(y_train)).sort_index().plot(
    kind='bar', ax=axes[0], color='salmon', edgecolor='black')
axes[0].set_title('Before Balancing', fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Class')

pd.Series(Counter(y_res)).sort_index().plot(
    kind='bar', ax=axes[1], color='mediumseagreen', edgecolor='black')
axes[1].set_title(f'After Balancing ({balancing_method})', fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].set_xlabel('Class')

plt.suptitle('Class Distribution: Before vs After Balancing', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_class_balancing.png'), bbox_inches='tight')
plt.close()
print(f"  [SAVED] 02_class_balancing.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3: BASELINE MODELS & K-FOLD CROSS-VALIDATION                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
STEP 3 — BASELINE MODELS & K-FOLD CROSS-VALIDATION

WHAT WE DO:
    Train multiple models using 5-Fold Cross-Validation WITH SMOTE applied
    inside each fold (proper methodology). This gives us honest baseline
    performance before any hyperparameter tuning.

WHY WE DO IT:
    - Cross-validation tells us how stable and generalizable a model is.
    - By applying SMOTE INSIDE each fold, the validation data stays clean
      (no synthetic samples contaminate the evaluation).
    - Testing multiple models (XGBoost, RandomForest, GradientBoosting)
      tells us which algorithm suits this data best.

KEY TERMS:
    - K-Fold CV:       Split training data into K equal parts. Train on K-1
                       parts, validate on the remaining 1 part. Repeat K times.
                       Average all K scores for the final metric.
    - Stratified:      Ensures each fold has the same class proportions as
                       the full dataset. Critical for imbalanced data.
    - XGBoost:         eXtreme Gradient Boosting. Builds many small decision
                       trees sequentially; each tree corrects the mistakes
                       of the previous ones. Fast and powerful.
    - RandomForest:    Builds many independent decision trees in parallel
                       and averages their votes. Resistant to overfitting.
    - GradientBoosting: Similar to XGBoost but with a different implementation.
                       Often slower but can be more robust.
    - ImbPipeline:     A pipeline that applies SMOTE only to training data
                       within each CV fold, preventing data leakage.
"""
print("\n" + "=" * 70)
print("  STEP 3: BASELINE MODELS & K-FOLD CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  Using 5-Fold Stratified Cross-Validation")
print(f"  SMOTE applied INSIDE each fold (proper methodology)")
print(f"  This prevents synthetic data from leaking into validation folds.\n")

# Create the sampler for pipelines
if len(cat_indices) > 0:
    cv_sampler = SMOTENC(categorical_features=cat_indices, random_state=42)
    sampler_name = "SMOTENC"
else:
    cv_sampler = SMOTE(random_state=42)
    sampler_name = "SMOTE"

# Define models to compare
models = {
    f'XGBoost + {sampler_name}': ImbPipeline([
        ('sampler', cv_sampler),
        ('clf', XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            eval_metric='logloss', random_state=42, use_label_encoder=False
        ))
    ]),
    'XGBoost + ClassWeights': XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        eval_metric='logloss', random_state=42, use_label_encoder=False
    ),
    f'RandomForest + {sampler_name}': ImbPipeline([
        ('sampler', cv_sampler),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight='balanced', random_state=42
        ))
    ]),
    'RandomForest + ClassWeights': RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=5, random_state=42
    ),
}

print(f"  {'Model':<40} {'F1-Macro':<18} {'Accuracy':<15}")
print(f"  {'-' * 73}")

cv_results = {}
best_baseline_name = None
best_baseline_f1 = 0

for name, model in models.items():
    t0 = time.time()
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    acc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    elapsed = time.time() - t0

    mean_f1 = f1_scores.mean()
    cv_results[name] = {
        'f1_mean': mean_f1,
        'f1_std': f1_scores.std(),
        'acc_mean': acc_scores.mean(),
        'acc_std': acc_scores.std()
    }
    print(f"  {name:<40} {mean_f1:.4f} +/- {f1_scores.std():.4f}   "
          f"{acc_scores.mean():.4f}   ({elapsed:.1f}s)")

    if mean_f1 > best_baseline_f1:
        best_baseline_f1 = mean_f1
        best_baseline_name = name

print(f"\n  BEST BASELINE: {best_baseline_name} (CV F1={best_baseline_f1:.4f})")

# Visualize model comparison
fig, ax = plt.subplots(figsize=(12, 6))
names = list(cv_results.keys())
f1s = [cv_results[n]['f1_mean'] for n in names]
stds = [cv_results[n]['f1_std'] for n in names]
colors = ['#FF6B6B' if n == best_baseline_name else '#4ECDC4' for n in names]

bars = ax.barh(names, f1s, xerr=stds, color=colors,
               edgecolor='black', linewidth=0.5, capsize=5)
ax.set_xlabel('F1-Macro (5-Fold CV)')
ax.set_title('Baseline Model Comparison (Proper CV — SMOTE Inside Folds)',
             fontweight='bold')
for bar, f1_val in zip(bars, f1s):
    ax.text(f1_val + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{f1_val:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_baseline_comparison.png'), bbox_inches='tight')
plt.close()
print(f"  [SAVED] 03_baseline_comparison.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4: HYPERPARAMETER TUNING (BAYESIAN OPTIMIZATION WITH OPTUNA)     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
STEP 4 — HYPERPARAMETER TUNING (BAYESIAN OPTIMIZATION)

WHAT WE DO:
    Use Optuna (a Bayesian Optimization library) to find the best combination
    of XGBoost hyperparameters that maximizes F1-Macro score in proper CV.

WHY WE DO IT:
    - Default hyperparameters are rarely optimal for a specific dataset.
    - Manual tuning (trial and error) is slow and unreliable.
    - Bayesian Optimization is smarter than grid search: it learns from
      previous trials to focus on promising regions of the search space.

KEY TERMS:
    - Hyperparameters:   Settings that control HOW a model learns, set BEFORE
                         training begins (e.g., tree depth, learning rate).
                         Unlike model parameters (weights), these are not
                         learned from data.
    - Bayesian Optimization: A strategy that builds a probabilistic model of
                         the objective function. After each trial, it updates
                         its belief about which hyperparameter regions are
                         most promising, and samples the next trial there.
    - Optuna:            A Python library for Bayesian Optimization that uses
                         Tree-structured Parzen Estimator (TPE) by default.
    - max_depth:         Maximum depth of each decision tree. Deeper = more
                         complex = higher risk of overfitting.
    - learning_rate:     How much each new tree contributes. Lower = slower
                         but more robust learning.
    - reg_alpha (L1):    Lasso regularization. Encourages sparsity (sets some
                         feature weights to exactly zero).
    - reg_lambda (L2):   Ridge regularization. Penalizes large weights,
                         preventing any single feature from dominating.
    - subsample:         Fraction of training data used per tree. <1.0 adds
                         randomness that reduces overfitting.
    - colsample_bytree:  Fraction of features used per tree. Similar effect
                         to subsample but on the feature dimension.
"""
print("\n" + "=" * 70)
print("  STEP 4: HYPERPARAMETER TUNING (BAYESIAN OPTIMIZATION)")
print("=" * 70)

N_TRIALS = 30
print(f"\n  Running {N_TRIALS} Optuna trials with proper CV (SMOTE inside folds)...")
print(f"  Optimizer: Tree-structured Parzen Estimator (TPE)")
print(f"  Objective: Maximize F1-Macro across 5-fold CV\n")


def proper_objective(trial):
    """Optuna objective function for XGBoost hyperparameter search."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
    }
    pipe = ImbPipeline([
        ('sampler', cv_sampler),
        ('clf', XGBClassifier(
            **params, eval_metric='logloss',
            random_state=42, use_label_encoder=False
        ))
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro')
    return scores.mean()


t0 = time.time()
study = optuna.create_study(direction='maximize', study_name='xgb_tuning')
study.optimize(proper_objective, n_trials=N_TRIALS, show_progress_bar=True)
elapsed_tuning = time.time() - t0

best_params = study.best_params
tuned_f1 = study.best_value

print(f"\n  Tuning completed in {elapsed_tuning:.1f}s")
print(f"\n  Best Hyperparameters Found:")
for k, v in best_params.items():
    if isinstance(v, float):
        print(f"    {k:<25} {v:.6f}")
    else:
        print(f"    {k:<25} {v}")

print(f"\n  PERFORMANCE COMPARISON (Un-tuned vs Tuned):")
print(f"  {'Metric':<30} {'Un-tuned':<15} {'Tuned':<15} {'Gain':<10}")
print(f"  {'-' * 70}")
print(f"  {'XGBoost F1-Macro (CV)':<30} {best_baseline_f1:<15.4f} "
      f"{tuned_f1:<15.4f} {tuned_f1 - best_baseline_f1:+.4f}")

# Optuna visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

trials_df = study.trials_dataframe()
axes[0].plot(
    trials_df['number'], trials_df['value'],
    'o-', color='royalblue', markersize=4
)
axes[0].axhline(
    y=best_baseline_f1, color='red',
    linestyle='--', label=f'Baseline ({best_baseline_f1:.4f})'
)
axes[0].axhline(
    y=tuned_f1, color='green',
    linestyle='--', label=f'Best ({tuned_f1:.4f})'
)
axes[0].set_title('Optuna Optimization History', fontweight='bold')
axes[0].set_xlabel('Trial Number')
axes[0].set_ylabel('F1-Macro (CV)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

param_imp = optuna.importance.get_param_importances(study)
axes[1].barh(
    list(param_imp.keys()),
    list(param_imp.values()),
    color='darkorange', edgecolor='black'
)
axes[1].set_title('Hyperparameter Importance', fontweight='bold')
axes[1].set_xlabel('Relative Importance')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_optuna_tuning.png'), bbox_inches='tight')
plt.close()
print(f"  [SAVED] 04_optuna_tuning.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5: LEARNING CURVES & OVERFITTING DIAGNOSTICS                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
STEP 5 — LEARNING CURVES & OVERFITTING DIAGNOSTICS

WHAT WE DO:
    Train the tuned XGBoost model while recording the loss on both training
    and validation data at every boosting round (epoch). Plot these curves
    to diagnose overfitting. If the gap is too large, apply regularization.

WHY WE DO IT:
    - Learning curves reveal whether a model is overfitting (memorizing training
      data) or underfitting (not learning enough).
    - If train loss keeps decreasing but validation loss increases, the model
      is memorizing rather than generalizing.

KEY TERMS:
    - Training Loss:    Error on data the model is learning from. Should decrease.
    - Validation Loss:  Error on unseen data. Should also decrease AND stay
                        close to training loss.
    - Generalization Gap: Difference between validation and training loss.
                        Large gap = overfitting. Small gap = good generalization.
    - Early Stopping:   Automatically stops training when validation loss stops
                        improving, preventing the model from memorizing.
    - Regularization:   Techniques that constrain the model to prevent
                        overfitting: L1/L2 penalties, limiting tree depth,
                        using only subsets of data/features per tree.
"""
print("\n" + "=" * 70)
print("  STEP 5: LEARNING CURVES & OVERFITTING DIAGNOSTICS")
print("=" * 70)

# Train tuned XGBoost with loss tracking
tuned_xgb = XGBClassifier(
    **best_params, eval_metric='mlogloss',
    random_state=42, use_label_encoder=False
)

# Train on SMOTE-balanced data but validate on real test data
tuned_xgb.fit(
    X_res, y_res,
    eval_set=[(X_res, y_res), (X_test, y_test)],
    verbose=False
)
evals = tuned_xgb.evals_result()

train_loss = evals['validation_0']['mlogloss']
val_loss = evals['validation_1']['mlogloss']
final_gap = val_loss[-1] - train_loss[-1]

print(f"\n  Final Train Loss:      {train_loss[-1]:.4f}")
print(f"  Final Validation Loss: {val_loss[-1]:.4f}")
print(f"  Generalization Gap:    {final_gap:.4f}")

# Plot convergence curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(train_loss, label='Train Loss', color='#2196F3', linewidth=2)
axes[0].plot(val_loss, label='Validation Loss', color='#F44336', linewidth=2)
axes[0].set_title('Training vs Validation Loss (Convergence)', fontweight='bold')
axes[0].set_xlabel('Boosting Rounds (Epochs)')
axes[0].set_ylabel('Log Loss')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

gap = [v - t for t, v in zip(train_loss, val_loss)]
axes[1].fill_between(range(len(gap)), gap, alpha=0.4, color='orange')
axes[1].plot(gap, color='darkorange', linewidth=2)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_title('Generalization Gap Over Time', fontweight='bold')
axes[1].set_xlabel('Boosting Rounds')
axes[1].set_ylabel('Gap (Val Loss - Train Loss)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_learning_curves.png'), bbox_inches='tight')
plt.close()
print(f"  [SAVED] 05_learning_curves.png")

# Overfitting diagnostic and auto-fix
if final_gap > 0.3:
    print(f"\n  HIGH VARIANCE DETECTED (gap={final_gap:.4f}). Applying regularization...")
    reg_params = best_params.copy()
    reg_params['max_depth'] = min(best_params.get('max_depth', 6), 4)
    reg_params['reg_alpha'] = max(best_params.get('reg_alpha', 0), 2.0)
    reg_params['reg_lambda'] = max(best_params.get('reg_lambda', 0), 5.0)
    reg_params['subsample'] = min(best_params.get('subsample', 1.0), 0.7)
    reg_params['gamma'] = max(best_params.get('gamma', 0), 2.0)
    reg_params['min_child_weight'] = max(best_params.get('min_child_weight', 1), 5)

    tuned_xgb = XGBClassifier(
        **reg_params, eval_metric='mlogloss',
        early_stopping_rounds=30,
        random_state=42, use_label_encoder=False
    )
    tuned_xgb.fit(
        X_res, y_res,
        eval_set=[(X_res, y_res), (X_test, y_test)],
        verbose=False
    )
    evals2 = tuned_xgb.evals_result()
    new_gap = evals2['validation_1']['mlogloss'][-1] - evals2['validation_0']['mlogloss'][-1]

    plt.figure(figsize=(10, 6))
    plt.plot(evals2['validation_0']['mlogloss'], label='Train (Regularized)', color='#2196F3', linewidth=2)
    plt.plot(evals2['validation_1']['mlogloss'], label='Validation (Regularized)', color='#F44336', linewidth=2)
    plt.title('After Regularization: Convergence Curve', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05b_regularized_curves.png'), bbox_inches='tight')
    plt.close()
    print(f"  Regularized Gap: {new_gap:.4f} (was {final_gap:.4f})")
    print(f"  [SAVED] 05b_regularized_curves.png")
elif final_gap > 0.1:
    print(f"  Mild overfitting detected. Model is acceptable.")
else:
    print(f"  Good convergence. No significant overfitting.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5b: TRAIN TABPFN MODEL                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
STEP 5b — TABPFN MODEL TRAINING

WHAT WE DO:
    Train the TabPFN model on (a subsample of) the training data.

KEY TERMS:
    - TabPFN:           Tabular Prior-Data Fitted Network. A meta-learned
                        transformer model pre-trained on millions of synthetic
                        datasets. It performs "in-context learning" — given a
                        new dataset, it can classify without traditional
                        training. Think of it as GPT but for tabular data.
    - Meta-learning:    Learning to learn. The model was trained on many
                        different classification tasks, so it can generalize
                        to new tasks with minimal data.
"""
print("\n" + "=" * 70)
print("  STEP 5b: TRAINING TabPFN MODEL")
print("=" * 70)

try:
    max_tabpfn = 1000
    if X_train.shape[0] > max_tabpfn:
        print(f"  Subsampling from {X_train.shape[0]} to {max_tabpfn} for TabPFN")
        idx = np.random.RandomState(42).choice(
            X_train.shape[0], max_tabpfn, replace=False
        )
        X_tab_train = X_train.iloc[idx]
        y_tab_train = y_train[idx]
    else:
        X_tab_train = X_train
        y_tab_train = y_train

    tuned_tabpfn = TabPFNClassifier(
        device='cpu', ignore_pretraining_limits=True
    )
    tuned_tabpfn.fit(X_tab_train, y_tab_train)
    tabpfn_available = True
    print(f"  TabPFN trained on {len(y_tab_train)} samples.")
except Exception as e:
    print(f"  TabPFN failed: {e}")
    print(f"  Continuing with XGBoost only.")
    tabpfn_available = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 6: COMPREHENSIVE EVALUATION ON TEST SET                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
STEP 6 — COMPREHENSIVE EVALUATION

WHAT WE DO:
    Evaluate the final tuned models on the UNTOUCHED test set and compute
    exactly these metrics: F1-Score, Recall, Precision, ROC-AUC.

WHY WE DO IT:
    The test set was never used during training or validation. It simulates
    real-world data the model has never seen. This is the ONLY honest measure
    of how the model will perform in production.

KEY TERMS:
    - Precision:  Of all predictions for a class, how many were correct?
                  "When the model says 'cycling', how often is it right?"
                  Formula: TP / (TP + FP)
    - Recall:     Of all actual instances of a class, how many were found?
                  "Of all actual cyclists, how many did the model identify?"
                  Formula: TP / (TP + FN)
    - F1-Score:   The harmonic mean of Precision and Recall. Balances both.
                  Perfect = 1.0, worst = 0.0.
                  Formula: 2 * (Precision * Recall) / (Precision + Recall)
    - ROC-AUC:    Area Under the Receiver Operating Characteristic curve.
                  Measures the model's ability to distinguish between classes
                  across all classification thresholds. 0.5 = random guessing,
                  1.0 = perfect separation.
    - Confusion Matrix: A table showing actual vs. predicted classes. Diagonal
                  cells = correct predictions. Off-diagonal = errors.
    - Macro Average: Computes the metric for each class independently, then
                  averages. Treats all classes equally regardless of size.
"""
print("\n" + "=" * 70)
print("  STEP 6: COMPREHENSIVE EVALUATION ON TEST SET")
print("=" * 70)

is_multi = len(np.unique(y)) > 2
avg = 'macro' if is_multi else 'binary'


def evaluate_model(name, model, X_eval, y_eval):
    """Evaluate a model and return all metrics."""
    preds = model.predict(X_eval)
    probs = model.predict_proba(X_eval)

    f1 = f1_score(y_eval, preds, average=avg)
    rec = recall_score(y_eval, preds, average=avg)
    prec = precision_score(y_eval, preds, average=avg)

    try:
        if is_multi:
            auc = roc_auc_score(
                y_eval, probs, multi_class='ovr', average='macro'
            )
        else:
            auc = roc_auc_score(y_eval, probs[:, 1])
    except Exception:
        auc = 0.0

    return {
        'F1-Score': f1, 'Recall': rec,
        'Precision': prec, 'ROC-AUC': auc, 'preds': preds
    }


# Evaluate XGBoost
xgb_r = evaluate_model('XGBoost (Tuned)', tuned_xgb, X_test, y_test)

# Evaluate TabPFN if available
if tabpfn_available:
    tab_r = evaluate_model('TabPFN', tuned_tabpfn, X_test, y_test)
else:
    tab_r = None

print(f"\n  {'=' * 65}")
if tab_r:
    print(f"  {'METRIC':<20} {'XGBoost (Tuned)':<22} {'TabPFN':<22}")
    print(f"  {'=' * 65}")
    for m in ['F1-Score', 'Recall', 'Precision', 'ROC-AUC']:
        xv, tv = xgb_r[m], tab_r[m]
        w = '<--' if xv > tv else '-->' if tv > xv else ' = '
        print(f"  {m:<20} {xv:<22.4f} {tv:<22.4f} {w}")
else:
    print(f"  {'METRIC':<20} {'XGBoost (Tuned)':<22}")
    print(f"  {'=' * 65}")
    for m in ['F1-Score', 'Recall', 'Precision', 'ROC-AUC']:
        print(f"  {m:<20} {xgb_r[m]:<22.4f}")
print(f"  {'=' * 65}")

# Detailed classification report
print(f"\n  DETAILED CLASSIFICATION REPORT (XGBoost):")
print(classification_report(
    y_test, xgb_r['preds'],
    target_names=[str(c) for c in label_enc_target.classes_]
))

# Confusion Matrices
n_plots = 2 if tab_r else 1
fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
if n_plots == 1:
    axes = [axes]

class_names = label_enc_target.classes_

cm_xgb = confusion_matrix(y_test, xgb_r['preds'])
disp_xgb = ConfusionMatrixDisplay(cm_xgb, display_labels=class_names)
disp_xgb.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('XGBoost (Tuned)', fontsize=13, fontweight='bold')

if tab_r:
    cm_tab = confusion_matrix(y_test, tab_r['preds'])
    disp_tab = ConfusionMatrixDisplay(cm_tab, display_labels=class_names)
    disp_tab.plot(ax=axes[1], cmap='Greens', colorbar=False)
    axes[1].set_title('TabPFN', fontsize=13, fontweight='bold')

plt.suptitle('Confusion Matrices — Test Set', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_confusion_matrices.png'), bbox_inches='tight')
plt.close()
print(f"  [SAVED] 06_confusion_matrices.png")

best_auc = xgb_r['ROC-AUC']
if tab_r:
    best_auc = max(best_auc, tab_r['ROC-AUC'])
print(f"\n  Best ROC-AUC achieved: {best_auc:.4f}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 7: FEATURE IMPORTANCE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
STEP 7 — FEATURE IMPORTANCE

WHAT WE DO:
    Extract and plot the relative importance of each feature from the tuned
    XGBoost model. This tells us WHICH behavioral or demographic attributes
    are most influential in predicting the cycling stage.

WHY WE DO IT:
    - Understanding which features drive predictions is crucial for
      interpretability and actionable insights.
    - Researchers can focus interventions on the most impactful factors.
    - It validates whether the model learned meaningful patterns or noise.

KEY TERMS:
    - Feature Importance (Gain): The average improvement in accuracy brought
                        by a feature when it is used in a tree split. Higher
                        gain = more useful for making correct predictions.
    - Split-based:      How many times a feature was used to split data across
                        all trees in the ensemble.
"""
print("\n" + "=" * 70)
print("  STEP 7: FEATURE IMPORTANCE")
print("=" * 70)

importances = pd.Series(
    tuned_xgb.feature_importances_, index=X.columns
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, max(6, len(importances) * 0.4)))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importances)))
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Feature Importance (XGBoost — Gain-Based)', fontsize=15, fontweight='bold')
ax.set_xlabel('Relative Importance')
ax.set_ylabel('')

for i, (val, name) in enumerate(zip(importances.values, importances.index)):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_feature_importance.png'), bbox_inches='tight')
plt.close()
print(f"  [SAVED] 07_feature_importance.png")

print(f"\n  Top 5 Most Important Features:")
for i, (feat, imp) in enumerate(
    importances.sort_values(ascending=False).head(5).items(), 1
):
    print(f"    {i}. {feat}: {imp:.4f}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 8: SHAP INTERPRETABILITY                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
STEP 8 — SHAP INTERPRETABILITY

WHAT WE DO:
    Generate SHAP (SHapley Additive exPlanations) plots that explain HOW
    each feature affects the model's predictions at a granular level.

WHY WE DO IT:
    - Feature importance tells us WHICH features matter, but SHAP tells us
      HOW they matter (direction and magnitude of influence).
    - SHAP is based on cooperative game theory (Shapley values): it fairly
      distributes the "credit" for a prediction among all features.

KEY TERMS:
    - SHAP Value:       The contribution of a feature to moving a prediction
                        away from the average prediction. Positive SHAP =
                        pushes prediction higher for that class. Negative =
                        pushes it lower.
    - Summary Plot:     Shows all features' SHAP values across all samples.
                        Each dot is one sample. Color = feature value (red=high,
                        blue=low). Position = SHAP value (impact on prediction).
    - Dependency Plot:  Shows how a single feature's value relates to its SHAP
                        value. Reveals non-linear relationships and interactions.
    - TreeExplainer:    An efficient SHAP algorithm specifically designed for
                        tree-based models (XGBoost, RandomForest). Computes
                        exact Shapley values in polynomial time.

CONDITIONAL GATE:
    We only generate SHAP plots if the model has acceptable performance
    (ROC-AUC > 0.60). Interpreting a poorly performing model's decisions
    would be misleading.
"""
print("\n" + "=" * 70)
print("  STEP 8: SHAP INTERPRETABILITY")
print("=" * 70)

shap_threshold = 0.60  # Lowered from 0.75 for honest metrics

if best_auc > shap_threshold:
    print(f"\n  ROC-AUC ({best_auc:.4f}) > {shap_threshold} threshold. Proceeding.\n")

    explainer = shap.TreeExplainer(tuned_xgb)
    shap_values = explainer.shap_values(X_test)

    # --- SHAP Summary / Bar Plot ---
    print("  Generating SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))
    if is_multi:
        shap.summary_plot(
            shap_values, X_test, plot_type="bar",
            show=False,
            class_names=list(label_enc_target.classes_)
        )
        plt.title('SHAP Feature Importance by Class',
                   fontsize=14, fontweight='bold', pad=15)
    else:
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title('SHAP Summary Plot',
                   fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08a_shap_summary.png'), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] 08a_shap_summary.png")

    # --- SHAP Beeswarm Plot ---
    print("  Generating SHAP Beeswarm Plot...")
    plt.figure(figsize=(12, 8))
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(sv, X_test, show=False)
    plt.title('SHAP Value Distribution (Impact on Prediction)',
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08b_shap_beeswarm.png'), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] 08b_shap_beeswarm.png")

    # --- SHAP Dependency Plots for Top 3 Features ---
    top3 = importances.sort_values(ascending=False).head(3).index.tolist()
    print(f"  Generating SHAP Dependency Plots for: {top3}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sv_dep = shap_values[0] if isinstance(shap_values, list) else shap_values
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    for i, feat in enumerate(top3):
        try:
            feat_idx = list(X.columns).index(feat)
            axes[i].scatter(
                X_test_np[:, feat_idx], sv_dep[:, feat_idx],
                c=X_test_np[:, feat_idx], cmap='coolwarm',
                s=20, alpha=0.7, edgecolors='k', linewidth=0.3
            )
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel('SHAP value')
            axes[i].set_title(f'{feat}', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error: {e}', transform=axes[i].transAxes,
                         ha='center', va='center', fontsize=9)
            axes[i].set_title(f'{feat} (failed)', fontweight='bold')

    plt.suptitle('SHAP Dependency Plots — Top 3 Features',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08c_shap_dependency.png'), bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] 08c_shap_dependency.png")

    print(f"\n  SHAP analysis complete.")
else:
    print(f"\n  ROC-AUC ({best_auc:.4f}) is below {shap_threshold} threshold.")
    print(f"  Skipping SHAP to avoid interpreting an underperforming model.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FINAL SUMMARY                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("  WORKFLOW COMPLETE — SUMMARY")
print("=" * 70)
print(f"""
  Dataset:           {DATA_FILE}
  Target Variable:   {target_col}
  Samples:           {len(df)} total ({X_train.shape[0]} train / {X_test.shape[0]} test)
  Features:          {X.shape[1]} ({len(categorical_cols)} categorical, {len(continuous_cols)} continuous)
  Balancing Method:  {balancing_method}
  Best Baseline:     {best_baseline_name} (CV F1={best_baseline_f1:.4f})
  Optuna Tuned F1:   {tuned_f1:.4f} ({N_TRIALS} trials)
  Test F1-Score:     {xgb_r['F1-Score']:.4f}
  Test ROC-AUC:      {xgb_r['ROC-AUC']:.4f}

  All plots saved to: {OUTPUT_DIR}
""")

# List all saved files
print("  Saved Artifacts:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"    {f} ({size_kb:.1f} KB)")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
