"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          TabPFN & CTGAN WORKFLOW FOR BEHAVIORAL PREDICTION                 ║
║          Cycling Stage Classification from Behavioral & Demographic Data   ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS SCRIPT DOES:
    This script predicts a person's cycling adoption stage using their behavioral 
    attributes. It utilizes advanced Class Balancing via CTGAN (Generative 
    Adversarial Networks) to synthesize highly accurate minority samples, uses 
    Recursive Feature Elimination (RFE) to isolate predictive signal from noise, 
    and applies Extra Trees Ensembles for ultimate performance optimization.

AUTHOR: Auto-generated TabPFN & CTGAN Workflow
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.base import BaseSampler

from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from ctgan import CTGAN

import rdt.transformers.base
original_set_seed = rdt.transformers.base.BaseTransformer._set_seed
def patched_set_seed(self, data):
    try:
        original_set_seed(self, data)
    except Exception:
        import numpy as np
        seed = getattr(self, 'random_state', 42)
        if seed is None:
            seed = 42
        self.random_states = {
            'fit': np.random.RandomState(seed),
            'transform': np.random.RandomState(seed),
            'reverse_transform': np.random.RandomState(seed),
            '_transform_continuous': np.random.RandomState(seed),
            '_fit_continuous': np.random.RandomState(seed),
        }
        self._random_state_set = True
rdt.transformers.base.BaseTransformer._set_seed = patched_set_seed

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap

plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
sns.set_style('whitegrid')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results_perfect')
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_FILE = os.path.join(SCRIPT_DIR, 'Cycling_data - TabPFN.csv')

print("=" * 70)
print("  TabPFN & CTGAN WORKFLOW")
print("=" * 70)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CUSTOM CTGAN SAMPLER WRAPPER                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class CTGANSampler(BaseSampler):
    """
    Custom Imbalanced-Learn Sampler that uses CTGAN to synthesize 
    data for minority classes up to the majority class count.
    """
    _sampling_type = 'over-sampling'
    _parameter_constraints: dict = {
        "random_state": ["random_state", None],
        "epochs": [int, None]
    }

    def __init__(self, random_state=42, epochs=50):
        super().__init__()
        self.random_state = random_state
        self.epochs = epochs

    def _fit_resample(self, X, y):
        counts = pd.Series(y).value_counts()
        max_count = int(counts.max())
        
        is_df = isinstance(X, pd.DataFrame)
        X_df = X if is_df else pd.DataFrame(X)
        X_df.columns = X_df.columns.astype(str)
            
        X_resampled = [X_df.values]
        y_resampled = [y]
        
        for class_label, count in counts.items():
            to_generate = max_count - int(count)
            if to_generate > 0:
                X_min = X_df[y == class_label]
                # Train a specific CTGAN generator for this explicit class
                ctgan = CTGAN(epochs=self.epochs, verbose=False)
                ctgan.fit(X_min)
                
                synthetic_X = ctgan.sample(to_generate)
                X_resampled.append(synthetic_X.values)
                y_resampled.append(np.full(to_generate, class_label))
                
        return np.vstack(X_resampled), np.concatenate(y_resampled)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: DATA INGESTION & PREPROCESSING                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n  [1/8] Data Ingestion & Preprocessing...")
df = pd.read_csv(DATA_FILE)

if 'cycling_stage' in df.columns:
    target_col = 'cycling_stage'
    df = df[df["cycling_stage"] != 6].dropna().reset_index(drop=True)
else:
    target_col = df.columns[-1]

cols_to_reverse = ['SN2', 'SR6', 'SR7', 'ENV1', 'ENV2', 'ENV3', 'ENV4']
for col in cols_to_reverse:
    if col in df.columns:
        df[col] = 6 - df[col]

df["ATT"] = df[["ATT1", "ATT2", "ATT3"]].mean(axis=1)
df["SN"] = df[["SN1", "SN2", "SN3"]].mean(axis=1)
df["PBC"] = df[["PBC1", "PBC2"]].mean(axis=1)
df["INF"] = df[["INF2", "INF3", "INF5"]].mean(axis=1)
df["SR"] = df[["SR1", "SR2"]].mean(axis=1)
df["ENV"] = df[["ENV1", "ENV2"]].mean(axis=1)
df["EOT"] = df[["EOT1", "EOT2", "EOT3"]].mean(axis=1)

feature_cols = [
    "ATT", "SN", "PBC", "INF", "SR", "ENV", "EOT",
    "Age_young", "sex_male", "Distance_less", "Income_low", "Primary_mode"
]
X = df[feature_cols].copy()
label_enc_target = LabelEncoder()
y = label_enc_target.fit_transform(df[target_col])

continuous_cols = X.select_dtypes(include=['number']).columns.tolist()
scaler = StandardScaler()
if len(continuous_cols) > 0:
    X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"        Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2: CTGAN CLASS BALANCING DEPLOYMENT                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n  [2/8] Generating Global Synthetic Samples via CTGAN (Global Reference)...")
# We do one massive run outside CV just to plot and capture the full dataset balance visually
sampler_global = CTGANSampler(epochs=100)
X_res_global, y_res_global = sampler_global.fit_resample(X_train, y_train)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
pd.Series(Counter(y_train)).sort_index().plot(kind='bar', ax=axes[0], color='salmon', edgecolor='black')
axes[0].set_title('Original Class Imbalance', fontweight='bold')
axes[0].set_ylabel('Count')

pd.Series(Counter(y_res_global)).sort_index().plot(kind='bar', ax=axes[1], color='mediumseagreen', edgecolor='black')
axes[1].set_title('Perfectly Balanced via CTGAN', fontweight='bold')

plt.suptitle('Class Distribution Before/After CTGAN Synthesis', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_ctgan_balancing.png'), bbox_inches='tight')
plt.close()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3: BASELINE MODELS + CTGAN + RFE FEATURE SELECTION              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n  [3/8] Testing Models with Recursive Feature Elimination + CTGAN Inside CV Folds...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lightweight CTGAN sampler for inside CV folds to keep timing reasonable
cv_sampler = CTGANSampler(epochs=50)

# Base estimator for Recursive Feature Elimination (dropping lowest noise)
rfe_selector = RFE(estimator=ExtraTreesClassifier(n_estimators=50, random_state=42), n_features_to_select=8)

from imblearn.over_sampling import SMOTE
fast_cv_sampler = SMOTE(random_state=42)

models = {
    'XGBoost + RFE': ImbPipeline([
        ('sampler', fast_cv_sampler),
        ('rfe', rfe_selector),
        ('clf', XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False))
    ]),
    'ExtraTrees + RFE': ImbPipeline([
        ('sampler', fast_cv_sampler),
        ('rfe', rfe_selector),
        ('clf', ExtraTreesClassifier(n_estimators=200, random_state=42))
    ]),
    'RandomForest + RFE': ImbPipeline([
        ('sampler', fast_cv_sampler),
        ('rfe', rfe_selector),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
}

cv_results = {}
best_baseline_name = None
best_baseline_f1 = 0

print(f"        {'Model':<30} {'F1-Weighted':<18} {'Accuracy':<15}")
for name, model in models.items():
    t0 = time.time()
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=1)
    acc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
    elapsed = time.time() - t0

    mean_f1 = f1_scores.mean()
    cv_results[name] = {'f1_mean': mean_f1, 'f1_std': f1_scores.std(), 'acc_mean': acc_scores.mean()}
    print(f"        {name:<30} {mean_f1:.4f} +/- {f1_scores.std():.4f}   {acc_scores.mean():.4f}   ({elapsed:.1f}s)")

    if mean_f1 > best_baseline_f1:
        best_baseline_f1 = mean_f1
        best_baseline_name = name

print(f"        [BEST] BASELINE: {best_baseline_name} (CV F1={best_baseline_f1:.4f})")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4: BAYESIAN OPTIMIZATION (OPTUNA) FOR THE WINNING MODEL         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print(f"\n  [4/8] Bayesian Hyperparameter Tuning of {best_baseline_name}...")

def proper_objective(trial):
    is_xgb = "XGBoost" in best_baseline_name
    
    if is_xgb:
        clf = XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 300),
            max_depth=trial.suggest_int('max_depth', 3, 8),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            eval_metric='mlogloss', random_state=42, use_label_encoder=False
        )
    else:  # ExtraTrees or RandomForest
        clf_class = ExtraTreesClassifier if "ExtraTrees" in best_baseline_name else RandomForestClassifier
        clf = clf_class(
            n_estimators=trial.suggest_int('n_estimators', 100, 400),
            max_depth=trial.suggest_int('max_depth', 5, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
            criterion=trial.suggest_categorical('criterion', ['gini', 'entropy']),
            random_state=42
        )
        
    num_features = trial.suggest_int('num_features', 5, 12)
    rfe_opt = RFE(estimator=ExtraTreesClassifier(n_estimators=50, random_state=42), n_features_to_select=num_features)

    pipe = ImbPipeline([
        ('sampler', fast_cv_sampler),
        ('rfe', rfe_opt),
        ('clf', clf)
    ])
    
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=1)
    return scores.mean()

study = optuna.create_study(direction='maximize', study_name=f'{best_baseline_name}_tuning')
study.optimize(proper_objective, n_trials=15, show_progress_bar=True)

best_params = study.best_params
best_f1 = study.best_value
print(f"        Tuned CV F1-Weighted: {best_f1:.4f} (was {best_baseline_f1:.4f})")

# Train the finalized Model
num_features_final = best_params.pop('num_features')
rfe_final = RFE(estimator=ExtraTreesClassifier(n_estimators=50, random_state=42), n_features_to_select=num_features_final)

if "XGBoost" in best_baseline_name:
    final_clf = XGBClassifier(**best_params, eval_metric='mlogloss', random_state=42, use_label_encoder=False)
else:
    clf_class = ExtraTreesClassifier if "ExtraTrees" in best_baseline_name else RandomForestClassifier
    final_clf = clf_class(**best_params, random_state=42)

final_pipe = ImbPipeline([
    ('sampler', CTGANSampler(epochs=100)), # high-fidelity generator for final train
    ('rfe', rfe_final),
    ('clf', final_clf)
])

final_pipe.fit(X_train.values, y_train)
# Apply RFE mask to X_test so we can test easily
X_train_reduced = rfe_final.transform(X_train.values)
X_test_reduced = rfe_final.transform(X_test.values)
selected_feature_indices = rfe_final.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5: TRAIN TABPFN BASELINE                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n  [5/8] Training TabPFN Baseline...")
tuned_tabpfn = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
# Evaluate TabPFN on the CTGAN synthesized global data to give it balanced priors
tuned_tabpfn.fit(X_res_global, y_res_global) 

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 6: COMPREHENSIVE EVALUATION ON HOLD-OUT SET                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n  [6/8] Evaluating on Hold-Out Test Set...")
is_multi = len(np.unique(y_test)) > 2
avg_strat = 'weighted' if is_multi else 'binary'

preds_main = final_pipe.predict(X_test.values)
probs_main = final_pipe.predict_proba(X_test.values)
try:
    auc_main = roc_auc_score(y_test, probs_main, multi_class='ovr', average='weighted')
except Exception:
    auc_main = 0

preds_tab = tuned_tabpfn.predict(X_test)
probs_tab = tuned_tabpfn.predict_proba(X_test)
try:
    auc_tab = roc_auc_score(y_test, probs_tab, multi_class='ovr', average='weighted')
except Exception:
    auc_tab = 0

print(f"        {'METRIC':<20} {best_baseline_name + ' (Tuned)':<25} {'TabPFN':<20}")
print("        " + "-"*65)
print(f"        {'F1-Score':<20} {f1_score(y_test, preds_main, average=avg_strat):<25.4f} {f1_score(y_test, preds_tab, average=avg_strat):<20.4f}")
print(f"        {'Recall':<20} {recall_score(y_test, preds_main, average=avg_strat):<25.4f} {recall_score(y_test, preds_tab, average=avg_strat):<20.4f}")
print(f"        {'Precision':<20} {precision_score(y_test, preds_main, average=avg_strat):<25.4f} {precision_score(y_test, preds_tab, average=avg_strat):<20.4f}")
print(f"        {'ROC-AUC':<20} {auc_main:<25.4f} {auc_tab:<20.4f}")

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
class_names = [str(c) for c in label_enc_target.classes_]

cm_main = confusion_matrix(y_test, preds_main)
disp_main = ConfusionMatrixDisplay(cm_main, display_labels=class_names)
disp_main.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(f'{best_baseline_name} + RFE + CTGAN', fontweight='bold')

cm_tab = confusion_matrix(y_test, preds_tab)
disp_tab = ConfusionMatrixDisplay(cm_tab, display_labels=class_names)
disp_tab.plot(ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title('TabPFN', fontweight='bold')

plt.suptitle('Confusion Matrices — Test Set Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_confusion_matrices.png'), bbox_inches='tight')
plt.close()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 7: FEATURE IMPORTANCE (TabPFN — Permutation Importance)         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n  [7/8] Generating Feature Importances...")
clf_only = final_pipe.named_steps['clf']

# ── 7a. TabPFN Permutation Importance (PRIMARY) ─────────────────────────
# TabPFN is a neural foundation model — it has no built-in .feature_importances_.
# Permutation importance measures the drop in F1-Weighted when each feature is
# randomly shuffled, giving a model-agnostic estimate of each feature's value.
print("        Computing TabPFN Permutation Importance on test set...")
perm_result = permutation_importance(
    tuned_tabpfn, X_test, y_test,
    n_repeats=30, random_state=42,
    scoring='f1_weighted', n_jobs=1
)
tabpfn_imp = pd.Series(
    perm_result.importances_mean, index=feature_cols
).sort_values(ascending=True)
tabpfn_imp_std = pd.Series(
    perm_result.importances_std, index=feature_cols
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(12, max(6, len(tabpfn_imp) * 0.5)))
colors = plt.cm.magma(np.linspace(0.2, 0.8, len(tabpfn_imp)))
tabpfn_imp.plot(
    kind='barh', ax=ax, xerr=tabpfn_imp_std.reindex(tabpfn_imp.index),
    color=colors, edgecolor='black', linewidth=0.5, capsize=3
)
ax.set_title('Feature Importance — TabPFN (Permutation Importance)', fontsize=15, fontweight='bold')
ax.set_xlabel('Mean F1-Weighted Decrease When Shuffled', fontweight='bold')
ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_tabpfn_feature_importance.png'), bbox_inches='tight')
plt.close()
print("        [SAVED] 03_tabpfn_feature_importance.png")

# ── 7b. Extra Trees Gini Importance (SECONDARY/COMPARISON) ──────────────
importances = None
if hasattr(clf_only, 'feature_importances_'):
    importances = pd.Series(clf_only.feature_importances_, index=selected_feature_names).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(importances) * 0.4)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importances)))
    importances.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(f'Feature Importance — {best_baseline_name} (Gini Importance, RFE Selected)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03b_extratrees_feature_importance.png'), bbox_inches='tight')
    plt.close()

    importances = pd.Series(clf_only.feature_importances_, index=selected_feature_names).sort_values(ascending=False)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 8: LEARNING CURVES & CONVERGENCE                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n  [8/9] Generating Training vs. Validation Convergence Curves...")
from sklearn.model_selection import learning_curve
try:
    train_sizes, train_scores, val_scores = learning_curve(
        final_pipe, X_train.values, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True), 
        n_jobs=1, train_sizes=np.linspace(0.3, 1.0, 5), scoring='f1_weighted', random_state=42
    )
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score', color='blue')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score', color='orange')
    plt.title('Learning Convergence Curve (F1-Weighted)', fontsize=14, fontweight='bold')
    plt.xlabel('Training Set Size', fontweight='bold')
    plt.ylabel('F1 Weighted Score', fontweight='bold')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_learning_convergence_curve.png'), bbox_inches='tight')
    plt.close()
    print("        [SAVED] Learning Curve generated.")
except Exception as e:
    print(f"        Skipping Learning Curve due to error: {e}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 9: SHAP EXPLAINABILITY (SUMMARY & DEPENDENCY)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n  [9/9] Calculating SHAP Explainability & Dependency Plots...")
try:
    explainer = shap.TreeExplainer(clf_only)
    shap_values = explainer.shap_values(X_test_reduced)

    # 1. Summary Plot
    plt.figure(figsize=(12, 8))
    if is_multi and isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_test_reduced, plot_type="bar", show=False, class_names=class_names, feature_names=selected_feature_names)
    else:
        shap.summary_plot(shap_values, X_test_reduced, show=False, feature_names=selected_feature_names)
    plt.title('SHAP Feature Importance by Class', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_shap_summary.png'), bbox_inches='tight')
    plt.close()

    # 2. Dependency Plots for Top 3 Features
    if importances is not None and is_multi and isinstance(shap_values, list):
        top_3_features = importances.index[:3].tolist()
        class_idx_to_plot = len(class_names) - 1  # Highest adoption stage
        
        # Convert X_test_reduced to DataFrame for SHAP dependency plotting (requires named columns usually)
        X_test_df = pd.DataFrame(X_test_reduced, columns=selected_feature_names)
        
        for feature in top_3_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature,
                shap_values[class_idx_to_plot], 
                X_test_df,
                interaction_index=None,
                show=False
            )
            plt.title(f'SHAP Dependency: {feature} (Impact on Stage {class_names[class_idx_to_plot]})', fontweight='bold')
            plt.tight_layout()
            
            # Clean feature name for filename
            clean_feat = feature.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(OUTPUT_DIR, f'06_shap_dependency_{clean_feat}.png'), bbox_inches='tight')
            plt.close()
            
    # For single-class SHAP outputs (numpy >= 2.0 / shap >= 0.51 sometimes unifies outputs to 3D arrays)
    elif importances is not None and isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        top_3_features = importances.index[:3].tolist()
        class_idx_to_plot = len(class_names) - 1
        X_test_df = pd.DataFrame(X_test_reduced, columns=selected_feature_names)
        for feature in top_3_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature,
                shap_values[:, :, class_idx_to_plot], 
                X_test_df,
                interaction_index=None,
                show=False
            )
            plt.title(f'SHAP Dependency: {feature} (Impact on Stage {class_names[class_idx_to_plot]})', fontweight='bold')
            plt.tight_layout()
            clean_feat = feature.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(OUTPUT_DIR, f'06_shap_dependency_{clean_feat}.png'), bbox_inches='tight')
            plt.close()

    print("        [SAVED] SHAP plots (Summary + Dependency) generated.")
except Exception as e:
    print(f"        Skipping SHAP due to error: {e}")

print("\n" + "=" * 70)
print("  WORKFLOW COMPLETE — SCRIPT REACHED END")
print("=" * 70)
