import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import spearmanr
import shap
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# HELPER: flatten SHAP values safely for any SHAP version
# ─────────────────────────────────────────────────────────────
def extract_shap_mean(shap_vals, n_features):
    """
    Handles all SHAP output shapes:
      - list of 2 arrays  (old RF binary): shape (n_samples, n_features) each
      - single 2D array   (XGB binary)   : shape (n_samples, n_features)
      - single 3D array   (new RF binary): shape (n_samples, n_features, n_classes)
    Returns 1-D array of mean |SHAP| per feature, length = n_features
    """
    if isinstance(shap_vals, list):
        # old-style list → pick class 1
        arr = np.abs(shap_vals[1])                  # (n_samples, n_features)
    else:
        arr = np.abs(shap_vals)
        if arr.ndim == 3:
            # new-style 3D → pick class 1 (last axis)
            arr = arr[:, :, 1]                      # (n_samples, n_features)
        # else already 2D
    return arr.mean(axis=0)                         # (n_features,)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('Fasting_NonFasting_9Subjects.csv')
df = df.drop(columns=['Id'])

X = df.drop(columns=['FastingStatus'])
y = df['FastingStatus']                  # binary: 1 or 2

feature_names = X.columns.tolist()
X_arr = X.values
y_arr = y.values

# XGBoost needs 0-based labels
y_xgb = (y_arr - 1).astype(int)         # 1→0, 2→1

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def cv_rf(features, X_df, y_full, cv):
    clf = RandomForestClassifier(random_state=42)
    scores = cross_val_score(clf, X_df[features], y_full,
                             cv=cv, scoring='accuracy')
    return round(float(scores.mean()), 4)

def cv_xgb(features, X_df, y_full, cv):
    clf = XGBClassifier(random_state=42,
                        eval_metric='logloss',
                        use_label_encoder=False)
    scores = cross_val_score(clf, X_df[features], y_full,
                             cv=cv, scoring='accuracy')
    return round(float(scores.mean()), 4)

# ─────────────────────────────────────────────────────────────
# 3. ALGORITHM 1 — Random Forest (RF)
# ─────────────────────────────────────────────────────────────
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_arr, y_arr)

rf_imp   = pd.Series(rf_model.feature_importances_, index=feature_names)
rf_top8  = rf_imp.nlargest(8).index.tolist()
rf_cv8   = cv_rf(rf_top8, X, y_arr, cv)

# remove highest → reduced → top 7
rf_highest    = rf_imp.idxmax()
X_rf_red      = X.drop(columns=[rf_highest])
rf_model_red  = RandomForestClassifier(random_state=42)
rf_model_red.fit(X_rf_red.values, y_arr)
rf_imp_red    = pd.Series(rf_model_red.feature_importances_,
                          index=X_rf_red.columns.tolist())
rf_top7       = rf_imp_red.nlargest(7).index.tolist()

print(f"[RF]       highest={rf_highest}")
print(f"[RF]       top8={rf_top8}")
print(f"[RF]       top7={rf_top7}  CV8={rf_cv8}\n")

# ─────────────────────────────────────────────────────────────
# 4. ALGORITHM 2 — XGBoost (XGB)
# ─────────────────────────────────────────────────────────────
xgb_model = XGBClassifier(random_state=42,
                           eval_metric='logloss',
                           use_label_encoder=False)
xgb_model.fit(X_arr, y_xgb)

xgb_imp   = pd.Series(xgb_model.feature_importances_, index=feature_names)
xgb_top8  = xgb_imp.nlargest(8).index.tolist()
xgb_cv8   = cv_xgb(xgb_top8, X, y_xgb, cv)

# remove highest → reduced → top 7
xgb_highest   = xgb_imp.idxmax()
X_xgb_red     = X.drop(columns=[xgb_highest])
xgb_model_red = XGBClassifier(random_state=42,
                               eval_metric='logloss',
                               use_label_encoder=False)
xgb_model_red.fit(X_xgb_red.values, y_xgb)
xgb_imp_red   = pd.Series(xgb_model_red.feature_importances_,
                           index=X_xgb_red.columns.tolist())
xgb_top7      = xgb_imp_red.nlargest(7).index.tolist()

print(f"[XGB]      highest={xgb_highest}")
print(f"[XGB]      top8={xgb_top8}")
print(f"[XGB]      top7={xgb_top7}  CV8={xgb_cv8}\n")

# ─────────────────────────────────────────────────────────────
# 5. ALGORITHM 3 — RF-SHAP
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
idx100   = np.random.choice(len(X_arr),
                            size=min(100, len(X_arr)),
                            replace=False)
X_s100   = X_arr[idx100]

rfshap_model = RandomForestClassifier(random_state=42)
rfshap_model.fit(X_arr, y_arr)

explainer_rf  = shap.TreeExplainer(rfshap_model)
shap_rf       = explainer_rf.shap_values(X_s100)
mean_rf       = extract_shap_mean(shap_rf, len(feature_names))

rfshap_imp    = pd.Series(mean_rf, index=feature_names)
rfshap_top8   = rfshap_imp.nlargest(8).index.tolist()
rfshap_cv8    = cv_rf(rfshap_top8, X, y_arr, cv)

# remove highest → reduced → top 7
rfshap_highest = rfshap_imp.idxmax()
X_rfshap_red   = X.drop(columns=[rfshap_highest])

np.random.seed(42)
idx100_r2      = np.random.choice(len(X_arr),
                                  size=min(100, len(X_arr)),
                                  replace=False)
rfshap_model2  = RandomForestClassifier(random_state=42)
rfshap_model2.fit(X_rfshap_red.values, y_arr)
explainer_rf2  = shap.TreeExplainer(rfshap_model2)
shap_rf2       = explainer_rf2.shap_values(X_rfshap_red.values[idx100_r2])
mean_rf2       = extract_shap_mean(shap_rf2, X_rfshap_red.shape[1])

rfshap_imp2    = pd.Series(mean_rf2, index=X_rfshap_red.columns.tolist())
rfshap_top7    = rfshap_imp2.nlargest(7).index.tolist()

print(f"[RF-SHAP]  highest={rfshap_highest}")
print(f"[RF-SHAP]  top8={rfshap_top8}")
print(f"[RF-SHAP]  top7={rfshap_top7}  CV8={rfshap_cv8}\n")

# ─────────────────────────────────────────────────────────────
# 6. ALGORITHM 4 — XGB-SHAP
# ─────────────────────────────────────────────────────────────
xgbshap_model = XGBClassifier(random_state=42,
                               eval_metric='logloss',
                               use_label_encoder=False)
xgbshap_model.fit(X_arr, y_xgb)

explainer_xgb  = shap.TreeExplainer(xgbshap_model)
shap_xgb       = explainer_xgb.shap_values(X_s100)
mean_xgb       = extract_shap_mean(shap_xgb, len(feature_names))

xgbshap_imp    = pd.Series(mean_xgb, index=feature_names)
xgbshap_top8   = xgbshap_imp.nlargest(8).index.tolist()
xgbshap_cv8    = cv_xgb(xgbshap_top8, X, y_xgb, cv)

# remove highest → reduced → top 7
xgbshap_highest = xgbshap_imp.idxmax()
X_xgbshap_red   = X.drop(columns=[xgbshap_highest])

xgbshap_model2  = XGBClassifier(random_state=42,
                                 eval_metric='logloss',
                                 use_label_encoder=False)
xgbshap_model2.fit(X_xgbshap_red.values, y_xgb)
explainer_xgb2  = shap.TreeExplainer(xgbshap_model2)
shap_xgb2       = explainer_xgb2.shap_values(
                      X_xgbshap_red.values[idx100_r2])
mean_xgb2       = extract_shap_mean(shap_xgb2, X_xgbshap_red.shape[1])

xgbshap_imp2    = pd.Series(mean_xgb2, index=X_xgbshap_red.columns.tolist())
xgbshap_top7    = xgbshap_imp2.nlargest(7).index.tolist()

print(f"[XGB-SHAP] highest={xgbshap_highest}")
print(f"[XGB-SHAP] top8={xgbshap_top8}")
print(f"[XGB-SHAP] top7={xgbshap_top7}  CV8={xgbshap_cv8}\n")

# ─────────────────────────────────────────────────────────────
# 7. ALGORITHM 5 — Feature Agglomeration (FA)
#    rank ALL features by variance across clusters → top 8
# ─────────────────────────────────────────────────────────────
fa = FeatureAgglomeration(n_clusters=8)
fa.fit(X_arr)

# Independent proxy: variance of each original feature
fa_var  = X.var(axis=0)
fa_imp  = pd.Series(fa_var.values, index=feature_names)
fa_top8 = fa_imp.nlargest(8).index.tolist()
fa_cv8  = cv_rf(fa_top8, X, y_arr, cv)

# remove highest → reduced → top 7
fa_highest  = fa_imp.idxmax()
X_fa_red    = X.drop(columns=[fa_highest])
fa_imp_red  = X_fa_red.var(axis=0)
fa_top7     = fa_imp_red.nlargest(7).index.tolist()

print(f"[FA]       highest={fa_highest}")
print(f"[FA]       top8={fa_top8}")
print(f"[FA]       top7={fa_top7}  CV8={fa_cv8}\n")

# ─────────────────────────────────────────────────────────────
# 8. ALGORITHM 6 — Highly Variable Gene Selection (HVGS)
#    pure variance-based, no model
# ─────────────────────────────────────────────────────────────
hvgs_var     = X.var(axis=0)
hvgs_imp     = pd.Series(hvgs_var.values, index=feature_names)
hvgs_top8    = hvgs_imp.nlargest(8).index.tolist()
hvgs_cv8     = cv_rf(hvgs_top8, X, y_arr, cv)

# remove highest → reduced → top 7
hvgs_highest = hvgs_imp.idxmax()
X_hvgs_red   = X.drop(columns=[hvgs_highest])
hvgs_imp_red = X_hvgs_red.var(axis=0)
hvgs_top7    = hvgs_imp_red.nlargest(7).index.tolist()

print(f"[HVGS]     highest={hvgs_highest}")
print(f"[HVGS]     top8={hvgs_top8}")
print(f"[HVGS]     top7={hvgs_top7}  CV8={hvgs_cv8}\n")

# ─────────────────────────────────────────────────────────────
# 9. ALGORITHM 7 — Spearman Correlation
#    |spearmanr| with target, RF cross-validation
# ─────────────────────────────────────────────────────────────
spear_scores = {}
for feat in feature_names:
    corr, _ = spearmanr(X[feat].values, y_arr)
    spear_scores[feat] = abs(corr)

spear_imp     = pd.Series(spear_scores)
spear_top8    = spear_imp.nlargest(8).index.tolist()
spear_cv8     = cv_rf(spear_top8, X, y_arr, cv)

# remove highest → reduced → top 7
spear_highest = spear_imp.idxmax()
X_spear_red   = X.drop(columns=[spear_highest])

spear_scores2 = {}
for feat in X_spear_red.columns:
    corr, _ = spearmanr(X_spear_red[feat].values, y_arr)
    spear_scores2[feat] = abs(corr)

spear_imp2    = pd.Series(spear_scores2)
spear_top7    = spear_imp2.nlargest(7).index.tolist()

print(f"[Spearman] highest={spear_highest}")
print(f"[Spearman] top8={spear_top8}")
print(f"[Spearman] top7={spear_top7}  CV8={spear_cv8}\n")

# ─────────────────────────────────────────────────────────────
# 10. SUMMARY TABLE  (4 columns)
# ─────────────────────────────────────────────────────────────
results = [
    {'Methods': 'RF',
     'CV8': rf_cv8,
     'Top8_Features': ', '.join(rf_top8),
     'Top7_Features': ', '.join(rf_top7)},

    {'Methods': 'XGB',
     'CV8': xgb_cv8,
     'Top8_Features': ', '.join(xgb_top8),
     'Top7_Features': ', '.join(xgb_top7)},

    {'Methods': 'RF-SHAP',
     'CV8': rfshap_cv8,
     'Top8_Features': ', '.join(rfshap_top8),
     'Top7_Features': ', '.join(rfshap_top7)},

    {'Methods': 'XGB-SHAP',
     'CV8': xgbshap_cv8,
     'Top8_Features': ', '.join(xgbshap_top8),
     'Top7_Features': ', '.join(xgbshap_top7)},

    {'Methods': 'FA',
     'CV8': fa_cv8,
     'Top8_Features': ', '.join(fa_top8),
     'Top7_Features': ', '.join(fa_top7)},

    {'Methods': 'HVGS',
     'CV8': hvgs_cv8,
     'Top8_Features': ', '.join(hvgs_top8),
     'Top7_Features': ', '.join(hvgs_top7)},

    {'Methods': 'Spearman',
     'CV8': spear_cv8,
     'Top8_Features': ', '.join(spear_top8),
     'Top7_Features': ', '.join(spear_top7)},
]

summary_df = pd.DataFrame(results,
                           columns=['Methods', 'CV8',
                                    'Top8_Features', 'Top7_Features'])

print("========== SUMMARY TABLE ==========")
print(summary_df.to_string(index=False))

summary_df.to_csv('result2.csv', index=False)
print("\nSaved → result2.csv")