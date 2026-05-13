import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import spearmanr
import shap
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. Load data
# ─────────────────────────────────────────────
df = pd.read_csv('Fasting_NonFasting_9Subjects.csv')
df = df.drop(columns=['Id'])

target = 'BGAfter'
X = df.drop(columns=[target])
y = df[target]

feature_names = list(X.columns)
N_SELECT     = 8
N_REDUCED    = 7
CV_FOLDS     = 5
SHAP_SAMPLE  = 100
RANDOM_STATE = 42

def cv_r2(model, features, X, y, cv=CV_FOLDS):
    scores = cross_val_score(model, X[features], y,
                             cv=cv, scoring='r2')
    return round(scores.mean(), 4)

results = {}   # method -> {'cv8': float, 'top8': list, 'top7': list}

# ─────────────────────────────────────────────
# 1. Random Forest (RF)
# ─────────────────────────────────────────────
rf = RandomForestRegressor(random_state=RANDOM_STATE)
rf.fit(X, y)
rf_imp = pd.Series(rf.feature_importances_,
                   index=feature_names).sort_values(ascending=False)
top8_rf = list(rf_imp.index[:N_SELECT])
cv8_rf  = cv_r2(RandomForestRegressor(random_state=RANDOM_STATE), top8_rf, X, y)

highest_rf = top8_rf[0]
X_red_rf   = X.drop(columns=[highest_rf])
rf2 = RandomForestRegressor(random_state=RANDOM_STATE)
rf2.fit(X_red_rf, y)
rf_imp2 = pd.Series(rf2.feature_importances_,
                    index=X_red_rf.columns).sort_values(ascending=False)
top7_rf = list(rf_imp2.index[:N_REDUCED])

results['RF'] = {'cv8': cv8_rf, 'top8': top8_rf, 'top7': top7_rf}

# ─────────────────────────────────────────────
# 2. XGBoost (XGB)
# ─────────────────────────────────────────────
xgb = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
xgb.fit(X, y)
xgb_imp = pd.Series(xgb.feature_importances_,
                    index=feature_names).sort_values(ascending=False)
top8_xgb = list(xgb_imp.index[:N_SELECT])
cv8_xgb  = cv_r2(XGBRegressor(random_state=RANDOM_STATE, verbosity=0), top8_xgb, X, y)

highest_xgb = top8_xgb[0]
X_red_xgb   = X.drop(columns=[highest_xgb])
xgb2 = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
xgb2.fit(X_red_xgb, y)
xgb_imp2 = pd.Series(xgb2.feature_importances_,
                     index=X_red_xgb.columns).sort_values(ascending=False)
top7_xgb = list(xgb_imp2.index[:N_REDUCED])

results['XGB'] = {'cv8': cv8_xgb, 'top8': top8_xgb, 'top7': top7_xgb}

# ─────────────────────────────────────────────
# 3. RF-SHAP  (100 randomly sampled instances)
# ─────────────────────────────────────────────
# --- full dataset ---
rng_rf = np.random.RandomState(RANDOM_STATE)
sample_idx_rf = rng_rf.choice(len(X), size=min(SHAP_SAMPLE, len(X)), replace=False)
X_shap_rf = X.iloc[sample_idx_rf]

rf_shap = RandomForestRegressor(random_state=RANDOM_STATE)
rf_shap.fit(X, y)
explainer_rf = shap.TreeExplainer(rf_shap)
shap_vals_rf = explainer_rf.shap_values(X_shap_rf)
rf_shap_imp  = pd.Series(np.abs(shap_vals_rf).mean(axis=0),
                          index=feature_names).sort_values(ascending=False)
top8_rfshap = list(rf_shap_imp.index[:N_SELECT])
cv8_rfshap  = cv_r2(RandomForestRegressor(random_state=RANDOM_STATE), top8_rfshap, X, y)

# --- reduced dataset ---
highest_rfshap = top8_rfshap[0]
X_red_rfshap   = X.drop(columns=[highest_rfshap])

sample_idx_rf2 = rng_rf.choice(len(X_red_rfshap),
                                size=min(SHAP_SAMPLE, len(X_red_rfshap)), replace=False)
X_shap_rf2     = X_red_rfshap.iloc[sample_idx_rf2]

rf_shap2 = RandomForestRegressor(random_state=RANDOM_STATE)
rf_shap2.fit(X_red_rfshap, y)
expl_rf2     = shap.TreeExplainer(rf_shap2)
sv_rf2       = expl_rf2.shap_values(X_shap_rf2)
rfshap_imp2  = pd.Series(np.abs(sv_rf2).mean(axis=0),
                          index=X_red_rfshap.columns).sort_values(ascending=False)
top7_rfshap = list(rfshap_imp2.index[:N_REDUCED])

results['RF-SHAP'] = {'cv8': cv8_rfshap, 'top8': top8_rfshap, 'top7': top7_rfshap}

# ─────────────────────────────────────────────
# 4. XGB-SHAP  (100 randomly sampled instances)
# ─────────────────────────────────────────────
# --- full dataset ---
rng_xgb = np.random.RandomState(RANDOM_STATE)
sample_idx_xgb = rng_xgb.choice(len(X), size=min(SHAP_SAMPLE, len(X)), replace=False)
X_shap_xgb = X.iloc[sample_idx_xgb]

xgb_shap = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
xgb_shap.fit(X, y)
explainer_xgb = shap.TreeExplainer(xgb_shap)
shap_vals_xgb = explainer_xgb.shap_values(X_shap_xgb)
xgb_shap_imp  = pd.Series(np.abs(shap_vals_xgb).mean(axis=0),
                            index=feature_names).sort_values(ascending=False)
top8_xgbshap = list(xgb_shap_imp.index[:N_SELECT])
cv8_xgbshap  = cv_r2(XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
                     top8_xgbshap, X, y)

# --- reduced dataset ---
highest_xgbshap = top8_xgbshap[0]
X_red_xgbshap   = X.drop(columns=[highest_xgbshap])

sample_idx_xgb2 = rng_xgb.choice(len(X_red_xgbshap),
                                   size=min(SHAP_SAMPLE, len(X_red_xgbshap)), replace=False)
X_shap_xgb2     = X_red_xgbshap.iloc[sample_idx_xgb2]

xgb_shap2 = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
xgb_shap2.fit(X_red_xgbshap, y)
expl_xgb2    = shap.TreeExplainer(xgb_shap2)
sv_xgb2      = expl_xgb2.shap_values(X_shap_xgb2)
xgbshap_imp2 = pd.Series(np.abs(sv_xgb2).mean(axis=0),
                          index=X_red_xgbshap.columns).sort_values(ascending=False)
top7_xgbshap = list(xgbshap_imp2.index[:N_REDUCED])

results['XGB-SHAP'] = {'cv8': cv8_xgbshap, 'top8': top8_xgbshap, 'top7': top7_xgbshap}

# ─────────────────────────────────────────────
# 5. Feature Agglomeration (FA) — fully standalone
#
#    FeatureAgglomeration groups features into k clusters
#    using hierarchical clustering (default settings).
#    Each individual original feature's own variance is
#    computed directly from its raw values (no mean pooling,
#    no transformation, no external metric).
#    Features are ranked globally by their own variance
#    across all clusters; top-N are selected.
# ─────────────────────────────────────────────
def fa_top_features(X_fa, n_select):
    n_feat = X_fa.shape[1]
    k = min(n_select, n_feat)
    feat_arr = np.array(X_fa.columns)

    # Fit FeatureAgglomeration — default settings, only n_clusters set
    fa = FeatureAgglomeration(n_clusters=k)
    fa.fit(X_fa)
    # fa.labels_: cluster assignment for each original feature

    # Compute variance of each individual original feature (no mean/transform)
    feature_variances = X_fa.var(axis=0)   # pandas Series, index = feature names

    # Rank ALL original features globally by their own variance (descending)
    # across all clusters — no per-cluster restriction
    ranked = feature_variances.sort_values(ascending=False)

    # Return top n_select feature names from the global ranking
    return list(ranked.index[:n_select])

# --- full dataset ---
top8_fa = fa_top_features(X, N_SELECT)
cv8_fa  = cv_r2(RandomForestRegressor(random_state=RANDOM_STATE), top8_fa, X, y)

# --- reduced dataset ---
highest_fa = top8_fa[0]
X_red_fa   = X.drop(columns=[highest_fa])
top7_fa    = fa_top_features(X_red_fa, N_REDUCED)

results['FA'] = {'cv8': cv8_fa, 'top8': top8_fa, 'top7': top7_fa}

# ─────────────────────────────────────────────
# 6. Highly Variable Gene Selection (HVGS)
#    Rank by variance; use RF for CV
# ─────────────────────────────────────────────
var_series = X.var().sort_values(ascending=False)
top8_hvgs  = list(var_series.index[:N_SELECT])
cv8_hvgs   = cv_r2(RandomForestRegressor(random_state=RANDOM_STATE), top8_hvgs, X, y)

highest_hvgs = top8_hvgs[0]
X_red_hvgs   = X.drop(columns=[highest_hvgs])
top7_hvgs    = list(X_red_hvgs.var().sort_values(ascending=False).index[:N_REDUCED])

results['HVGS'] = {'cv8': cv8_hvgs, 'top8': top8_hvgs, 'top7': top7_hvgs}

# ─────────────────────────────────────────────
# 7. Spearman correlation
#    Rank by |spearmanr|; use RF for CV
# ─────────────────────────────────────────────
spearman_scores = {}
for f in feature_names:
    r, _ = spearmanr(X[f], y)
    spearman_scores[f] = abs(r)
spearman_series = pd.Series(spearman_scores).sort_values(ascending=False)
top8_spearman   = list(spearman_series.index[:N_SELECT])
cv8_spearman    = cv_r2(RandomForestRegressor(random_state=RANDOM_STATE),
                        top8_spearman, X, y)

highest_spearman = top8_spearman[0]
X_red_spearman   = X.drop(columns=[highest_spearman])
sp2 = {}
for f in X_red_spearman.columns:
    r, _ = spearmanr(X_red_spearman[f], y)
    sp2[f] = abs(r)
top7_spearman = list(pd.Series(sp2).sort_values(ascending=False).index[:N_REDUCED])

results['Spearman'] = {'cv8': cv8_spearman, 'top8': top8_spearman, 'top7': top7_spearman}

# ─────────────────────────────────────────────
# 8. Build summary table and save
# ─────────────────────────────────────────────
rows = []
for method, info in results.items():
    rows.append({
        'Methods'       : method,
        'CV8'           : info['cv8'],
        'Top8_Features' : ', '.join(info['top8']),
        'Top7_Features' : ', '.join(info['top7'])
    })

summary = pd.DataFrame(rows, columns=['Methods', 'CV8', 'Top8_Features', 'Top7_Features'])
summary.to_csv('result.csv', index=False)

print(summary.to_string(index=False))