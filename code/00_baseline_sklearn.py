"""
Reproduce reference paper baseline: Extra Trees + Gradient Boosting + XGBoost
on 80:20 random split with same seed. Uses Morgan fingerprints (fast alternative
to Mordred; reported for comparison purposes only).
"""

import os, sys, pickle, json, warnings
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.dirname(CODE_DIR)
RESULT_DIR = os.path.join(DATA_DIR, "results")


def main():
    with open(os.path.join(RESULT_DIR, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)

    train_fps = data['random_split']['train_fps']
    test_fps  = data['random_split']['test_fps']
    train_idx = data['random_split']['train_idx']
    test_idx  = data['random_split']['test_idx']
    pIC50     = data['df']['pIC50'].values

    y_train = pIC50[train_idx]
    y_test  = pIC50[test_idx]

    print(f"Baseline sklearn models on Morgan fingerprints (radius=2, 2048 bits)")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}\n")

    models = {
        'Extra Trees':      ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'Random Forest':    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }

    results = []
    print(f"{'Model':<22} {'CV R² (10f)':>12} {'Test R²':>9} {'RMSE':>8} {'MAE':>8}")
    print("-"*65)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for name, clf in models.items():
        # 10-fold CV
        cv_scores = cross_val_score(clf, train_fps, y_train, cv=kf,
                                    scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std  = cv_scores.std(ddof=1)

        # External test
        clf.fit(train_fps, y_train)
        y_pred = clf.predict(test_fps)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)

        print(f"{name:<22} {cv_mean:.4f}±{cv_std:.4f} {r2:>9.4f} {rmse:>8.4f} {mae:>8.4f}")
        results.append({'model': name, 'cv_r2_mean': float(cv_mean),
                        'cv_r2_std': float(cv_std),
                        'test_r2': float(r2), 'test_rmse': float(rmse),
                        'test_mae': float(mae),
                        'predictions': y_pred.tolist(), 'targets': y_test.tolist()})

    print("\nNote: reference paper uses Mordred descriptors (>1800) with feature selection.")
    print("These Morgan fingerprint results are for code validation only.\n")

    out_path = os.path.join(RESULT_DIR, "baseline_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == '__main__':
    main()
