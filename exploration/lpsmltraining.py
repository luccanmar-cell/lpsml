import pandas as pd
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lpsmltest import computemetrics

DATASET_PATH = "tarifacompleto.parquet"
TARGET_COLUMN = "Prima"

df = pd.read_parquet(DATASET_PATH)
df = df.apply(pd.to_numeric, errors="coerce")

X = df.drop(columns=[TARGET_COLUMN, "NroPoliza"], errors="ignore")
Y = df[TARGET_COLUMN]

cv = KFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_result = None

parameter_grid = [
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 300, "max_depth": None},
    {"n_estimators": 200, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
]

for params in parameter_grid:
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
        ),
    )

    scores = cross_val_score(
        pipeline,
        X,
        Y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    mean_mae = -scores.mean()
    print(
        f"n_estimators={params['n_estimators']}, max_depth={params['max_depth']} -> "
        f"CV MAE: {mean_mae:.4f}"
    )

    if best_result is None or mean_mae < best_result:
        best_result = mean_mae
        best_model = pipeline

print(f"Best cross-validated MAE: {best_result:.4f}")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
best_model.fit(X_train, Y_train)
computemetrics(best_model, X_test, Y_test, X_train)
