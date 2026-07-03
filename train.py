import pandas as pd
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("C:/Users/jenifer/Downloads/model/crop-datasets (1).csv")   
df.columns = df.columns.str.strip().str.lower()
df = df.dropna(
    subset=[
        "crop",
        "season",
        "state"
    ]
)
for col in ["crop", "season", "state"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )
df = df[
    (df["area"] > 0)
    & (df["production"] > 0)
    ]
global_prod_mean = df["production"].mean()
global_area_mean = df["area"].mean()

valid_values = {
    "crops": sorted(
        df["crop"].dropna().astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    ),
    "states": sorted(
        df["state"].dropna().astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    ),
    "seasons": sorted(
        df["season"].dropna().astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
}


baseline_mean = (
    df.groupby(
        ["crop", "season", "state"]
    )[["production", "area"]]
    .mean()
)
baseline_median = (
    df.groupby(
        ["crop", "season", "state"]
    )[["production", "area"]]
    .median()
)
def mean_score(row):
    try:
        baseline = baseline_mean.loc[
            (
                row["crop"],
                row["season"],
                row["state"]
            )
        ]
    except KeyError:
        return 0.0

    baseline_production = float(baseline["production"])
    baseline_area = float(baseline["area"])

    if baseline_production <= 0 or baseline_area <= 0:
        return 0.0

    production_score = min(
        (row["production"] / baseline_production) * 80,
        80
    )

    area_score = min(
        (row["area"] / baseline_area) * 20,
        20
    )

    return production_score + area_score


def median_score(row):
    try:
        baseline = baseline_median.loc[
            (
                row["crop"],
                row["season"],
                row["state"]
            )
        ]
    except KeyError:
        return 0.0

    baseline_production = float(baseline["production"])
    baseline_area = float(baseline["area"])

    if baseline_production <= 0 or baseline_area <= 0:
        return 0.0

    production_score = min(
        (row["production"] / baseline_production) * 80,
        80
    )

    area_score = min(
        (row["area"] / baseline_area) * 20,
        20
    )

    return production_score + area_score


def calculate_score(row, baseline_table):
    try:
        baseline = baseline_table.loc[
            (
                row["crop"],
                row["season"],
                row["state"]
            )
        ]
        baseline_production = float(baseline["production"])
        baseline_area = float(baseline["area"])
    except (KeyError, TypeError):
        baseline_production = global_prod_mean
        baseline_area = global_area_mean

    if baseline_production <= 0 or baseline_area <= 0:
        return 0.0

    productivity = row["production"] / row["area"]
    baseline_productivity = baseline_production / baseline_area
    productivity_ratio = productivity / baseline_productivity

    production_ratio = row["production"] / baseline_production
    area_ratio = row["area"] / baseline_area

    production_ratio = min(production_ratio, 5)
    area_ratio = min(area_ratio, 5)
    productivity_ratio = min(productivity_ratio, 5)

    score = (
        50 * np.log1p(productivity_ratio)
        + 30 * np.log1p(production_ratio)
        + 20 * np.log1p(area_ratio)
    )

    score = min(score, 100)
    return float(score)

df["cropping_pattern_score"] = df.apply(
    lambda row: calculate_score(
        row,
        baseline_mean
    ),
    axis=1
)

df["score_mean"] = df.apply(
    mean_score,
    axis=1
)

df["score_median"] = df.apply(
    median_score,
    axis=1
)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

global_prod_mean = train_df["production"].mean()
global_area_mean = train_df["area"].mean()


df["score_mean"] = df.apply(mean_score, axis=1)
df["score_median"] = df.apply(median_score, axis=1)

def add_ratio_features(data):

    out = data.merge(
        baseline_table,
        on=["crop", "season", "state"],
        how="left"
    )

    out["baseline_production"] = (
        out["baseline_production"]
        .fillna(global_prod_mean)
    )

    out["baseline_area"] = (
        out["baseline_area"]
        .fillna(global_area_mean)
    )

    out["production_ratio"] = (
        out["production"]
        /
        out["baseline_production"]
    ).clip(0, 5)

    out["area_ratio"] = (
        out["area"]
        /
        out["baseline_area"]
    ).clip(0, 5)

    out["productivity_ratio"] = (
        (out["production"] / out["area"])
        /
        (
            out["baseline_production"]
            /
            out["baseline_area"]
        )
    ).clip(0, 5)

    return out
baseline_table = (
    train_df.groupby(
        ["crop", "season", "state"]
    )[["production", "area"]]
    .mean()
    .rename(
        columns={
            "production": "baseline_production",
            "area": "baseline_area"
        }
    )
)
train_df = add_ratio_features(train_df)
test_df = add_ratio_features(test_df)

features = [
    "crop",
    "season",
    "state",
    "area",
    "production",
    "crop_year"
]
X_train = train_df[features]
X_test = test_df[features]

y_train = train_df["cropping_pattern_score"]
y_test = test_df["cropping_pattern_score"]

X_train_mean = train_df[features]
X_test_mean = test_df[features]

y_train_mean = train_df["score_mean"]
y_test_mean = test_df["score_mean"]

X_train_median = train_df[features]
X_test_median = test_df[features]

y_train_median = train_df["score_median"]
y_test_median = test_df["score_median"]

categorical_features = [
    "crop",
    "season",
    "state"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            [
                "area",
                "production",
                "crop_year"
            ]
        )
    ]
)

xgb_params = {}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        **xgb_params,
        n_estimators=500,
        learning_rate=0.02,
        max_depth=10,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2,
        random_state=42,
        tree_method="hist",
        objective="reg:squarederror"
    )),
])

mean_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        **xgb_params,
        n_estimators=500,
        learning_rate=0.02,
        max_depth=10,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2,
        random_state=42,
        tree_method="hist",
        objective="reg:squarederror"
    )),
])
median_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        **xgb_params,
        n_estimators=500,
        learning_rate=0.02,
        max_depth=10,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2,
        random_state=42,
        tree_method="hist",
        objective="reg:squarederror"
    )),
])

pipeline.fit(X_train, y_train)
mean_pipeline.fit(
    X_train_mean,
    y_train_mean
)

median_pipeline.fit(
    X_train_median,
    y_train_median
)

pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE:", rmse)
print(
    f"MAE: {mean_absolute_error(y_test, pred):.2f}"
)

print(
    f"R² Score: {r2_score(y_test, pred):.3f}"
)
feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": pipeline.named_steps["model"].feature_importances_
})

pred = pipeline.predict(X_test)

pred_mean = mean_pipeline.predict(
    X_test_mean
)

pred_median = median_pipeline.predict(
    X_test_median
)

model_package = {
    "pipeline": pipeline,
    "mean_model": mean_pipeline,
    "median_model": median_pipeline,
    "valid_values": valid_values
}

joblib.dump(
    model_package,
    "crop_pattern_model_XGB.joblib"
)
print(" Model saved successfully!")