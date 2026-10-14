import numpy as np
import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

data_path = "car_web_scraped_dataset.csv"
artifact_dir = "artifacts"
current_year = 2026

two_word_brand = ["Land Rover", "Alfa Romeo", "Aston Martin"]
def extract_brand_name(name):
    for b in two_word_brand:
        if name.startswith(b):
            return b
    return name.split()[0]

def clean_row_datafram(data):
    df = data.copy()
    df["miles"] = (df["miles"].astype(str).str.replace(",", "", regex=False).str.replace("miles", "", regex=False).str.strip())
    df["miles"] = pd.to_numeric(df["miles"], errors="coerce")

    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    color_split = df["color"].astype(str).str.split(",", expand=True)
    df["exterior_color"] = color_split[0].str.replace("exterior", "", regex=False).str.strip()
    if 1 in color_split.columns:
        df["interior_color"] = color_split[1].str.replace("interior", "", regex=False).str.strip()
    else:
        df["interior_color"] = "Unknown"
    df["exterior_color"] = df["exterior_color"].fillna("Unknown")
    df["interior_color"] = df["interior_color"].fillna("Unknown")

    df["accidents"] = df["condition"].astype(str).str.extract(r"(\d+|No) accident")[0]
    df["accidents"] = df["accidents"].replace("No", "0")
    df["accidents"] = pd.to_numeric(df["accidents"], errors="coerce").fillna(0).astype(int)
 
    df["owners"] = df["condition"].astype(str).str.extract(r"(\d+) Owner")[0]
    df["owners"] = pd.to_numeric(df["owners"], errors="coerce")

    df["car_age"] = current_year - df["year"]
    df["miles_per_year"] = df["miles"] / df["car_age"].replace(0, 1)

    df["brand"] = df["name"].astype(str).apply(extract_brand_name)
    df["model"] = df.apply(lambda r: r["name"][len(r["brand"]):].strip(), axis=1)
 
    return df

def main():
    print("Loading Data")
    raw = pd.read_csv(data_path)
    print("Cleaning Data")
    df = clean_row_datafram(raw)

    df = df.dropna(subset=["price", "miles"]).copy()
    df["owners"] = df["owners"].fillna(df["owners"].median())

    q1, q3 = df["price"].quantile([0.01, 0.99])
    before = len(df)
    df = df[(df["price"] >= q1) & (df["price"] <= q3)].copy()
    print(f"   Dropped {before - len(df)} extreme price outlier rows (top/bottom 1%)")

    model_count = df["model"].value_counts()
    brand_count = df["brand"].value_counts()
    rare_model = model_count[model_count < 5].index
    df["model_grp"] = df["model"].replace(rare_model, "Other")
    rare_brand = brand_count[brand_count < 15].index
    df["brand"] = df["brand"].replace(rare_brand, "Other")

    df.to_parquet(f"{artifact_dir}/clean_data.parquet", index=False)

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    train_data = train_data.copy()
    test_data = test_data.copy()

    train_data["price_log"] = np.log1p(train_data["price"])
    test_data["price_log"] = np.log1p(test_data["price"])

    overall_avg = train_data["price_log"].mean()

    brand_avg_price = train_data.groupby("brand")["price_log"].mean()
    train_data["brand_encoded"] = train_data["brand"].map(brand_avg_price).fillna(overall_avg)
    test_data["brand_encoded"] = test_data["brand"].map(brand_avg_price).fillna(overall_avg)
 
    model_avg_price = train_data.groupby("model")["price_log"].mean()
    train_data["model_encoder"] = train_data["model"].map(model_avg_price).fillna(overall_avg)
    test_data["model_encoder"] = test_data["model"].map(model_avg_price).fillna(overall_avg)
 
    ext_avg_price = train_data.groupby("exterior_color")["price_log"].mean()
    train_data["exterior_encoder_color"] = train_data["exterior_color"].map(ext_avg_price).fillna(overall_avg)
    test_data["exterior_encoder_color"] = test_data["exterior_color"].map(ext_avg_price).fillna(overall_avg)
 
    int_avg_price = train_data.groupby("interior_color")["price_log"].mean()
    train_data["interior_encoder_color"] = train_data["interior_color"].map(int_avg_price).fillna(overall_avg)
    test_data["interior_encoder_color"] = test_data["interior_color"].map(int_avg_price).fillna(overall_avg)

    feature_column = [
        "miles",
        "car_age",
        "miles_per_year",
        "accidents",
        "owners",
        "brand_encoded",
        "model_encoder",
        "exterior_encoder_color",
        "interior_encoder_color",
    ]

    x_train = train_data[feature_column]
    x_test = test_data[feature_column]
    y_train = train_data["price_log"]
    y_test = test_data["price_log"]
    y_test_actual_price = np.expm1(y_test)
 
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print("Training all Models")
    models = {
        "Linear Regression": (LinearRegression(), True),
        "KNN Regressor": (KNeighborsRegressor(n_neighbors=7), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=8, random_state=42), False),
        "Random Forest": (RandomForestRegressor(n_estimators=400, random_state=42), False),
        "XG Boost": (
            XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42),
            False,
        ),
    }

    results = {}
    trained_models = {}
 
    for model_name, (model, needs_scaling) in models.items():
        if needs_scaling:
            model.fit(x_train_scaled, y_train)
            predicted_log_price = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            predicted_log_price = model.predict(x_test)
 
        predicted_price = np.expm1(predicted_log_price)
 
        mae = mean_absolute_error(y_test_actual_price, predicted_price)
        rmse = np.sqrt(mean_squared_error(y_test_actual_price, predicted_price))
        r2 = r2_score(y_test_actual_price, predicted_price)
 
        results[model_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "needs_scaling": needs_scaling,
        }
        trained_models[model_name] = model
        print(f"   {model_name:20s} MAE: {mae:>8,.0f}  RMSE: {rmse:>8,.0f}  R2: {r2:.3f}")

    results_table = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    best_model_name = results_table.index[0]
    print(f"\nBest Model: {best_model_name} (R2={results_table.loc[best_model_name, 'R2']:.3f})")

    print("Saving artifacts")
    joblib.dump(trained_models, f"{artifact_dir}/trained_models.joblib")
    joblib.dump(scaler, f"{artifact_dir}/scaler.joblib")

    encoders = {
        "overall_avg": overall_avg,
        "brand_avg_price": brand_avg_price,
        "model_avg_price": model_avg_price,
        "exterior_avg_price": ext_avg_price,
        "interior_avg_price": int_avg_price,
        "rare_brand": list(rare_brand),
        "rare_model": list(rare_model),
        "owners_median": float(df["owners"].median()),
    }
    joblib.dump(encoders, f"{artifact_dir}/encoders.joblib")

    meta = {
        "feature_column": feature_column,
        "best_model_name": best_model_name,
        "current_year": current_year,
        "n_rows_raw": int(len(raw)),
        "n_rows_clean": int(len(df)),
        "brands": sorted(df["brand"].unique().tolist()),
        "models_by_brand": {
            b: sorted(df.loc[df["brand"] == b, "model"].unique().tolist())
            for b in df["brand"].unique()
        },
        "exterior_colors": sorted(df["exterior_color"].dropna().unique().tolist()),
        "interior_colors": sorted(df["interior_color"].dropna().unique().tolist()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "miles_median": float(df["miles"].median()),
    }
    with open(f"{artifact_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
 
    results_table.to_json(f"{artifact_dir}/results_table.json", orient="index")
 
    print("\nDone! All artifacts saved in ./artifacts/")
 
 
if __name__ == "__main__":
    main()
