
# VahanValue — Used Car Price Prediction
 
A Streamlit web app that estimates the market price of a used car from its brand, model, year, mileage, colours, accident history and number of owners. It supports valuing a single car or thousands at once by uploading a CSV, XLSX or JSON file, and shows analytics charts for every batch.
 
Five regression models were trained and compared on the same test set. The best one, **XGBoost**, reaches an R² of about **0.80**.
 
---


## Features
 
- **Single car prediction**: fill in a form and get an instant valuation.
- **Bulk prediction**: upload `.csv`, `.xlsx` or `.json`, value every row at once, and download the results as CSV, XLSX or JSON.
- **Batch analytics**: after each bulk run the app shows KPI cards (cars valued, average, highest, total portfolio value) and four charts (price distribution, top valued cars, miles vs price, average price by brand).
- **Dataset overview and charts**: explore the cleaned data, price distribution, correlations, and price by brand and model.
- **Model details**: side-by-side comparison of all five trained models.
- **Currency toggle**: switch between ₹ INR (Indian number format) and $ USD from the sidebar.
- **About page**: the full project description inside the app.

## Sections
 
The sidebar is used only for navigation. The selected section opens full width in the main area.
 
| Section | What it shows |
|---|---|
| Dataset Overview | KPI cards, cleaned dataset preview, statistical summary |
| Charts | Six exploratory charts |
| Model Details | Best model, comparison table, R² chart, feature list |
| Single Car Prediction | Input form and result card |
| Bulk Prediction | File upload, predictions table, downloads, batch analytics |
| About | Pipeline, dataset, features, models, usage guide, limits |
 
---


## Quick start
 
**Requirements:** Python 3.10 or newer.
 
```bash
# 1. create and activate a virtual environment
python -m venv venv

venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
 
# 3. Run the app
streamlit run app.py
```

The trained models are already saved in the `artifacts/` folder, so the app starts immediately and no retraining is needed.
 
---

## Dataset
 
Web-scraped used car listings: **2,840** rows, of which **2,782** are used after cleaning, covering **27** brands and model years **2000–2024**.
 
| Column | Example | Meaning |
|---|---|---|
| `name` | `Honda Civic` | Brand and model together |
| `year` | `2020` | Model year |
| `miles` | `37,190 miles` | Odometer reading |
| `color` | `Blue exterior, Black interior` | Exterior and interior colour |
| `condition` | `No accidents reported, 1 Owner` | Accident history and owners |
| `price` | `$21,998` | Asking price (the target) |

### Preparation steps
 
1. Convert text values such as `37,190 miles` and `$21,998` into numbers.
2. Split `color` into exterior and interior colour.
3. Extract `accidents` and `owners` from `condition`. Missing owner counts are filled with the median.
4. Split `name` into brand and model, handling two-word brands (Land Rover, Alfa Romeo, Aston Martin).
5. Remove the cheapest 1% and the most expensive 1% of listings as outliers.
6. Group rare brands (fewer than 15 listings) as `Other`.
7. Split 80% train and 20% test with a fixed random seed.
8. Train on `log1p(price)` and convert predictions back with `expm1`.


### Features used by the models
 
| Feature | Meaning |
|---|---|
| `miles` | Odometer reading |
| `car_age` | 2026 minus model year |
| `miles_per_year` | Miles divided by car age |
| `accidents` | Reported accidents |
| `owners` | Previous owners |
| `brand_encoded` | Mean log-price of the brand in the training data |
| `model_encoder` | Mean log-price of the model in the training data |
| `exterior_encoder_color` | Mean log-price of the exterior colour |
| `interior_encoder_color` | Mean log-price of the interior colour |
 
Brand, model and colour use target (mean) encoding computed on the training split only, so the test score stays honest. Unseen values fall back to the overall average.
 
---


## Models and results
 
All models were evaluated on the same held-out test set. Errors are in US dollars.
 
| Model | Settings | MAE ($) | RMSE ($) | R² |
|---|---|---:|---:|---:|
| **XGBoost** | 400 trees, learning rate 0.05, depth 5 | **2,803** | **4,828** | **0.800** |
| Random Forest | 400 trees | 2,909 | 4,967 | 0.788 |
| KNN Regressor | 7 neighbours, scaled features | 3,792 | 5,767 | 0.714 |
| Linear Regression | scaled features | 3,997 | 5,983 | 0.692 |
| Decision Tree | max depth 8 | 3,719 | 6,037 | 0.687 |
 
MAE is the average gap between predicted and actual price (lower is better). R² closer to 1 is better.
 
> Numbers can vary slightly if you retrain with a different library version.
 
---

## Bulk prediction file format
 
The uploaded file must contain these five columns, in the same style as the raw dataset:
 
| name | year | miles | color | condition |
|---|---|---|---|---|
| Honda Civic | 2020 | 37,190 miles | Blue exterior, Black interior | No accidents reported, 1 Owner |
| Toyota RAV4 | 2022 | 32,879 miles | Silver exterior, Unknown interior | 1 accident reported, 1 Owner |
 
- `miles` can be written as `37,190 miles` or just `37190`.
- Supported formats: `.csv`, `.xlsx`, `.json` (a list of records).
- A ready-made template can be downloaded from the Bulk Prediction page.
- The results file adds a `predicted_price_₹` (or `predicted_price_$`) column.
---

## Limitations
 
- Predictions are estimates, not a formal appraisal or a guaranteed selling price.
- Brands, models or colours not seen in training fall back to the overall average, so those estimates are less reliable.
- The model learns from asking prices in a scraped dataset, so it reflects that market and time period.
- The ₹ conversion uses a fixed rate, not a live exchange rate.
---

## Built with
 
Python · Streamlit · pandas · NumPy · scikit-learn · XGBoost · Plotly · joblib