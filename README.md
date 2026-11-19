# MandiForecast — Cross-Market Selection for Agricultural Mandi Price Forecasting

A React + Flask research dashboard based on the uploaded Kaggle notebook.

## What it implements

- Target-only, correlation, Granger, hybrid, and all-markets strategies.
- Pratapgarh + Onion defaults.
- Daily market series; duplicate date-market observations aggregated using mean `Modal_Price`.
- External prices forward-filled for up to 3 days.
- 60/20/20 chronological split.
- Selection scores computed on training dates only.
- Target lags: 1, 2, 3, 7, 14, 30 previous recorded observations.
- Target rolling means: 7, 14, 30 previous recorded observations.
- External market lags: 1, 2, 3, 7 calendar days.
- Median imputation in a training-fitted sklearn pipeline.
- Random Forest and XGBoost (HistGradientBoosting fallback if XGBoost is unavailable).
- Validation RMSE selects the preferred strategy/model; test set remains held out.
- Validation and test results, predictions for validation-selected model, and price-regime diagnostics.

## Requirements

- Python 3.10+
- Node.js 18+

## Start backend

Open a terminal:

```bash
cd backend
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Backend runs at `http://localhost:5000`.

If XGBoost installation is not possible, remove `xgboost` from requirements and the backend uses HistGradientBoosting instead. Note: this fallback is not numerically identical to the notebook's XGBoost results.

## Start frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Dataset

In the dashboard, upload the original `Agriculture_price_dataset.csv` file.

Required columns (exact names):
- `Price Date`
- `Market Name`
- `Commodity`
- `Modal_Price`

The notebook's configuration uses `Pratapgarh` and `Onion` by default. Choose any market/commodity available in your CSV.

## Important research interpretation

The dashboard recomputes metrics from the uploaded CSV and installed libraries. It does not hardcode the research tables supplied in the project discussion. Small differences can occur if the dataset version, dependency versions, or notebook execution environment differs.

The optional geographic strategy is not included as an active strategy because the notebook's coordinate path is `None`. No coordinates file was supplied.

The Granger calculation can be computationally expensive for hundreds of eligible markets. The all-markets model can also be memory- and time-intensive because it creates multiple lagged features for every eligible market.

## API

- `GET /api/health`
- `GET /api/metadata`
- `POST /api/upload` with multipart field `file`
- `POST /api/run` JSON:
  ```json
  {
    "target_market": "Pratapgarh",
    "commodity": "Onion",
    "top_k": 5,
    "min_overlap": 60
  }
  ```

## Scope note

This is a local research prototype, not a live mandi price service. Predictions are based on the historical dataset uploaded by the user.
