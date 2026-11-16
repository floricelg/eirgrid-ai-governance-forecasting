# EirGrid AI Governance and Electricity Demand Forecasting

Reproducible code accompanying the study **Reducerea decalajului de guvernanță
în selecția instrumentelor de inteligență artificială: prognoza cererii de
energie electrică, riscul asimetric și sustenabilitatea**.

## Data
Place the official EirGrid 2025 quarter-hourly workbook in `date/`.
See [date/README.md](date/README.md). Raw data are not redistributed.

## Setup
Python 3.10+ recommended.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python scripturi/run_experiments.py --stage audit
python scripturi/run_experiments.py --stage all
```

`--stage main` skips LSTM for a quicker diagnostic run.
The `all` stage trains the five planned forecasting methods and writes
CSV predictions and H1/H2/H3 tables under `rezultate/`.

## Methods and limits
- Models: seasonal naive, ridge linear regression, Random Forest,
  histogram gradient boosting and a compact LSTM.
- Direct forecasts **only at the endpoints** 24h, 7d and 14d ahead,
  **not** complete hourly/quarter-hourly forecast trajectories.
- Daily forecast origins; target dates are split chronologically
  (60/20/20) with an embargo between partitions.
- Features use demand history available by the forecast origin and
  calendar information at the origin; no future demand or weather.
- Baseline: same quarter-hour from the preceding week at the origin.
- H1/H2: MAE, MAPE and relative MAE gain vs baseline; exploratory
  HAC Diebold–Mariano approximation, not definitive inference.
- H3: pinball loss at tau=0.5, 0.7, 0.9 **applied to point forecasts**.
  These are risk-weighted scores, **not trained/calibrated quantile models**.
- Single-year data and limited long-horizon independent samples mean
  that results should be interpreted as exploratory.
- Training data and tests are not published until experiments are run.
  No empirical improvement is claimed in advance.

## Outputs
`rezultate/data_audit.json`, `predictions_*.csv`,
`tabele/H1_H2_accuracy.csv`, `tabele/H3_pinball.csv`,
`tabele/DM_exploratory.csv`.

## Source
[EirGrid official reports](https://www.eirgrid.ie/grid/system-and-renewable-data-reports).
