# Expected Shot Quality: Modeling NBA Shot-Make Probability

This project estimates the probability that an NBA shot attempt is made using public shot-location and game-context data. It compares several machine learning models and examines how players' actual field goal percentage (FG%) differs from their expected FG%.

**Author:** Bryce Ferguson  
**Course:** COMP365

## What this project does

- Collects 2024–25 NBA shot data from NBA.com using the `nba_api` package.
- Cleans shot records and engineers spatial and game-context features.
- Trains and compares Logistic Regression, Random Forest, and XGBoost against a constant-probability baseline.
- Evaluates the models using accuracy, AUC-ROC, Brier score, log loss, and calibration.
- Produces court and player-level visualizations and a Gradio demo.

## Scope and limitations

- **Players:** The dataset covers the top 100 players by field-goal attempts in the 2024–25 season, not the entire league. Findings may not generalize to other players or seasons.
- **Features:** The available data does not include defender distance, shot-clock time, or fatigue.
- **`LATE_GAME`:** Flags period 4 or later with 300 seconds or less remaining. It does not consider score margin and is not the NBA's official definition of clutch time.
- **`FINAL_5S_OF_PERIOD`:** Flags attempts in the last five seconds of a period; it is not a shot-clock feature.
- **Evaluation:** The split is grouped by game, but is not chronological. The reported scores therefore do not establish how well the models predict future seasons.

## Repository contents

| File | Purpose |
|---|---|
| `data._collection.ipynb` | Retrieves 2024–25 data from NBA.com and creates raw and cleaned shot CSV files |
| `feature_engineering.ipynb` | Creates model features, grouped train/test data, and shot metadata |
| `modeling.ipynb` | Trains and evaluates the baseline and three machine learning models |
| `visualization.ipynb` | Creates calibration, court, and player figures and model predictions |
| `app.py` | Interactive Gradio demo for viewing a player's held-out shots |
| `nba_shot_data_2024_25.csv` | Collected shot-level data |
| `nba_shot_data_2024_25_cleaned.csv` | Cleaned shot-level data used for feature engineering |

Other generated files include `x_all.csv`, `x_train.csv`, `x_test.csv`, `y_train.csv`, `y_test.csv`, `shots_meta.csv`, and `shots_meta_preds.csv`; model artifacts include `log_reg_baseline.pk1`, `random_forest.pkl`, and `xgb_model.json`. The notebooks also generate `calibration_curve.png`, `fig_calibration.png`, `fig_court.png`, and `fig_players.png`.

## Setup

The project was developed in a Windows environment with Python 3.14.7. A GPU is not required. No total notebook runtime has been measured; data collection requires an internet connection and may be slowed by NBA.com rate limits.

Create and activate a virtual environment, then install the packages used by the notebooks and app:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install nba_api pandas numpy scikit-learn xgboost matplotlib joblib gradio jupyter
```

The repository does not currently include a root-level `requirements.txt`. These commands install the direct project dependencies; exact versions may differ from the development environment.

## Data

- **Source:** NBA.com's public statistics endpoints, accessed with `nba_api` (`leaguedashplayerstats` to rank players by field-goal attempts and `shotchartdetail` for shot-level records).
- **Season and scope:** 2024–25 season, top 100 players by field-goal attempts.
- **Size:** The current cleaned CSV contains 101,192 shot records.
- **Data files:** Raw and cleaned CSV snapshots are included in the project folder as `nba_shot_data_2024_25.csv` and `nba_shot_data_2024_25_cleaned.csv`.
- **Refreshing the data:** Run `data._collection.ipynb`. It pauses between requests to reduce rate-limit risk. NBA.com data may be revised, so a new pull may not exactly match the included snapshot. The original pull date is not recorded.

## Reproducing the results

Run the notebooks from the project folder in this order:

```powershell
jupyter nbconvert --to notebook --execute --inplace data._collection.ipynb
jupyter nbconvert --to notebook --execute --inplace feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace visualization.ipynb
```

The collection notebook also performs cleaning, so there is no separate cleaning notebook. Alternatively, open the notebooks in Jupyter and run all cells in the same order. The model metrics are produced by `modeling.ipynb`; figures and prediction metadata are produced by `visualization.ipynb`.

## Modeling settings

- **Random seed:** `random_state=42`.
- **Split:** `GroupShuffleSplit` on `GAME_ID`, with 80% of games for training and 20% for testing. The current split contains 80,962 training shots and 20,230 test shots, with no game shared between the sets. No chronological split was performed.
- **Features:** `SHOT_DISTANCE`, `SHOT_ANGLE`, `PERIOD`, `TIME_REMAINING_SEC`, `FINAL_5S_OF_PERIOD`, `LATE_GAME`, and one-hot encodings of `SHOT_ZONE_BASIC`, `SHOT_ZONE_AREA`, `SHOT_TYPE`, and grouped `ACTION_TYPE`.
- **Constant baseline:** Predicts the training-set make rate for every test shot.
- **Logistic Regression:** `max_iter=10000`.
- **Random Forest:** `n_estimators=100`, `max_depth=16`, `min_samples_leaf=20`, `n_jobs=-1`.
- **XGBoost:** `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `eval_metric='logloss'`.
- Models are trained from scratch; no pretrained models or hyperparameter tuning are used.

## Results

Metrics below are calculated on the held-out test games only. Lower Brier score and log loss are better; higher accuracy and AUC are better.

| Model | Accuracy | AUC-ROC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Constant baseline | 0.5333 | 0.5000 | 0.2489 | 0.6910 |
| Logistic Regression | 0.6159 | 0.6365 | 0.2306 | 0.6514 |
| Random Forest | 0.6161 | 0.6412 | 0.2294 | 0.6491 |
| XGBoost | 0.6116 | 0.6343 | 0.2307 | 0.6514 |

Random Forest achieved the best scores among the three trained models, though the differences were small. Calibration results also show that probability estimates can be inaccurate in some bins; model outputs should therefore be interpreted as estimates, not guarantees.

## Running the demo

After generating `shots_meta_preds.csv` by running `visualization.ipynb`, start the app from the project directory:

```powershell
python app.py
```

Open the local URL printed in the terminal (usually `http://127.0.0.1:7860`), select a player, and view that player's held-out shots coloured by predicted make probability alongside actual and expected FG%.

## AI assistance disclosure

Claude (Anthropic) was used to assist with code development and debugging. Copilot was used to help refine project documentation, including this README. The reported evaluation metrics are outputs from the project notebooks.

## Data source and licensing

Shot data is sourced from NBA.com through `nba_api`. Use of NBA.com data is subject to NBA.com's terms of use. The `nba_api` package is an open-source third-party client; consult its project license and NBA.com terms before redistribution or other reuse. The project does not claim ownership of the underlying NBA data.
