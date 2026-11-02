# Ames Housing Price Prediction

This project explores the Ames, Iowa housing data and builds regression models to predict residential home sale prices. The analysis and experiments are in [`notebooks/model.ipynb`](./notebooks/model.ipynb).

## What I did

1. **Loaded and inspected the data.** Read the training and test CSV files, checked their shapes and column types, summarized numeric columns, and counted missing values.
2. **Explored the target and feature relationships.** Plotted the SalePrice distribution before and after a `log1p` transform, examined LotArea, plotted SalePrice against GrLivArea and OverallQual, inspected large homes and the highest quality ratings, and reviewed numeric correlations with SalePrice.
3. **Created a baseline pipeline.** Removed `Id` from the model inputs and transformed the target with `log1p`. The preprocessing pipeline imputes numeric missing values with the median, scales numeric features, imputes categorical missing values with the most frequent category, and one-hot encodes categories. A Ridge regressor was evaluated with shuffled five-fold cross-validation (random seed 2) using RMSE on log-transformed SalePrice.
4. **Tried a cleaned training set.** Removed training rows where `GrLivArea > 4000` and compared the CV score. For later pipelines, missing categorical values were instead filled with the string `"None"` before one-hot encoding.
5. **Tried transforming skewed numeric features.** Selected numeric columns with skewness greater than 2, applied `log1p` to those columns, and compared Ridge results. Ridge's `alpha` was tuned with `GridSearchCV`.
6. **Added house-related features.** Built `TotalSF` from basement and above-ground floor areas; `TotalBath` from full and half baths; `TotalPorchSF` from porch areas and WoodDeckSF; and `HouseAge` and `RemodAge` from the sale, build, and remodel years. Recomputed the skewed-feature list and tuned an ElasticNet model on these features.
7. **Compared other regressors.** Evaluated Gradient Boosting, CatBoost, and XGBoost with the same five-fold split strategy, and tried several tree depths for CatBoost and XGBoost.
8. **Generated a test prediction file.** Applied the engineered features to the test set, predicted with the fitted ElasticNet grid search, converted predictions back from log space with `expm1`, and saved `submissions/submission2.csv` with `Id` and `SalePrice` columns. An earlier Ridge baseline is saved as `submissions/submission_baseline.csv`.

## Results recorded in the notebook

The best recorded CV result is an ElasticNet RMSE of approximately **0.1122** on log-transformed SalePrice. Other recorded results include Ridge with skewed-feature transforms (about 0.1141 after alpha tuning), Gradient Boosting (about 0.1138), CatBoost (about 0.1163), and XGBoost (about 0.1199; the depth-2 trial is about 0.1188).

These are exploratory cross-validation results, not a final unbiased estimate: several models and settings were compared using the same folds. The skewed-feature list is also selected using the full cleaned training set before cross-validation, so the validation process is not fully isolated.

## Important modeling caveat

The `GrLivArea > 4000` cutoff removes four training examples, including expensive homes, while the test set contains a home with `GrLivArea = 5095`. This means the final model must extrapolate beyond the range retained in training. The cutoff should be reviewed and justified before treating the score or predictions as final.

Also, `TotalPorchSF` currently includes `WoodDeckSF`, despite a deck not being a porch.

## Repository layout

```text
data/                  Training, test, and sample submission CSVs
notebooks/model.ipynb  Exploration, preprocessing, model experiments, and predictions
submissions/           Baseline and engineered-feature prediction CSVs
```

## Running the notebook

Install the libraries used by the notebook:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn catboost xgboost jupyter
```

Run Jupyter with the working directory set to `notebooks/`, because the notebook reads the CSV files using paths such as `../data/train.csv`:

```bash
cd notebooks
jupyter notebook model.ipynb
```

The data files are included in this repository. If redistributing the dataset, check the terms and source from which it was obtained.
