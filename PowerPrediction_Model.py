import pandas as pd
import sys

print(sys.executable)

# Load Excel file
file = "GEN.xlsx"

excel_file = pd.ExcelFile(file)

# Empty list to store dataframes
dfs = []

# Ignore these sheets
ignore_sheets = ['Sheet2', 'Sheet3']

for sheet in excel_file.sheet_names:

    if sheet not in ignore_sheets:

        print("Reading:", sheet)
        df = pd.read_excel(
    file,
    sheet_name=sheet,
    header=1
)


        # Add month name
        df['Month'] = sheet

        dfs.append(df)

# Merge all sheets
master_df = pd.concat(dfs, ignore_index=True)
# Remove rows having empty Date
master_df = master_df.dropna(subset=['Date'])

# Remove TOTAL rows
master_df = master_df[
    master_df['Date'].astype(str)
    .str.contains('TOTAL', case=False, na=False) == False
]

# Reset index
master_df = master_df.reset_index(drop=True)

print(master_df.shape)

print()
print("Dataset Shape:")
print(master_df.shape)

print()
print(master_df.columns)
print()
print(master_df.shape)
features = [
'Total Export ',
'Total Import',
'SMS Consumption',
'RM Consumption',
'CPP Aux.(KKL)',
'DRI(KKL)',
'LF or CF(KKL)'
]

target = 'TG generation(KKL)'

# Keep only required columns
data = master_df[features + [target]]
print(data.isna().sum())

# Remove rows containing missing values
data = data.dropna()

# Create X and Y
X = data[features]

Y = data[target]
print()

print(X.describe())

print()

print(Y.describe())

print(X.shape)
print(Y.shape)

print(X.shape)
print(Y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

from sklearn.ensemble import RandomForestRegressor

RF_Model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

RF_Model.fit(X_train, y_train)


y_pred = RF_Model.predict(X_test)


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

R2 = r2_score(y_test, y_pred)

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nR² Score =", R2)

print("RMSE =", RMSE)

import matplotlib.pyplot as plt

importance = RF_Model.feature_importances_

feature_names = X.columns

plt.figure(figsize=(10,6))
plt.bar(feature_names, importance)
plt.xticks(rotation=45)
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual TG Generation")
plt.ylabel("Predicted TG Generation")

plt.title("Actual vs Predicted")

plt.grid()

plt.show()

import joblib

joblib.dump(RF_Model,'PowerPrediction_Model.pkl')

print("Model Saved Successfully")

# Feature Importance
importance = RF_Model.feature_importances_
feature_names = X.columns

# Plot Feature Importance
plt.figure(figsize=(10,6))
plt.bar(feature_names, importance)
plt.xticks(rotation=45)
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()

# Print numerical values
for feature, score in zip(feature_names, importance):
    print(feature,":",round(score,4))


# ---------------------------
# Actual vs Predicted Plot
# ---------------------------

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual TG Generation")
plt.ylabel("Predicted TG Generation")
plt.title("Actual vs Predicted")
plt.grid()

plt.show()


from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import numpy as np

# XGBoost model
XGB_Model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

# Training
XGB_Model.fit(X_train, y_train)

# Prediction
y_pred_xgb = XGB_Model.predict(X_test)

# Metrics
R2_XGB = r2_score(y_test, y_pred_xgb)
RMSE_XGB = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print()
print("===== XGBOOST RESULTS =====")
print("R² =", R2_XGB)
print("RMSE =", RMSE_XGB)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    RF_Model,
    X,
    Y,
    cv=5,
    scoring='r2'
)

print()
print("Cross Validation Scores:")
print(scores)

print()
print("Average R²:", scores.mean())

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print()
print("Best CV Score:")
print(grid_search.best_score_)

Best_RF_Model = grid_search.best_estimator_

Best_RF_Model.fit(X_train, y_train)

y_pred_best = Best_RF_Model.predict(X_test)

R2_best = r2_score(y_test, y_pred_best)

RMSE_best = np.sqrt(mean_squared_error(y_test, y_pred_best))

print()
print("===== FINAL OPTIMIZED RANDOM FOREST =====")
print("R² =", R2_best)
print("RMSE =", RMSE_best)

import joblib

joblib.dump(
    Best_RF_Model,
    "PowerPrediction_Final_RF_Model.pkl"
)

print("Final Model Saved Successfully")

import shap
import matplotlib.pyplot as plt
# Create SHAP Explainer
explainer = shap.TreeExplainer(Best_RF_Model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

print("SHAP Values Calculated Successfully")
plt.figure(figsize=(10,6))

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    "SHAP_Summary.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()
plt.figure(figsize=(10,6))

shap.summary_plot(
    shap_values,
    X_test,
    plot_type='bar',
    show=False
)

plt.tight_layout()

plt.savefig(
    "SHAP_Bar.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

sample_index = 0
sample = X_test.iloc[sample_index]

explanation = shap.Explanation(
    values=shap_values[sample_index],
    base_values=explainer.expected_value.mean(),
    data=sample.values,
    feature_names=X_test.columns
)

shap.plots.waterfall(explanation)

shap.dependence_plot(
    "LF or CF(KKL)",
    shap_values,
    X_test
)

shap.dependence_plot(
    "CPP Aux.(KKL)",
    shap_values,
    X_test
)