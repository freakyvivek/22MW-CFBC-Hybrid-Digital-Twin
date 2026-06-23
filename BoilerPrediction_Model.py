
import pandas as pd
file = "22MW_CFBC_Master_Data_Final.xlsx"
df = pd.read_excel(
    file,
    sheet_name="Boiler_Data"
)
print()

print(df.head())

print()

print("Shape =", df.shape)
features = [

'CFBC COAL CONSUMPTION',

'CFBC CHAR CONSUMPTION',

'D DUST',

'MIX GCV',

'TOTAL STEAM GENERATION',

'TG INLET STEAM FLOW'

]

target = 'BOILER EFFICIENCY'

print()

print("Features Selected Successfully")

data = df[features + [target]]

print()

print("Original Shape =", data.shape)
data = data[data['BOILER EFFICIENCY'] > 0]
data = data.dropna()

print("After Removing NaN =", data.shape)

print()

print(data.head())
X = data[features]

y = data[target]

print()

print("X Shape =", X.shape)

print("y Shape =", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)

print()

print("Training Set Shape :", X_train.shape)

print("Testing Set Shape :", X_test.shape)

from sklearn.ensemble import RandomForestRegressor

RF_Model = RandomForestRegressor(

    n_estimators=200,

    max_depth=5,

    min_samples_split=5,

    random_state=42

)
RF_Model.fit(

    X_train,

    y_train

)

print()

print("Random Forest Training Completed Successfully")
y_pred = RF_Model.predict(

    X_test

)

print()

print("Prediction Completed Successfully")
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

R2 = r2_score(

    y_test,

    y_pred

)

RMSE = np.sqrt(

    mean_squared_error(

        y_test,

        y_pred

    )

)

print()

print("R² Score =",R2)

print("RMSE =",RMSE)

from sklearn.model_selection import GridSearchCV

param_grid = {

    'n_estimators': [100, 200, 300],

    'max_depth': [5, 10, 15],

    'min_samples_split': [2, 5]

}

grid = GridSearchCV(

    estimator=RandomForestRegressor(random_state=42),

    param_grid=param_grid,

    cv=3,

    scoring='r2',

    n_jobs=-1

)

grid.fit(

    X_train,

    y_train

)

print()

print("Best Parameters:")

print(grid.best_params_)

print()

print("Best CV Score:")

print(grid.best_score_)

importance = RF_Model.feature_importances_

print()

print("Feature Importance")

for feature, score in zip(features, importance):

    print(

        feature,

        ":",

        round(score,4)

    )
    import joblib

joblib.dump(

    RF_Model,

    "BoilerPrediction_RF_Model.pkl"

)

print()

print("Model Saved Successfully")
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

# Scatter plot
plt.scatter(
    y_test,
    y_pred
)

# Ideal prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2,
    label="Perfect Prediction"
)

plt.xlabel("Actual Boiler Efficiency")

plt.ylabel("Predicted Boiler Efficiency")

plt.title("Actual vs Predicted Boiler Efficiency")

plt.legend()

plt.grid()

plt.show()

from xgboost import XGBRegressor

# Create XGBoost model
XGB_Model = XGBRegressor(

    n_estimators=200,

    random_state=42

)

# Train model
XGB_Model.fit(

    X_train,

    y_train

)

# Prediction
y_pred_xgb = XGB_Model.predict(

    X_test

)

# Performance metrics
R2_XGB = r2_score(

    y_test,

    y_pred_xgb

)

RMSE_XGB = np.sqrt(

    mean_squared_error(

        y_test,

        y_pred_xgb

    )

)

print()

print("===== XGBOOST RESULTS =====")

print("R² =",R2_XGB)

print("RMSE =",RMSE_XGB)

import shap

# Create explainer
explainer = shap.TreeExplainer(RF_Model)

# SHAP values
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(
    shap_values,
    X_test
)
joblib.dump(

    RF_Model,

    "BoilerPrediction_Final_RF_Model.pkl"

)

print()

print("Final Boiler Model Saved Successfully")