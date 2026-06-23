import pandas as pd

file = "22MW_CFBC_Master_Data_Final.xlsx"

boiler_df = pd.read_excel(
    file,
    sheet_name="Boiler_Data"
)

health_df = pd.read_excel(
    file,
    sheet_name="Health_Data"
)
df = pd.DataFrame()

df['PLF'] = health_df['PLF']

df['THR'] = health_df['TURBINE HEAT RATE']

df['SHR'] = health_df['STATION HEAT RATE']

df['SFC'] = health_df['SPECIFIC FUEL CONSUMPTION']

df['POWER LOSS'] = health_df['POWER LOSS']

df['BOILER EFFICIENCY'] = boiler_df['BOILER EFFICIENCY']
df = df.dropna()

df = df[
    (df['PLF'] > 0)
    &
    (df['THR'] > 0)
    &
    (df['SHR'] > 0)
    &
    (df['SFC'] > 0)
    &
    (df['POWER LOSS'] > 0)
    &
    (df['BOILER EFFICIENCY'] > 0)
]

print()

print(df.shape)

print()

print(df.head())

# PLF Score
df['PLF_SCORE'] = 100 * (
    (df['PLF'] - df['PLF'].min())
    /
    (df['PLF'].max() - df['PLF'].min())
)

# THR Score (lower is better)
df['THR_SCORE'] = 100 * (
    (df['THR'].max() - df['THR'])
    /
    (df['THR'].max() - df['THR'].min())
)

# SHR Score
df['SHR_SCORE'] = 100 * (
    (df['SHR'].max() - df['SHR'])
    /
    (df['SHR'].max() - df['SHR'].min())
)

# SFC Score
df['SFC_SCORE'] = 100 * (
    (df['SFC'].max() - df['SFC'])
    /
    (df['SFC'].max() - df['SFC'].min())
)

# Power Loss Score
df['POWERLOSS_SCORE'] = 100 * (
    (df['POWER LOSS'].max() - df['POWER LOSS'])
    /
    (df['POWER LOSS'].max() - df['POWER LOSS'].min())
)

# Health Index
df['HEALTH_INDEX'] = (

0.30 * df['PLF_SCORE']

+ 0.25 * df['THR_SCORE']

+ 0.20 * df['SHR_SCORE']

+ 0.15 * df['SFC_SCORE']

+ 0.10 * df['POWERLOSS_SCORE']

)
df['BE_SCORE'] = 100 * (
    (df['BOILER EFFICIENCY'] - df['BOILER EFFICIENCY'].min())
    /
    (df['BOILER EFFICIENCY'].max() - df['BOILER EFFICIENCY'].min())
)
df['MAINTENANCE_INDEX'] = (

0.35 * df['HEALTH_INDEX']

+ 0.25 * df['BE_SCORE']

+ 0.20 * df['PLF_SCORE']

+ 0.10 * df['THR_SCORE']

+ 0.10 * df['POWERLOSS_SCORE']

)
print()

print(
    df[['HEALTH_INDEX',
        'BE_SCORE',
        'MAINTENANCE_INDEX']].head()
)

print()

print(
    df['MAINTENANCE_INDEX'].describe()
)
features = [

'PLF',

'THR',

'SHR',

'SFC',

'POWER LOSS',

'BOILER EFFICIENCY',

'HEALTH_INDEX'

]

target = 'MAINTENANCE_INDEX'

X = df[features]

y = df[target]

print()

print("X Shape =",X.shape)

print("y Shape =",y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)

print()

print("Training Shape =",X_train.shape)

print("Testing Shape =",X_test.shape)
from sklearn.ensemble import RandomForestRegressor

RF_Model = RandomForestRegressor(

    n_estimators=200,

    random_state=42

)

RF_Model.fit(

    X_train,

    y_train

)

print()

print("Random Forest Training Completed")
y_pred = RF_Model.predict(

    X_test

)
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

    "MaintenancePrediction_RF_Model.pkl"

)

print()

print("Model Saved Successfully")
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2,
    label="Perfect Prediction"
)

plt.xlabel("Actual Maintenance Index")

plt.ylabel("Predicted Maintenance Index")

plt.title("Actual vs Predicted Maintenance Index")

plt.legend()

plt.grid()

plt.show()
plt.figure(figsize=(10,6))

plt.plot(
    y_test.values,
    label="Actual"
)

plt.plot(
    y_pred,
    label="Predicted"
)

plt.xlabel("Sample Number")

plt.ylabel("Maintenance Index")

plt.title("Actual vs Predicted Maintenance Index")

plt.legend()

plt.grid()

plt.show()
from xgboost import XGBRegressor

# Create model
XGB_Model = XGBRegressor(

    n_estimators=200,

    random_state=42

)

# Train
XGB_Model.fit(

    X_train,

    y_train

)

# Prediction
y_pred_xgb = XGB_Model.predict(

    X_test

)

# Metrics
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
from sklearn.model_selection import GridSearchCV

param_grid = {

    'n_estimators':[100,200,300],

    'max_depth':[5,10,15],

    'min_samples_split':[2,5]

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
import shap

explainer = shap.TreeExplainer(RF_Model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(

    shap_values,

    X_test

)
import joblib

joblib.dump(

    RF_Model,

    "MaintenancePrediction_Final_RF_Model.pkl"

)

print()

print("Final Maintenance Model Saved Successfully")