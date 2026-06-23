import joblib
import numpy as np

# Load trained model
model = joblib.load('PowerPrediction_Final_RF_Model.pkl')

# Example Inputs
import pandas as pd

inputs = pd.DataFrame({

'Total Export ':[10000],
'Total Import':[500],
'SMS Consumption':[40000],
'RM Consumption':[30000],
'CPP Aux.(KKL)':[15000],
'DRI(KKL)':[45000],
'LF or CF(KKL)':[92]

})

prediction = model.predict(inputs)

print(prediction[0])