# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("housing_price_api")

# Create input/output pydantic models
input_model = create_model("housing_price_api_input", **{'MSSubClass': 'SPLIT FOYER', 'MSZoning': 'RL', 'LotFrontage': 57.0, 'LotArea': 8846, 'Alley': nan, 'LandContour': 'Lvl', 'LotConfig': 'CulDSac', 'LandSlope': 0, 'BldgType': '1Fam', 'HouseStyle': 'SFoyer', 'OverallQual': 5, 'OverallCond': 5, 'YearBuilt': 1996, 'YearRemodAdd': 1996, 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'MasVnrType': 'None', 'MasVnrArea': 0, 'ExterQual': 4, 'ExterCond': 3, 'Foundation': 'PConc', 'BsmtQual': 4, 'BsmtCond': 3, 'BsmtExposure': 2, 'BsmtFinType1': 6, 'BsmtFinSF1': 298, 'BsmtFinType2': 1, 'BsmtFinSF2': 0, 'BsmtUnfSF': 572, 'TotalBsmtSF': 870, 'Heating': 'GasA', 'HeatingQC': 5, 'CentralAir': 1, 'Electrical': 5, '1stFlrSF': 914, '2ndFlrSF': 0, 'GrLivArea': 914, 'GarageType': 'Detchd', 'GarageYrBlt': 1998, 'GarageFinish': 1, 'GarageCars': 2, 'GarageArea': 576, 'GarageQual': 3, 'PavedDrive': 2, 'WoodDeckSF': 0, 'OpenPorchSF': 0, 'EnclosedPorch': 0, '3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 'PoolQC': 0, 'Fence': 0, 'MiscFeature': nan, 'MoSold': 'July', 'YrSold': 2006})
output_model = create_model("housing_price_api_output", prediction=148000)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
