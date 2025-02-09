from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.responses import  FileResponse
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any

# Load saved model and preprocessing tools
model = joblib.load("best_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Define API
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Define request model
class InputData(BaseModel):
    data: Dict[str, Any] = {
        "City": "",
        "Date": "",
        "Humidity (%)": 0,
        "Weather Description": "",
        "Wind Speed(m/s)": 0
    }

# Preprocess input function
def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    
    # Extract date features
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year.fillna(0).astype(int)
        df['Month'] = df['Date'].dt.month.fillna(0).astype(int)
        df['Day'] = df['Date'].dt.day.fillna(0).astype(int)
        df.drop(columns=['Date'], inplace=True)
    
    # Handle missing values
    if 'Humidity (%)' in df.columns:
        df['Humidity (%)'].fillna(df['Humidity (%)'].mean(), inplace=True)
    
    # Encode categorical features
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    # Ensure only trained features are used
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    # Scale numerical features
    df[feature_columns] = scaler.transform(df[feature_columns])
    
    return df

@app.post("/predict")
async def predict(
    City: str = Form(...),
    Date: str = Form(...),
    Humidity: float = Form(...),
    Weather: str = Form(...),
    WindSpeed: float = Form(...)
):
    try:
        input_data = {
            "City": City,
            "Date": Date,
            "Humidity (%)": Humidity,
            "Weather Description": Weather,
            "Wind Speed(m/s)": WindSpeed
        }
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)[0]
       # Change response format: Adding a "status" and "message"
        response = {
            "status": "success",
            "message": "Prediction was successful.",
            "result": {
                "predicted_temperature": prediction
            }
        }
        return JSONResponse(content=response)
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}



@app.get("/")
async def read_root():
    return FileResponse("static/index.html")
