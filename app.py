import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize the FastAPI application
app = FastAPI(
    title="E-Commerce Return Prediction API", 
    description="API to predict the risk and financial cost of product returns.",
    version="1.0"
)

# 2. Load the pre-trained pipeline
model = joblib.load('return_prediction_model.pkl')

# 3. Define the structural schema for incoming checkout requests
class OrderInput(BaseModel):
    customer_age: int
    purchase_amount: float
    product_category: str
    discount_applied: int
    customer_rating: int
    previous_returns: int

# 4. Create the API endpoint
@app.post("/predict")
def predict_return(order: OrderInput):
    # Convert incoming JSON payload into a Pandas DataFrame format expected by the model
    input_data = pd.DataFrame([order.model_dump()])

    # Run predictions
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    # Financial Impact calculation ($15 default reverse logistics cost)
    logistics_cost_risk = 15.00 if prediction == 1 else 0.00

    return {
        "return_predicted": prediction,
        "return_probability": round(probability, 4),
        "estimated_financial_risk_usd": logistics_cost_risk,
        "recommendation": "Flag for logistics optimization" if prediction == 1 else "Standard Processing"
    }
