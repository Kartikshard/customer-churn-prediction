from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model and threshold
model = joblib.load("stacking_model.pkl")
threshold = joblib.load("threshold.pkl")


@app.route("/")
def home():
    return "Customer Churn Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    customer = pd.DataFrame([data])

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    customer["service_count"] = (
        customer[service_columns] == "Yes"
    ).sum(axis=1)

    customer["avg_monthly_spend"] = np.where(
        customer["tenure"] == 0,
        0,
        customer["TotalCharges"] / customer["tenure"]
    )

    additional_services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    customer["additional_service_count"] = (
        customer[additional_services] == "Yes"
    ).sum(axis=1)

    customer["is_month_to_month"] = (
        customer["Contract"] == "Month-to-month"
    )

    support_services = [
        "OnlineSecurity",
        "TechSupport"
    ]

    customer["has_support_service"] = np.where(
        (customer[support_services] == "Yes").sum(axis=1) > 0,
        1,
        0
    )

    customer["is_electronic_check"] = np.where(
        customer["PaymentMethod"] == "Electronic check",
        1,
        0
    )

    customer["is_fiber_optic"] = np.where(
        customer["InternetService"] == "Fiber optic",
        1,
        0
    )

    # -----------------------------
    # Prediction
    # -----------------------------

    probability = model.predict_proba(customer)[0, 1]

    prediction = (
        "Yes"
        if probability >= threshold
        else "No"
    )

    return jsonify({
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction
    })


if __name__ == "__main__":
    app.run(debug=True)