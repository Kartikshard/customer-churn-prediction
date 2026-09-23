from flask import Flask
import joblib

app = Flask(__name__)

model = joblib.load("stacking_model.pkl")
threshold = joblib.load("threshold.pkl")


@app.route("/")
def home():
    return "Customer Churn Prediction API is running!"


if __name__ == "__main__":
    app.run(debug=True)