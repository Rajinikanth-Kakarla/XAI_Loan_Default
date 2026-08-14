from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd


app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "xgboost_credit_default_model.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # ----------------------------------------------------
        # Convert user-friendly categorical values
        # ----------------------------------------------------

        sex_map = {
            "Male": 1,
            "Female": 2
        }

        education_map = {
            "Graduate School": 1,
            "University": 2,
            "High School": 3,
            "Other": 4
        }

        marriage_map = {
            "Married": 1,
            "Single": 2,
            "Other": 3
        }


        # ----------------------------------------------------
        # Create model input
        # ----------------------------------------------------

        customer = {

            "LIMIT_BAL": float(data["LIMIT_BAL"]),

            "SEX": sex_map[data["SEX"]],

            "EDUCATION":
                education_map[data["EDUCATION"]],

            "MARRIAGE":
                marriage_map[data["MARRIAGE"]],

            "AGE": int(data["AGE"]),

            "PAY_0": int(data["PAY_0"]),
            "PAY_2": int(data["PAY_2"]),
            "PAY_3": int(data["PAY_3"]),
            "PAY_4": int(data["PAY_4"]),
            "PAY_5": int(data["PAY_5"]),
            "PAY_6": int(data["PAY_6"]),

            "BILL_AMT1": float(data["BILL_AMT1"]),
            "BILL_AMT2": float(data["BILL_AMT2"]),
            "BILL_AMT3": float(data["BILL_AMT3"]),
            "BILL_AMT4": float(data["BILL_AMT4"]),
            "BILL_AMT5": float(data["BILL_AMT5"]),
            "BILL_AMT6": float(data["BILL_AMT6"]),

            "PAY_AMT1": float(data["PAY_AMT1"]),
            "PAY_AMT2": float(data["PAY_AMT2"]),
            "PAY_AMT3": float(data["PAY_AMT3"]),
            "PAY_AMT4": float(data["PAY_AMT4"]),
            "PAY_AMT5": float(data["PAY_AMT5"]),
            "PAY_AMT6": float(data["PAY_AMT6"])
        }


        # ----------------------------------------------------
        # Convert to DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [customer],
            columns=FEATURES
        )


        # ----------------------------------------------------
        # REAL MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]


# Convert NumPy values to normal Python values
        prediction = int(prediction)

        probability = float(probability)

        probability_percent = round(
            probability * 100,
            2
        )


        if prediction == 1:

            risk = "Higher Risk of Default"

        else:

            risk = "Lower Risk of Default"


        return jsonify({

            "success": True,

            "prediction": prediction,

            "risk": risk,

            "probability": probability_percent

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )