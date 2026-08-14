from flask import Flask, send_from_directory, request, jsonify
from pathlib import Path

import joblib
import pandas as pd
import shap
import numpy as np


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = BASE_DIR / "xgboost_credit_default_model.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)


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
# FRIENDLY FEATURE NAMES FOR XAI
# ============================================================

FEATURE_NAMES = {

    "LIMIT_BAL":
        "Credit Limit",

    "SEX":
        "Sex",

    "EDUCATION":
        "Education",

    "MARRIAGE":
        "Marriage Status",

    "AGE":
        "Age",

    "PAY_0":
        "Repayment Status - Latest Month",

    "PAY_2":
        "Repayment Status - 2 Months Ago",

    "PAY_3":
        "Repayment Status - 3 Months Ago",

    "PAY_4":
        "Repayment Status - 4 Months Ago",

    "PAY_5":
        "Repayment Status - 5 Months Ago",

    "PAY_6":
        "Repayment Status - 6 Months Ago",

    "BILL_AMT1":
        "Bill Amount - Latest Month",

    "BILL_AMT2":
        "Bill Amount - 2 Months Ago",

    "BILL_AMT3":
        "Bill Amount - 3 Months Ago",

    "BILL_AMT4":
        "Bill Amount - 4 Months Ago",

    "BILL_AMT5":
        "Bill Amount - 5 Months Ago",

    "BILL_AMT6":
        "Bill Amount - 6 Months Ago",

    "PAY_AMT1":
        "Payment Amount - Latest Month",

    "PAY_AMT2":
        "Payment Amount - 2 Months Ago",

    "PAY_AMT3":
        "Payment Amount - 3 Months Ago",

    "PAY_AMT4":
        "Payment Amount - 4 Months Ago",

    "PAY_AMT5":
        "Payment Amount - 5 Months Ago",

    "PAY_AMT6":
        "Payment Amount - 6 Months Ago"
}


# ============================================================
# CATEGORY MAPPINGS
# ============================================================

SEX_MAP = {
    "Male": 1,
    "Female": 2
}


EDUCATION_MAP = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Other": 4
}


MARRIAGE_MAP = {
    "Married": 1,
    "Single": 2,
    "Other": 3
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Get JSON data from website
        # ----------------------------------------------------

        data = request.get_json()

        if data is None:

            return jsonify({
                "success": False,
                "error": "No JSON data received."
            }), 400


        # ====================================================
        # CREATE CUSTOMER DATA
        # ====================================================

        customer = {

            "LIMIT_BAL":
                float(data["LIMIT_BAL"]),

            "SEX":
                SEX_MAP[data["SEX"]],

            "EDUCATION":
                EDUCATION_MAP[data["EDUCATION"]],

            "MARRIAGE":
                MARRIAGE_MAP[data["MARRIAGE"]],

            "AGE":
                int(data["AGE"]),

            "PAY_0":
                int(data["PAY_0"]),

            "PAY_2":
                int(data["PAY_2"]),

            "PAY_3":
                int(data["PAY_3"]),

            "PAY_4":
                int(data["PAY_4"]),

            "PAY_5":
                int(data["PAY_5"]),

            "PAY_6":
                int(data["PAY_6"]),

            "BILL_AMT1":
                float(data["BILL_AMT1"]),

            "BILL_AMT2":
                float(data["BILL_AMT2"]),

            "BILL_AMT3":
                float(data["BILL_AMT3"]),

            "BILL_AMT4":
                float(data["BILL_AMT4"]),

            "BILL_AMT5":
                float(data["BILL_AMT5"]),

            "BILL_AMT6":
                float(data["BILL_AMT6"]),

            "PAY_AMT1":
                float(data["PAY_AMT1"]),

            "PAY_AMT2":
                float(data["PAY_AMT2"]),

            "PAY_AMT3":
                float(data["PAY_AMT3"]),

            "PAY_AMT4":
                float(data["PAY_AMT4"]),

            "PAY_AMT5":
                float(data["PAY_AMT5"]),

            "PAY_AMT6":
                float(data["PAY_AMT6"])
        }


        # ====================================================
        # CREATE DATAFRAME
        # ====================================================

        input_df = pd.DataFrame(
            [customer],
            columns=FEATURES
        )


        # ====================================================
        # REAL XGBOOST PREDICTION
        # ====================================================

        prediction = model.predict(
            input_df
        )[0]

        probability = model.predict_proba(
            input_df
        )[0][1]


        # ----------------------------------------------------
        # Convert NumPy values to Python values
        # ----------------------------------------------------

        prediction = int(prediction)

        probability = float(probability)

        probability_percent = round(
            probability * 100,
            2
        )


        # ====================================================
        # RISK CLASSIFICATION
        # ====================================================

        if prediction == 1:

            risk = "Higher Risk of Default"

        else:

            risk = "Lower Risk of Default"


        # ====================================================
        # SHAP XAI
        # ====================================================

        shap_values = explainer.shap_values(
            input_df
        )


        # ----------------------------------------------------
        # Handle different SHAP output formats
        # ----------------------------------------------------

        if isinstance(shap_values, list):

            shap_values = shap_values[-1]


        shap_values = np.asarray(
            shap_values
        )


        # ----------------------------------------------------
        # Single customer
        # ----------------------------------------------------

        if shap_values.ndim > 1:

            shap_values = shap_values[0]


        # ----------------------------------------------------
        # Make sure number of SHAP values matches features
        # ----------------------------------------------------

        if len(shap_values) != len(FEATURES):

            raise ValueError(
                "SHAP output does not match "
                "the number of model features."
            )


        # ====================================================
        # CREATE XAI DATAFRAME
        # ====================================================

        explanation_df = pd.DataFrame({

            "feature":
                FEATURES,

            "value":
                input_df.iloc[0].values,

            "shap_value":
                shap_values

        })


        # ----------------------------------------------------
        # Absolute SHAP importance
        # ----------------------------------------------------

        explanation_df["importance"] = (
            explanation_df["shap_value"].abs()
        )


        # ----------------------------------------------------
        # Sort by importance
        # ----------------------------------------------------

        explanation_df = explanation_df.sort_values(
            "importance",
            ascending=False
        )


        # ----------------------------------------------------
        # Get top 5 factors
        # ----------------------------------------------------

        top_features = explanation_df.head(5)


        # ====================================================
        # PREPARE XAI RESPONSE
        # ====================================================

        xai_factors = []


        for _, row in top_features.iterrows():

            feature = row["feature"]

            shap_value = float(
                row["shap_value"]
            )

            value = row["value"]


            # ------------------------------------------------
            # Convert value safely
            # ------------------------------------------------

            if isinstance(
                value,
                (np.integer, np.floating)
            ):

                value = float(value)


            # ------------------------------------------------
            # Direction
            # ------------------------------------------------

            if shap_value > 0:

                direction = "increases risk"

            elif shap_value < 0:

                direction = "decreases risk"

            else:

                direction = "has minimal impact"


            # ------------------------------------------------
            # Add factor
            # ------------------------------------------------

            xai_factors.append({

                "feature":
                    FEATURE_NAMES.get(
                        feature,
                        feature
                    ),

                "value":
                    value,

                "impact":
                    round(
                        shap_value,
                        4
                    ),

                "direction":
                    direction

            })


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        return jsonify({

            "success":
                True,

            "prediction":
                prediction,

            "risk":
                risk,

            "probability":
                probability_percent,

            "xai":
                xai_factors

        })


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except KeyError as e:

        return jsonify({

            "success":
                False,

            "error":
                f"Missing or invalid input: {str(e)}"

        }), 400


    except ValueError as e:

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 400


    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return jsonify({

            "success":
                False,

            "error":
                "Prediction failed. Check the server logs."

        }), 500


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )