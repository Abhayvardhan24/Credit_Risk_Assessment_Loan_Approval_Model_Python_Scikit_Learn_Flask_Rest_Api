from flask import Flask, render_template, request
import pickle
import pandas as pd
from utils.explain import generate_eligibility_explanation

app = Flask(__name__)

# Load trained model
with open("model/model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Home Page
@app.route("/")
def index():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    # Collect all form data
    form_data = {
        "age": int(request.form["age"]),
        "occupation_status": request.form["occupation_status"],
        "years_employed": float(request.form["years_employed"]),
        "annual_income": float(request.form["annual_income"]),
        "credit_score": float(request.form["credit_score"]),
        "credit_history_years": float(request.form["credit_history_years"]),
        "savings_assets": float(request.form["savings_assets"]),
        "current_debt": float(request.form["current_debt"]),
        "defaults_on_file": int(request.form["defaults_on_file"]),
        "delinquencies_last_2yrs": int(request.form["delinquencies_last_2yrs"]),
        "derogatory_marks": int(request.form["derogatory_marks"]),
        "product_type": request.form["product_type"],
        "loan_intent": request.form["loan_intent"],
        "loan_amount": float(request.form["loan_amount"]),
        "interest_rate": float(request.form["interest_rate"]),
        "debt_to_income_ratio": float(request.form["debt_to_income_ratio"]),
        "loan_to_income_ratio": float(request.form["loan_to_income_ratio"]),
        "payment_to_income_ratio": float(request.form["payment_to_income_ratio"])
    }

    # Prepare data for model
    df = pd.DataFrame([form_data])

    # Model prediction
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    # Generate human-readable explanations
    explanation = generate_eligibility_explanation(form_data, prediction)

    return render_template(
        "result.html",
        data=form_data,
        eligibility=explanation["decision"],
        probability=round(proba * 100, 2),
        reasons=explanation["reasons"]
    )

if __name__ == "__main__":
    app.run(debug=True)
