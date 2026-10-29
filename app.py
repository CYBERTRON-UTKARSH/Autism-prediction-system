from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask import render_template

app = Flask(__name__)

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-form", methods=["POST"])
def predict_form():

    data = {
        "A1_Score": int(request.form["A1_Score"]),
        "A2_Score": int(request.form["A2_Score"]),
        "A3_Score": int(request.form["A3_Score"]),
        "A4_Score": int(request.form["A4_Score"]),
        "A5_Score": int(request.form["A5_Score"]),
        "A6_Score": int(request.form["A6_Score"]),
        "A7_Score": int(request.form["A7_Score"]),
        "A8_Score": int(request.form["A8_Score"]),
        "A9_Score": int(request.form["A9_Score"]),
        "A10_Score": int(request.form["A10_Score"]),
        "age": int(request.form["age"]),
        "gender": request.form["gender"],
        "ethnicity": request.form["ethnicity"],
        "jaundice": request.form["jaundice"],
        "austim": request.form["austim"],
        "contry_of_res": request.form["contry_of_res"],
        "used_app_before": request.form["used_app_before"],
        "result": float(request.form["result"]),
        "relation": request.form["relation"]
    }

    df = pd.DataFrame([data])

    for col in encoders:
        if col in df.columns:
            df[col] = encoders[col].transform(df[col])

    prediction = model.predict(df)

    result = (
    "⚠️ Autism Detected"
    if prediction[0] == 1
    else "✅ No Autism Detected"
)

    return render_template(
        "index.html",
        prediction=result
    )
    
print(encoders.keys())

for col, encoder in encoders.items():
    print(f"\n{col}")
    print(encoder.classes_)
    
if __name__ == "__main__":
    app.run(debug=True)