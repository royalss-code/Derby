from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = pickle.load(open(os.path.join(os.path.dirname(__file__), "model.pkl"), "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["player_hr_rate"]),
            float(request.form["barrel_rate"]),
            float(request.form["pitcher_hr9"]),
            float(request.form["flyball_rate"]),
            float(request.form["matchup"]),
            float(request.form["park_factor"])
        ]

        prob = model.predict_proba([features])[0][1]
        prediction = f"{prob:.2%}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
