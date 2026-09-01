from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load models using paths relative to this project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
encoder_path = os.path.join(BASE_DIR, "models", "encoder.pkl")

model = joblib.load(model_path)
le = joblib.load(encoder_path)


@app.route("/", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:
            features = [
                float(request.form["chamber_humidity_pct"]),
                float(request.form["bed_temp_actual"]),
                float(request.form["filament_diameter_mm"]),
                float(request.form["vibration_y_g"]),
                float(request.form["travel_speed_mm_s"]),
                float(request.form["temp_variance"]),
                float(request.form["retraction_distance_mm"]),
                float(request.form["retraction_speed_mm_s"]),
                float(request.form["fan_speed_pct"])
            ]

            prediction = model.predict([features])[0]

            ans = le.inverse_transform([prediction])[0]

            return render_template(
                "index.html",
                ans=ans
            )

        except Exception as e:
            return render_template(
                "index.html",
                error=str(e)
            )

    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
