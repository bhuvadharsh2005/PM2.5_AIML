from pathlib import Path
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = PROJECT_ROOT / "models" / "pm25_linear_regression.joblib"

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model file was not found: {MODEL_FILE}")

artifact = joblib.load(MODEL_FILE)

model_pipeline = artifact["pipeline"]
features = artifact["features"]
metrics = artifact.get("metrics", {})

FEATURE_LABELS = {
    "pm10": "PM10",
    "no2": "Nitrogen Dioxide",
    "so2": "Sulphur Dioxide",
    "co": "Carbon Monoxide",
    "ozone": "Ozone",
    "humidity": "Relative Humidity",
    "wind_speed": "Wind Speed",
}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error_message = None
    form_values = {}

    if request.method == "POST":
        try:
            numerical_values = {}

            for feature in features:
                raw_value = request.form.get(feature, "").strip()

                form_values[feature] = raw_value

                if raw_value == "":
                    raise ValueError(
                        f"Please enter a value for {FEATURE_LABELS[feature]}."
                    )

                numerical_values[feature] = float(raw_value)

            pollutant_features = [
                "pm10",
                "no2",
                "so2",
                "co",
                "ozone",
            ]

            for feature in pollutant_features:
                if numerical_values[feature] < 0:
                    raise ValueError(
                        f"{FEATURE_LABELS[feature]} cannot be negative."
                    )

            if not 0 <= numerical_values["humidity"] <= 100:
                raise ValueError(
                    "Relative humidity must be between 0 and 100."
                )

            if not 0 <= numerical_values["wind_speed"] <= 50:
                raise ValueError(
                    "Wind speed must be between 0 and 50 m/s."
                )

            input_frame = pd.DataFrame(
                [numerical_values],
                columns=features,
            )

            prediction = model_pipeline.predict(input_frame)[0]

            prediction = max(0.0, float(prediction))

        except ValueError as error:
            error_message = str(error)

        except Exception:
            error_message = (
                "The prediction could not be completed. "
                "Please check the entered values."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        error_message=error_message,
        form_values=form_values,
        metrics=metrics,
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)