from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model (no scaler needed now)
model = pickle.load(open('model.pkl', 'rb'))

# Features must match training order
features = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]

# AQI category logic
def categorize_aqi(aqi):
    if aqi <= 50:
        return "Very Good"
    elif aqi <= 100:
        return "Good"
    elif aqi <= 150:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input
        data = []
        for f in features:
            value = request.form.get(f)
            if value is None or value.strip() == "":
                raise ValueError(f"Missing value for {f}")
            data.append(float(value))

        # Convert to DataFrame (not scaled)
        input_df = pd.DataFrame([data], columns=features)

        # Predict using unscaled model
        prediction = model.predict(input_df)[0]
        category = categorize_aqi(prediction)

        return render_template('index.html', result=round(prediction, 2), category=category)

    except ValueError as ve:
        print("ValueError:", ve)
        return render_template('index.html', result="Invalid input! Enter numbers only.", category="")
    except Exception as e:
        print("Error:", e)
        return render_template('index.html', result=f"Error: {str(e)}", category="")

if __name__ == '__main__':
    app.run(debug=True)
