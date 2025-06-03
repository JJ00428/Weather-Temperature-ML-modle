from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load models and encoders
temp_model = joblib.load('temp_model.pkl')
weather_model = joblib.load('weather_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_city = joblib.load('label_encoder_city.pkl')
label_encoder_weather = joblib.load('label_encoder_weather.pkl')
monthly_averages = pd.read_csv('monthly_averages.csv')
city_coords = joblib.load('city_coords.pkl')

@app.route('/predict', methods=['POST'])
def predict_weather():
    # Get input data
    data = request.get_json()
    date_str = data.get('date')  # YYYY-MM-DD
    city = data.get('city')
    hour = 12  # Fixed at 12 PM

    # Validate inputs
    if city not in city_coords:
        return jsonify({'error': f"City '{city}' not found in the dataset."}), 400
    
    try:
        date = pd.to_datetime(date_str)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid date (use YYYY-MM-DD).'}), 400

    # Encode city
    try:
        city_encoded = label_encoder_city.transform([city])[0]
    except ValueError:
        return jsonify({'error': f"City '{city}' not found in the encoder."}), 400

    # Get monthly averages
    month = date.month
    avg_data = monthly_averages[(monthly_averages['city_name'] == city) & (monthly_averages['month'] == month)]
    if avg_data.empty:
        return jsonify({'error': f"No average data for {city} in month {month}."}), 400
    humidity = avg_data['humidity'].iloc[0]
    pressure = avg_data['pressure'].iloc[0]
    prev_temp = avg_data['avg_temp_c'].iloc[0]

    # Prepare features
    day_of_year = date.dayofyear
    month = date.month
    year = date.year
    features = np.array([[day_of_year, city_encoded, month, year, humidity, pressure, prev_temp]])
    scaled_features = scaler.transform(features)

    # Predict
    predicted_temperature = temp_model.predict(scaled_features)[0]
    predicted_weather_encoded = weather_model.predict(scaled_features)[0]
    predicted_weather = label_encoder_weather.inverse_transform([predicted_weather_encoded])[0]

    # Adjust for climate trend (2025 vs. 2024)
    temp_trend = 0.15  # °C/year
    predicted_temperature += temp_trend * (date.year - 2024)

    print("date_str")
    print(date_str)
    print("temp")
    print(float(predicted_temperature))
    print("weather")
    print(predicted_weather)


    return jsonify({
        'temperature': float(predicted_temperature),
        'weather': predicted_weather,
        'city': city,
        'date': date_str,
        'hour': hour
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)