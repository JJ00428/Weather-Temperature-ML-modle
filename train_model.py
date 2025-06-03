import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
print("Loading dataset...")
weather_data = pd.read_parquet('daily_weather.parquet')

print("Missing values:")
print(weather_data.isna().sum())

# Derive weather condition
print("Deriving weather condition...")
weather_data['weather'] = weather_data['precipitation_mm'].apply(lambda x: 'Rainy' if x > 0.1 else 'Clear')

# Preprocess date and features
print("Preprocessing date...")
weather_data['date'] = pd.to_datetime(weather_data['date'])
weather_data['day_of_year'] = weather_data['date'].dt.dayofyear
weather_data['month'] = weather_data['date'].dt.month
weather_data['year'] = weather_data['date'].dt.year
weather_data['hour'] = 12  # Fixed for 12 PM
weather_data['prev_temp'] = weather_data.groupby('city_name')['avg_temp_c'].shift(1)
weather_data = weather_data.dropna()

# Compute monthly averages for app.py
print("Computing monthly averages...")
monthly_averages = weather_data.groupby(['city_name', 'month'])[['humidity', 'pressure', 'avg_temp_c']].mean().reset_index()
monthly_averages.to_csv('monthly_averages.csv', index=False)

# Create city_coords
city_coords = weather_data.groupby('city_name').first()[['latitude', 'longitude']].to_dict('index')
city_coords = {city: (data['latitude'], data['longitude']) for city, data in city_coords.items() if pd.notna(data['latitude']) and pd.notna(data['longitude'])}

# Encode city and weather
print("Encoding city names and weather...")
label_encoder_city = LabelEncoder()
weather_data['city_encoded'] = label_encoder_city.fit_transform(weather_data['city_name'])
label_encoder_weather = LabelEncoder()
weather_data['weather_encoded'] = label_encoder_weather.fit_transform(weather_data['weather'])

# Define features and targets
print("Defining features and targets...")
X = weather_data[['day_of_year', 'city_encoded', 'month', 'year', 'humidity', 'pressure', 'prev_temp']]
y_temperature = weather_data['avg_temp_c']
y_weather = weather_data['weather_encoded']

# Split data
print("Splitting data...")
X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temperature, test_size=0.2, random_state=42)
X_train_weather, X_test_weather, y_weather_train, y_weather_test = train_test_split(X, y_weather, test_size=0.2, random_state=42)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest models
print("Training RandomForest models...")
temp_model = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
temp_model.fit(X_train_scaled, y_temp_train)
weather_model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
weather_model.fit(X_train_scaled, y_weather_train)

# Evaluate models
print("Evaluating models...")
temp_score = temp_model.score(X_test_scaled, y_temp_test)
weather_score = weather_model.score(X_test_scaled, y_weather_test)
y_temp_pred = temp_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_temp_test, y_temp_pred))
print(f"Temperature R^2: {temp_score:.2f}")
print(f"Temperature RMSE: {rmse:.2f}°C")
print(f"Weather Accuracy: {weather_score:.2f}")

# # Feature importance
# print("Feature importance (temperature):")
# feature_names = ['day_of_year', 'city_encoded', 'month', 'year', 'humidity', 'pressure', 'prev_temp']
# for name, importance in zip(feature_names, temp_model.feature_importances_):
#     print(f"{name}: {importance:.4f}")

# Save models and encoders
print("Saving models and encoders...")
joblib.dump(temp_model, 'temp_model.pkl')
joblib.dump(weather_model, 'weather_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder_city, 'label_encoder_city.pkl')
joblib.dump(label_encoder_weather, 'label_encoder_weather.pkl')
joblib.dump(city_coords, 'city_coords.pkl')
print("Models and encoders saved!")