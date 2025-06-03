import pandas as pd

# Load CSV
print("Loading GlobalWeatherRepository.csv...")
weather_data = pd.read_csv('GlobalWeatherRepository.csv')

# Print columns for verification
print("Dataset columns:", weather_data.columns.tolist())

# Rename columns
weather_data = weather_data.rename(columns={
    'last_updated': 'date',
    'location_name': 'city_name',
    'temperature_celsius': 'avg_temp_c',
    'humidity': 'humidity',
    'precip_mm': 'precipitation_mm',
    'pressure_mb': 'pressure',
    'latitude': 'latitude',
    'longitude': 'longitude'
})

# Select columns
columns = ['date', 'city_name', 'avg_temp_c', 'humidity', 'precipitation_mm', 'pressure', 'latitude', 'longitude']
weather_data = weather_data[columns]

# Save to Parquet
print("Saving to daily_weather.parquet...")
weather_data.to_parquet('daily_weather.parquet', index=False)
print("Conversion complete!")