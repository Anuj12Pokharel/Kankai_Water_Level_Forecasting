import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import time
import logging
import threading
from flask import Flask 
from flask import Flask, render_template,jsonify,request
import requests
import pandas as pd
import logging
import json


app = Flask(__name__)
# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Environment Variables
load_dotenv()
BASE_URL_RIVER1 = os.getenv("BASE_URL")
SOCKET_NAMESPACE = os.getenv("SOCKET_NAMESPACE")
MODEL_SAVE_DIR = "./"  # Directory to save models
MODEL_FILE_NAME = "lstm_water_level_forecast.h5"
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILE_NAME)
plot_path = "observed_data_plot_corrected.png"
BASE_API_URL = "https://alphaforecast.wscada.net"
DATA_ORIGIN = "forecast_kankai"  # From Data Origin
USERNAME = "anujpokharel2@gmail.com"
PASSWORD = "anujpokharel"

# As per the response you provided

# Update the endpoint URL for forecast data
FORECAST_ENDPOINT = f"{BASE_API_URL}/import"
OBSERVATION_ENDPOINT = f"{BASE_API_URL}/import"

if not BASE_URL_RIVER1 or not SOCKET_NAMESPACE:
    raise ValueError("Environment variables BASE_URL and SOCKET_NAMESPACE are required.")

BASE_URL_RIVER2_3 = "https://hydrology.gov.np"

# Load Pretrained Model or Initialize New One
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, compile=False)
    logging.info("Model loaded successfully.")
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")  # Ensure compilation
else:
    logging.error("Pre-trained model not found! Initializing new model.")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(50, 1)),
        Dropout(0.1),
        LSTM(50),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

# API Endpoints
##ENDPOINTS_RIVER1 = {
 ##   "observationDataSeries": lambda series_id, start_date, end_date: f"api/observation?series_id={series_id}&date_from={start_date}&date_to={end_date}",
##}
# Base configuration
session = requests.Session()
session.headers.update({'Content-Type': 'application/json'})

ENDPOINTS_RIVER1 = {
    "socket_response": f"api/socket/{SOCKET_NAMESPACE}/response",
    "station": "api/station/",
    "station_by_id": lambda station_id: f"api/station/{station_id}/",
    "station_data_series": lambda station_id: f"api/station/{station_id}/data-series/",
    "observation_data_series": lambda series_id, start_date: (
        f"api/observation?series_id={series_id}&date_from={start_date}"
    ),}

# Fetch Data Functions
def fetch_data_river1(series_id, start_date):

    try:
        current_time = datetime.utcnow().isoformat()
        ##print(f"Fetching data for series_id={series_id} from {start_date} to {current_time}...")
        url = f"{BASE_URL_RIVER1}/{ENDPOINTS_RIVER1['observation_data_series'](series_id, start_date)}&date_to={current_time}"
        print(f"Request URL: {url}")
        response = session.get(url)
        print(f"HTTP Status Code: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        print(f"Response Data: {data}")
        if not data:
            logging.warning(f"No data found for series_id {series_id} between {start_date} and {current_time}.")
        return data
    except requests.RequestException as e:
        logging.error(f"Network error while fetching data for River 1 (series_id {series_id}): {e}")
    except ValueError as e:
        logging.error(f"JSON decoding error for River 1 (series_id {series_id}): {e}")
    return []

def fetch_data_river2_3(series_id, start_date):

    try:
        current_time = datetime.utcnow().isoformat()
        url = f"{BASE_URL_RIVER2_3}/gss/api/observation?series_id={series_id}&date_from={start_date}&date_to={current_time}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching data for River 2/3 (series_id {series_id}): {e}")
        return None
# testing for github
# Data Processing Functions
def extract_and_format(data):
    rows = []
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    for item in data:
        if isinstance(item, dict):
            datetime = item.get("datetime") or item.get("timestamp")
            value = item.get("value") or item.get("measurement")
            if datetime and value is not None:
                rows.append({"Datetime": datetime, "Value": value})

    if not rows:
        logging.warning("No valid rows to process. Returning an empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if no rows

    df = pd.DataFrame(rows)

    if "Datetime" not in df.columns:
        logging.error("Missing 'Datetime' column in the processed data.")
        return pd.DataFrame()  # Return an empty DataFrame if column is missing

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df.dropna(subset=["Datetime"], inplace=True)
    df["Datetime"] = df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.set_index("Datetime")
    return df

def prepare_data(df, sequence_length):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i : i + sequence_length].values)
        y.append(df.iloc[i + sequence_length])
    return np.array(X), np.array(y)

# Save forecasted data to a CSV file
def save_forecasted_data(forecast_df, file_path="forecasted_data.csv", time_column="Forecast_Time"):
    try:
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path, parse_dates=[time_column])
            combined_data = pd.concat([existing_data, forecast_df], ignore_index=True)
            
            # Keep the latest forecast data by overwriting old data for the same timestamp
            combined_data = combined_data.sort_values(by=time_column).drop_duplicates(subset=[time_column], keep='last')
        else:
            combined_data = forecast_df

        combined_data.to_csv(file_path, index=False)
        logging.info(f"Forecasted data saved to {file_path}.")

    except Exception as e:
        logging.error(f"Error saving forecasted data: {e}")

# Load forecasted data from a CSV file
def load_forecasted_data(file_path="forecasted_data.csv"):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=["Forecast_Time"])
    else:
        logging.info(f"No historical forecasted data found at {file_path}.")
        return pd.DataFrame(columns=["Forecast_Time", "Forecasted_Water_Level"])

# Forecast function with data saving
def forecast(model, last_sequence, scaler, steps=30, observed_data=None, forecast_file_path="forecasted_data.csv"):
    predictions = []
    time_steps = []
    current_sequence = last_sequence
    last_forecast_value = None

    for step in range(steps):
        prediction = model.predict(current_sequence.reshape(1, current_sequence.shape[0], 1), verbose=0)
        prediction_value = prediction[0][0]

        if last_forecast_value is not None and np.isclose(prediction_value, last_forecast_value, atol=0.000001):
            logging.info("Forecast stopped due to repetitive value.")
            break

        predictions.append(prediction_value)
        forecast_time = pd.Timestamp.now() + pd.Timedelta(minutes=(step + 1) * 10)
        time_steps.append(forecast_time)
        current_sequence = np.append(current_sequence[1:], prediction_value)
        last_forecast_value = prediction_value

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    time_steps = pd.to_datetime(time_steps)
    global forecast_df
    forecast_df = pd.DataFrame({
    "Forecast_Time": time_steps,
    "Forecasted_Water_Level": predictions.flatten()})
    forecast_df['Forecast_Time'] = forecast_df['Forecast_Time'].dt.round('10min')
    forecast_df['Forecasted_Water_Level'] = forecast_df['Forecasted_Water_Level'].round(4)
    save_forecasted_data(forecast_df, forecast_file_path)
    print("\nForecasted Data:")
    print(forecast_df)
    if observed_data is not None:
        if isinstance(observed_data, pd.DataFrame):
            observed_data['Timestamp'] = observed_data['River3'].index
            observed_data = observed_data[['Timestamp'] + [col for col in observed_data.columns if col != 'Timestamp']]
            observed_data.to_csv("observed_data.csv", index=False)
            logging.info("Observed data saved directly to 'observed_data.csv'.")
        else:
            logging.warning("Observed data is not a DataFrame. Unable to save.")
    else:
        logging.warning("No observed data provided. Skipping save.")
   # forecast_df = pd.DataFrame({"Forecast_Time": time_steps, "Forecasted_Water_Level": predictions.flatten()})
    #save_forecasted_data(forecast_df, forecast_file_path)
    #print("\nForecasted Data:")
    #print(forecast_df)
    get_access_token()
    post_forecast_to_api()
    post_observation_data()
    plot_observed_vs_forecasted()



def plot_observed_vs_forecasted(observed_file="observed_data.csv", forecasted_file="forecasted_data.csv"):
    # Load the observed data
    observed_df = pd.read_csv(observed_file, parse_dates=["Timestamp"])
    # Handle duplicate timestamps by averaging the water levels
    observed_df = observed_df.groupby("Timestamp", as_index=False).mean()

    # Load the forecasted data
    forecasted_df = pd.read_csv(forecasted_file, parse_dates=["Forecast_Time"])

    # Plot the data
    plt.figure(figsize=(20, 6))
    plt.plot(observed_df["Timestamp"], observed_df["River3"], label="Observed Water Level", color='blue', linestyle='solid')
    plt.plot(forecasted_df["Forecast_Time"], forecasted_df["Forecasted_Water_Level"], label="Forecasted Water Level", color='red', linestyle='solid')

    plt.xlabel("Time")
    plt.ylabel("Water Level")
    plt.title("Observed vs Forecasted Water Level")
    plt.legend()
    plt.grid()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path)

def get_access_token():
    """Fetches the access token from the authentication API."""
    auth_url = f"{BASE_API_URL}/auth/login"
    credentials = {"username": USERNAME, "password": PASSWORD}

    try:
        response = requests.post(auth_url, json=credentials)
        response.raise_for_status()
        response_json = response.json()

        # Extract token
        token = response_json.get("access_token") or response_json.get("token")
        if token:
            logging.info("Access token retrieved successfully.")
            return token
        else:
            logging.error("Access token not found in response.")
            return None
    except requests.RequestException as e:
        logging.error(f"Authentication failed: {e}")
        return None

def post_forecast_to_api():
    """Posts forecast data to the API."""
    token = get_access_token()
    if not token:
        logging.error("Unable to get access token. Exiting.")
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        # Load forecast data from CSV
        forecast_df = pd.read_csv("forecasted_data.csv")

        # Ensure required columns are present
        if "Forecast_Time" not in forecast_df.columns or "Forecasted_Water_Level" not in forecast_df.columns:
            logging.error("CSV file missing required columns.")
            return

        # Convert datetime to ISO 8601 format
        forecast_df["Forecast_Time"] = pd.to_datetime(forecast_df["Forecast_Time"]).dt.strftime('%Y-%m-%dT%H:%M:%S')
        print(forecast_df["Forecast_Time"].head())

        # Prepare forecast payload based on the API's expected format
        forecast_payload = [
            {
                "origin_code": "forecast_kankai",
                "parameter_code": "ML2",  # Update as needed
                "value": f"{float(row['Forecasted_Water_Level']):.3f}",
                "time": row["Forecast_Time"]
            }
            for _, row in forecast_df.iterrows()
        ]

        logging.info(f"Posting forecast data to {FORECAST_ENDPOINT}")
        logging.debug(f"Forecast Payload: {json.dumps(forecast_payload, indent=2)}")

        # Send POST request
        response = requests.post(FORECAST_ENDPOINT, json=forecast_payload, headers=headers)
        response.raise_for_status()

        if response.status_code in [200, 201]:
            logging.info("Forecast data successfully posted.")
        else:
            logging.error(f"Failed to post forecast data: {response.status_code}, {response.text}")

    except FileNotFoundError:
        logging.error("forecasted_data.csv file not found.")
    except Exception as e:
        logging.error(f"Error preparing forecast data for API: {e}")

def post_observation_data():
    """Posts observation data to the API."""
    token = get_access_token()
    if not token:
        logging.error("Unable to get access token. Exiting.")
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        # Load observed data from CSV
        observed_df = pd.read_csv("observed_data.csv")

        # Ensure required columns are present
        if "Timestamp" not in observed_df.columns or "River3" not in observed_df.columns:
            logging.error("CSV file missing required columns.")
            return

        # Convert datetime to ISO 8601 format
        observed_df["Timestamp"] = pd.to_datetime(observed_df["Timestamp"]).dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Prepare observation payload based on the API's expected format
        observations_payload = [
            {
                "origin_code": "Observation",
                "parameter_code": "ML1",  # Example of parameter, update as needed
                "value": f"{float(row['River3']):.3f}",                        #float(row["River3"]),
                "time": row["Timestamp"]                      # row["Timestamp"]
            }
            for _, row in observed_df.iterrows()
        ]

        logging.info(f"Posting observation data to {OBSERVATION_ENDPOINT}")
        logging.debug(f"Observation Payload: {json.dumps(observations_payload, indent=2)}")

        # Send POST request
        response = requests.post(OBSERVATION_ENDPOINT, json=observations_payload, headers=headers)
        response.raise_for_status()

        if response.status_code in [200, 201]:
            logging.info("Observation data successfully posted.")
        else:
            logging.error(f"Failed to post observation data: {response.status_code}, {response.text}")

    except FileNotFoundError:
        logging.error("observed_data.csv file not found.")
    except Exception as e:
        logging.error(f"Error preparing observation data for API: {e}")
# Call function to send observations
# Continuous Training and Forecasting
def continuous_train_and_forecast(series_ids_rivers, start_date, sequence_length=1, interval=660, forecast_steps=30):
    scalers = {}
    while True:
        river_data_frames = []

        for river, series_id in series_ids_rivers.items():
            if river == "River1":
                data = fetch_data_river1(series_id, start_date)
            else:
                data = fetch_data_river2_3(series_id, start_date)

            if data:
                df = extract_and_format(data)
                df = df.rename(columns={"Value": river})
                river_data_frames.append(df)

        if river_data_frames:
            global merged_data
            merged_data = pd.concat(river_data_frames, axis=1).interpolate()
            kankai_river = merged_data[["River3"]]
            print(kankai_river.head())

            for column in merged_data.columns:
                if column not in scalers:
                    scalers[column] = MinMaxScaler(feature_range=(0, 1))
                merged_data[column] = scalers[column].fit_transform(merged_data[[column]])

            X, y = prepare_data(merged_data.iloc[:, -1], sequence_length)

            if X.size > 0:
                X = X.reshape((X.shape[0], X.shape[1], 1))
                model.fit(X, y, epochs=5, batch_size=32, verbose=1)
                model.save(MODEL_PATH)
                logging.info(f"Model updated and saved at {MODEL_PATH}.")

                last_sequence = merged_data.iloc[-sequence_length:, -1].values.reshape(-1, 1)
                forecasts = forecast(model, last_sequence, scalers[merged_data.columns[-1]], steps=forecast_steps, observed_data=kankai_river)
                
                logging.info(f"Forecast for next {forecast_steps} steps: {forecasts}")

        logging.info(f"Waiting for the next cycle ({interval} seconds)...")
        time.sleep(interval)



def start_forecasting():
    series_ids_rivers = {"River1": 3114, "River2": 19643, "River3": 3635}
    start_date = "2024-12-14T00:00:00"
    continuous_train_and_forecast(series_ids_rivers, start_date)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    threading.Thread(target=start_forecasting, daemon=True).start()
    app.run(debug=True)
