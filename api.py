import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

import tensorflow as tf
import joblib
import os
import time
import requests
import json
import threading

from flask import Flask, render_template, url_for
from flask_restful import Resource, Api, reqparse
from flask_apscheduler import APScheduler
import mysql.connector
import mysql.connector as mariadb

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest

import tensorflow as tf
from tensorflow import keras
import os
import json

from datetime import datetime
from datetime import date, timedelta

import mysql.connector as mariadb

app = Flask(__name__)
api = Api(app)

scheduler = APScheduler()
scheduler.api_enabled = True
scheduler.init_app(app)
scheduler.start()

parser = reqparse.RequestParser()
parser.add_argument('date')
parser.add_argument('sitecode')
parser.add_argument('model')
parser.add_argument('modeltime')

# Database connection parameters
db_config = {
    'host': 'ens-datacenter.kr',
    'user': 'kookmin',
    'password': 'kookmin',
    'database': 'ens_datacenter'
}

# Function to fetch power data
def fetch_power_data(conn, c_code, devno, start_date, end_date):
    query = f"""
        SELECT D_date, F_totpower
        FROM tbl_pv_dat
        WHERE C_scode = '{c_code}'
        AND I_devno = {devno}
        AND D_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY DATE(D_date), HOUR(D_date)
    """
    df = pd.read_sql(query, conn)
    df['D_date'] = pd.to_datetime(df['D_date'])
    return df

# Function to fetch weather data
def fetch_weather_data(conn, table_name, c_code, start_date, end_date):
    query = f"""
        SELECT D_date, C_scode, F_temp, F_wind_speed, F_humidity, F_daylight, F_solar_radiation, F_snowfall, F_total_cloud_cover, 
               C_cloud_pattern, C_visibility, F_ground_state, F_ground_temp, F_precipitation, F_wind_direction
        FROM {table_name}
        WHERE C_scode = {c_code} 
        AND D_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY DATE(D_date), HOUR(D_date)
    """
    df = pd.read_sql(query, conn)
    df['D_date'] = pd.to_datetime(df['D_date'])
    return df

# Function to replace outliers with interpolation
def replace_outlier(df_in, col_name, scale):
    outliers_fraction = float(scale)
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(df_in[col_name].values.reshape(-1, 1))
    data = pd.DataFrame(np_scaled)
    
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(data)
    df_in['anomaly'] = model.predict(data)
    df_in.loc[df_in['anomaly'] == -1, col_name] = np.nan  # Replace anomalies with NaN
    df_in[col_name] = df_in[col_name].interpolate(method='nearest').ffill().bfill()
    df_in.drop(columns='anomaly', inplace=True)
    return df_in

# Function to align data with a complete datetime range
def align_with_datetime_range(df, start_date, end_date, freq='H'):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    df_expected = pd.DataFrame(date_range, columns=['datetime'])
    df['datetime'] = df['D_date'].dt.strftime('%Y-%m-%d %H:00:00')
    df_expected['datetime'] = df_expected['datetime'].dt.strftime('%Y-%m-%d %H:00:00')
    df_combined = pd.merge(df_expected, df, on='datetime', how='left')
    df_combined['D_date'] = pd.to_datetime(df_combined['datetime'])
    df_combined.drop(columns=['datetime'], inplace=True)
    return df_combined

# Main function to combine power and weather data
def data_combined(power_sites, start_date, end_date, weather_sites, weather_tables):
    try:
        conn = mysql.connector.connect(**db_config)
        print("Connected to MariaDB!")

        # Initialize lists for final data
        final_power_dataframes = []
        final_weather_dataframes = []

        # Process power data
        for devno, sites in power_sites.items():
            for c_code in sites:
                df_power_raw = fetch_power_data(conn, c_code, devno, start_date, end_date)
                df_power_aligned = align_with_datetime_range(df_power_raw, start_date, end_date)
                df_power_aligned['F_totpower'] = df_power_aligned['F_totpower'].ffill().bfill().interpolate(method='nearest')

                # Calculate PowerAggregate
                np_allPow = df_power_aligned['F_totpower'].values.reshape(-1, 1)
                np_pow_aggregate = np.abs(np.diff(np_allPow, axis=0, prepend=np_allPow[0:1]))
                df_power_aligned['PowerAggregate'] = pd.Series(np_pow_aggregate.flatten())
                df_power_aligned = replace_outlier(df_power_aligned, 'PowerAggregate', 0.01)
                df_power_aligned.loc[df_power_aligned['PowerAggregate'] < 0, 'PowerAggregate'] = 0
                
                final_power_dataframes.append(df_power_aligned)

        # Process weather data
        for c_code in weather_sites:
            weather_frames = []
            for table in weather_tables:
                df_weather = fetch_weather_data(conn, table, c_code, start_date, end_date)
                if not df_weather.empty:
                    weather_frames.append(df_weather)
            if weather_frames:
                df_weather_combined = pd.concat(weather_frames, ignore_index=True)
                df_weather_aligned = align_with_datetime_range(df_weather_combined, start_date, end_date)
                for column in ['F_temp', 'F_wind_speed', 'F_humidity', 'F_daylight', 'F_solar_radiation', 'F_snowfall', 
                               'F_total_cloud_cover', 'F_precipitation', 'F_wind_direction', 'F_ground_state', 'F_ground_temp']:
                    df_weather_aligned[column] = df_weather_aligned[column].ffill().bfill().interpolate(method='linear')
                for column in ['C_scode', 'C_cloud_pattern', 'C_visibility']:
                    df_weather_aligned[column] = df_weather_aligned[column].ffill().bfill()

                # Add a new column with transformed F_total_cloud_cover values
                df_weather_aligned['F_total_cloud_cover_t'] = df_weather_aligned['F_total_cloud_cover'].apply(
                    lambda x: 1 if 0 <= x <= 5 else (3 if 6 <= x <= 8 else (4 if 9 <= x <= 10 else np.nan))
                )

                final_weather_dataframes.append(df_weather_aligned)

        # Combine all processed data
        df_power_final = pd.concat(final_power_dataframes, ignore_index=True) if final_power_dataframes else pd.DataFrame()
        df_weather_final = pd.concat(final_weather_dataframes, ignore_index=True) if final_weather_dataframes else pd.DataFrame()

        # Merge power and weather data
        if not df_power_final.empty and not df_weather_final.empty:
            df_combined = pd.merge(df_power_final, df_weather_final, on='D_date', how='inner')
            
            # Handle missing values
            df_combined = df_combined.fillna(0)
            
            output_file_combined = "finalcombined_tsetdata_10.csv"
            df_combined.to_csv(output_file_combined, index=False)
            print(f"Shape of the combined DataFrame: {df_combined.shape}")
            
            plt.figure(figsize=(20, 10))  # Set the figure size first
            sns.heatmap(df_combined.drop(columns=['D_date']).select_dtypes(include=[float, int]).corr(), annot=True, vmin=0, vmax=1, cmap="coolwarm")
            
            # Select required columns
            selected_columns = [
                'F_temp', 'F_wind_speed', 'F_humidity', 'F_total_cloud_cover_t', 'F_precipitation',  'F_wind_direction', 'PowerAggregate', 
            ]
            df_final = df_combined[selected_columns]
            
            # Save the final dataset
            output_file = "final_testdata_10.csv"
            df_final.to_csv(output_file, index=False)
            print(f"Final dataset saved to {output_file}.")
            print(f"Shape of the combined DataFrame: {df_final.shape}")
            print("Preview of the final dataset:")
            print(df_final.head())
        else:
            print("One or both datasets are empty.")

    except mysql.connector.Error as e:
        print(f"Error connecting to MariaDB: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()
            print("Connection closed.")

# 1. Sequence Creation Function
def create_sequences(data, input_sequence_length, output_sequence_length):
    sequences, targets = [], []
    for i in range(len(data) - input_sequence_length - output_sequence_length + 1):
        seq = data[i:i + input_sequence_length, 0:6]  # Input features (first 6 columns)
        target = data[i + input_sequence_length:i + input_sequence_length + output_sequence_length, 6:7]  # Target (column 7)
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# 2. Preprocess New Data
def preprocess_new_data(new_data, scaler, input_sequence_length, output_sequence_length):
    # Scale the data
    data_scaled = scaler.transform(new_data)
    
    # Create sequences
    sequences, _ = create_sequences(data_scaled, input_sequence_length, output_sequence_length)
    return sequences

# 3. Load Model and Metadata
def load_model_and_metadata(model_type, models_dir="saved_models_new"):
    metadata_path = os.path.join(models_dir, f"{model_type}_metadata.pkl")
    metadata = joblib.load(metadata_path)
    scaler = metadata["scaler"]
    model_path = metadata["model_path"]
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print(f"{model_type} model loaded successfully.")
    
    return model, scaler

# 4. Prediction with All Models
def predict_with_all_models(new_data, models_dir="saved_models_new", input_sequence_length=24*7, output_sequence_length=1):
    model_types = ['BiLSTM', 'BiLSTM_SingleDense', 'BiLSTM_MultiDense', 'LSTM', 'ConvLSTM', 'RNN']
    predictions_results = {}

    for model_type in model_types:
        print(f"\nProcessing {model_type}...")
        
        # Load model and metadata
        model, scaler = load_model_and_metadata(model_type, models_dir)
        
        # Preprocess new data
        new_data_sequences = preprocess_new_data(new_data, scaler, input_sequence_length, output_sequence_length)
        
        # Generate 1-hour ahead prediction
        one_hour_pred = model.predict(new_data_sequences)
        
        # Inverse transform the 1-hour ahead prediction
        # We only want to inverse transform the predicted target (last column)
        one_hour_pred_inv = scaler.inverse_transform(np.concatenate([np.zeros((one_hour_pred.shape[0], 6)), one_hour_pred], axis=1))[:, -1]
        
        # Save the 1-hour prediction
        predictions_results[model_type] = {'1_hour_ahead': one_hour_pred_inv}
        print(f"1-hour ahead predictions completed for {model_type}.")

        # For 24-hour prediction: We will loop and predict 1 hour at a time
        next_24_hours_pred = []
        last_input_sequence = new_data_sequences[-1, :, :]  # Get the last sequence

        for _ in range(24):
            one_hour_pred = model.predict(last_input_sequence.reshape(1, input_sequence_length, 6))  # Predict 1 hour ahead
            one_hour_pred_inv = scaler.inverse_transform(np.concatenate([np.zeros((one_hour_pred.shape[0], 6)), one_hour_pred], axis=1))[:, -1]
            next_24_hours_pred.append(one_hour_pred_inv[0])
            
            # Update the sequence by adding the predicted value and removing the oldest value
            last_input_sequence = np.roll(last_input_sequence, shift=-1, axis=0)  # Shift sequence to the left
            last_input_sequence[-1, -1] = one_hour_pred[0, 0]  # Add the predicted value to the last position
        
        # Save the 24-hour prediction
        predictions_results[model_type]['24_hours_ahead'] = np.array(next_24_hours_pred)
        print(f"24-hour ahead predictions completed for {model_type}.")
    
    return predictions_results

# 5. Print Predictions
def print_predictions(predictions_results):
    for model_type, predictions in predictions_results.items():
        print(f"\n{model_type}:")
        print(f"1-hour ahead predictions: {predictions['1_hour_ahead'][:10]}...")  # Print first 10 for preview
        print(f"24-hour ahead predictions: {predictions['24_hours_ahead'][:10]}...")  # Print first 10 for preview

# 6. Visualize Predictions
def plot_predictions(predictions_results):
    for model_type, predictions in predictions_results.items():
        plt.figure(figsize=(12, 6))
        plt.plot(predictions['1_hour_ahead'], label=f'{model_type} 1-hour Predictions', color='red', alpha=0.7)
        plt.plot(predictions['24_hours_ahead'], label=f'{model_type} 24-hour Predictions', color='blue', alpha=0.7)
        plt.title(f"{model_type} Predictions on New Data")
        plt.xlabel("Time Steps")
        plt.ylabel("Predicted Value")
        plt.legend()
        plt.grid(True)

# Create a folder to save predictions
def save_predictions(predictions_results, save_dir="predictions"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for model_type, predictions in predictions_results.items():
        file_path = os.path.join(save_dir, f"{model_type}_24_hour_predictions.csv")
        # Save 24-hour predictions as CSV
        pd.DataFrame(predictions['24_hours_ahead'], columns=["24_hour_predictions"]).to_csv(file_path, index=False)
        print(f"24-hour predictions saved for {model_type} at {file_path}")  

# Create a folder to save predictions
def save_predictions_database(predictions_results, power_sites, end_date, model_times):
    for model_type, predictions in predictions_results.items():
        # Take Sitecode Value from Power_sites
        site_code = list(power_sites.values())[0][0]
        
        # Establish database connection
        db_conn = mariadb.connect(host="203.246.113.248", user="abc", password="123", database="PVPowerGeneration", port=12360)
        db_cursor = db_conn.cursor()
        print("Connected to MariaDB!")

        datenext = end_date
        print('Start Predicting!', model_type)
        # Insert prediction result into database
        for m in range(24):
            datenextstr = datenext.strftime('%Y-%m-%d')
            sqlKMA = f"INSERT IGNORE INTO  predictionresult(datename, modeltime, sitename, methodname, predictionvalue)\
                                VALUES ('{datenextstr}', {'12'}, {site_code}, '{model_type}', {predictions['24_hours_ahead'][m]})"
            db_cursor.execute(sqlKMA) 
            db_conn.commit()
            datenext += timedelta(hours=1)
                
        db_cursor.close()
        db_conn.close()
            
        # Save 24-hour predictions as CSV
        print("Predictions successfully saved to MariaDB!") 

# # 7. Example Usage
# if __name__ == "__main__":
def mainn():  
    # Global Variable for Input
    power_sites = {2: ["717800010"]} # INPUT SITECODE
    # Set Date
    end_date = datetime.now().date() # datetime.strptime('2025-02-07', '%Y-%m-%d') # 2025-02-04' # INPUT END DATE
    start_date = (end_date - timedelta(days=7)).strftime('%Y-%m-%d')  # '2025-01-28' # INPUT START DATE 
    # Set modeltimes
    model_times = ['12']
    
    # INPUT FOR WEATHER
    weather_sites = ["717805001"]
    weather_tables = [
        "tbl_kma_weather"
    ]
    
    # Call main function for the combined data
    data_combined(power_sites, start_date, end_date, weather_sites, weather_tables)

    # Load the CSV file into a DataFrame
    new_data = pd.read_csv("final_testdata_10.csv")
    print(new_data.shape)
    
    # Define input/output sequence lengths
    input_sequence_length = 24 * 7  # 7 days worth of hourly data (adjust as needed)
    output_sequence_length = 1      # Predicting the next hour (adjust as needed)

    # Load the CSV file into a DataFrame
    new_data = pd.read_csv("final_testdata_10.csv")

    # Predict with all models
    predictions_results = predict_with_all_models(new_data, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length)

    # Print the predictions (first 10 for each model as a preview)
    print_predictions(predictions_results)

    # Visualize predictions
    plot_predictions(predictions_results)

    # Save the 24-hour predictions for each model
    save_predictions(predictions_results, save_dir="model_predictions")
    save_predictions_database(predictions_results, power_sites, end_date, model_times)
    plot_predictions(predictions_results)

# Scheduling the time to set predict time in 00:30 AM
@scheduler.task('cron', id='prediction_10', minute='25', hour='0')
def prediction_at_12():
    mainn()

# Taking data from database to be shown in URLs
class getPrediction(Resource):
    def post(self):
        args = parser.parse_args()
        print(args['date'],args['sitecode'],args['model'],{args['modeltime']}, time.time())

        db_conn = mariadb.connect(host="203.246.113.248", user="abc", password="123", database="PVPowerGeneration", port=12360)
        db_cursor = db_conn.cursor()
        db_command = f"SELECT predictionValue FROM predictionresult WHERE sitename = {int(args['sitecode'])} AND \
            DATE(datename) = {int(args['date'])} AND methodname = '{args['model']}' AND modeltime = '{args['modeltime']}' ORDER BY `datename` ASC"
        # print(db_command)
        db_cursor.execute(db_command)
        response = db_cursor.fetchall()
        db_cursor.close()
        db_conn.close()

        conn = mysql.connector.connect(host="ens-datacenter.kr",port="3306",user="kookmin",password="kookmin",database="ens_datacenter")
        cur = conn.cursor()
        cur.execute(f"SELECT F_tot FROM tbl_pv_power WHERE date(D_date) = {int(args['date'])} AND C_scode = {int(args['sitecode'])} GROUP BY DATE(D_date),HOUR(D_date)")
        response2 = cur.fetchall()
        print("Raw Power : ", response2)
        cur.close()
        conn.close()

        sumPower = 0
        listData = []
        for qq in range(5,20):
            if not response:
                sumPower += np.array(float(0))
                kk = np.array(float(0))
            else:
                sumPower += float(response[qq][0])
                kk = np.array(float(response[qq][0]))
            listData.append(np.around(kk, 2))

        sumPower = np.around(sumPower,2)

        data2 = []
        sumPower2 = 0
        for qq in range(5,20):
            if qq < len(response2):
                sumPower2 += float(abs(response2[qq][0] - response2[qq-1][0]))
                kk = np.array(float(abs(response2[qq][0] - response2[qq-1][0])))
            else :
                sumPower2 += np.array(float(0))
                kk = np.array(float(0))
            data2.append(int(np.around(kk, 2)))
        
        errRate = str(round((abs(sumPower-sumPower2)/sumPower2*100),2))+"%"

        pred = {"type": "예측",
                        "hr5": listData[0],
                        "hr6": listData[1],
                        "hr7": listData[2],
                        "hr8": listData[3],
                        "hr9": listData[4],
                        "hr10": listData[5],
                        "hr11": listData[6],
                        "hr12": listData[7],
                        "hr13": listData[8],
                        "hr14": listData[9],
                        "hr15": listData[10],
                        "hr16": listData[11],
                        "hr17": listData[12],
                        "hr18": listData[13],
                        "hr19": listData[14],
                        "sum": sumPower,
                        "erRate": errRate}
        true = {"type": "진실",
                        "hr5": data2[0],
                        "hr6": data2[1],
                        "hr7": data2[2],
                        "hr8": data2[3],
                        "hr9": data2[4],
                        "hr10": data2[5],
                        "hr11": data2[6],
                        "hr12": data2[7],
                        "hr13": data2[8],
                        "hr14": data2[9],
                        "hr15": data2[10],
                        "hr16": data2[11],
                        "hr17": data2[12],
                        "hr18": data2[13],
                        "hr19": data2[14],
                        "sum": sumPower2,
                        "erRate": "-"}

        json_response =  {'pred':pred, 'true':true}

        print(json_response)

        json_response = json.dumps(json_response)
        
        return {'data': json_response}

api.add_resource(getPrediction, '/data/api/powerpred')
        
@app.route('/')
def index():
    return render_template('powerPlot.html')
    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5008, use_reloader=False)