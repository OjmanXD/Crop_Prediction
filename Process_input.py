from utils.pred_temp_hum import get_temp_hum
import pandas as pd
import pandas as pd
import joblib


# state = "ANDAMAN And NICOBAR ISLANDS"
# district = "NICOBAR"
# month = 1  # January

import pandas as pd

# Load the rainfall dataset
rainfall_data = pd.read_csv(r"data\rainfall_lat_long_fuzzy.csv")

# Function to get rainfall for a given state, district, and month
def get_input_data(state, district, month):
    # Filter the dataset for the given state and district
    filtered_data = rainfall_data[(rainfall_data['STATE_UT_NAME'] == state) & 
                                  (rainfall_data['DISTRICT'] == district)]
    
    # Extract the rainfall value for the given month
    rainfall = filtered_data.iloc[0, month + 2]  # Month columns start from index 2
    temperature , humidity = get_temp_hum("hyderabad")
    
    loaded_model = joblib.load("model\pkl_files\crop_yield_prediction_model.pkl")
    predicted_values = loaded_model.predict([[rainfall, temperature, humidity]])
    N, P, K, ph = predicted_values[0]
    
    # print(N, P, K, ph, rainfall, temperature, humidity)
    return list(N, P, K, temperature, humidity, ph, rainfall) 

# Example usage
# get_input_data(state, district, month)