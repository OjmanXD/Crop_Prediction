import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
data = pd.read_csv(r"data\Crop_recommendation.csv")

# Separate features and target variables
X = data[['rainfall', 'temperature', 'humidity']]
y = data[['N', 'P', 'K', 'ph']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Example usage to predict N, P, K, and pH for given rainfall, temperature, and humidity
rain = 200  # mm
temp = 25  # Celsius
hum = 75  # Percentage

predicted_values = rf_regressor.predict([[rain, temp, hum]])

# Export the trained model
joblib.dump(rf_regressor, "model\pkl_files\crop_yield_prediction_model.pkl")