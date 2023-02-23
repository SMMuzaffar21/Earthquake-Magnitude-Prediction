import numpy as np
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Load data from CSV file
data = pd.read_csv("Earthquakes.csv")

# Extract useful features
timestamp = []
for d, t in zip(data['Date'], data['Time']):
    try:
        ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
        timestamp.append(time.mktime(ts.timetuple()))
    except ValueError:
        timestamp.append(np.nan)
data['Timestamp'] = timestamp
data.dropna(inplace=True)
data = data[['Timestamp', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

# Plot earthquake locations
plt.figure(figsize=(8, 6))
plt.scatter(data['Longitude'], data['Latitude'], c=data['Magnitude'], cmap='coolwarm')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Earthquake Locations')
plt.colorbar()
plt.show()

# Plot earthquake magnitudes over time
plt.figure(figsize=(8, 6))
plt.plot(data['Timestamp'], data['Magnitude'], 'o', markersize=2)
plt.xlabel('Timestamp')
plt.ylabel('Magnitude')
plt.title('Earthquake Magnitudes Over Time')
plt.show()

# Split data into training and testing sets
X = data[['Timestamp', 'Latitude', 'Longitude', 'Depth']]
y = data[['Magnitude']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the neural network model using keras 
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Evaluate the model for test data
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score)

# Predict earthquakes for new data
new_data = pd.DataFrame({
    'Timestamp': [time.mktime(datetime.datetime(2022, 2, 17, 10, 0, 0).timetuple())],
    'Latitude': [34.05],
    'Longitude': [-118.25],
    'Depth': [10.0]
})
prediction = model.predict(new_data)
print("Predicted magnitude:", prediction[0][0])

# Set a threshold for predicted earthquake magnitude
threshold = 6.0

# Predict earthquakes for new data
new_data = pd.DataFrame({
    'Timestamp': [time.mktime(datetime.datetime(2022, 2, 17, 10, 0, 0).timetuple())],
    'Latitude': [34.05],
    'Longitude': [-118.25],
    'Depth': [10.0]
})
prediction = model.predict(new_data)

# Check if predicted magnitude exceeds the threshold
if prediction[0][0] > threshold:
    print("Alert: An earthquake with magnitude", prediction[0][0], "is predicted!")


else:
    print("No earthquake is predicted.")
