import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the CSV data
csv_file = 'candlestickchart/intraday_5min_IBM.csv'
data = pd.read_csv(csv_file)

# Convert 'timestamp' to datetime and set as index
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Prepare data for prediction (selecting 'close' price)
data_for_prediction = data[['open', 'high', 'low', 'close', 'volume']]

# ================== PART 1: CSV-Based Prediction (Linear Regression) ==================

# 1. Normalize the 'close' price data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_for_prediction['close'].values.reshape(-1, 1))

# 2. Create training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Prepare data for supervised learning
def prepare_data(data, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i + window_size, 0])
        y.append(data[i + window_size, 0])
    return np.array(x), np.array(y)

window_size = 10
x_train, y_train = prepare_data(train_data, window_size)
x_test, y_test = prepare_data(test_data, window_size)

# Reshape data for Linear Regression
x_train_lr = np.reshape(x_train, (x_train.shape[0], window_size))
x_test_lr = np.reshape(x_test, (x_test.shape[0], window_size))

# 3. Train a Linear Regression model
model_lr = LinearRegression()
model_lr.fit(x_train_lr, y_train)

# 4. Predict the test set
predicted_stock_price = model_lr.predict(x_test_lr)

# 5. Inverse scale the predicted values to get actual stock prices
predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 6. Plot the actual vs predicted prices
plt.figure(figsize=(10, 5))
plt.plot(data.index[-len(y_test):], y_test, label="Actual Price", color='blue')
plt.plot(data.index[-len(predicted_stock_price):], predicted_stock_price, label="Predicted Price", color='red')
plt.title("Stock Price Prediction (CSV-Based)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# ================== PART 2: Candlestick Chart Generation (for CNN) ==================

# Create a directory to save candlestick chart images
chart_image_dir = "chart_images"
os.makedirs(chart_image_dir, exist_ok=True)

# 1. Generate candlestick chart images
data_for_plot = data[['open', 'high', 'low', 'close', 'volume']]
data_for_plot.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Split the data into multiple time windows for chart generation
chart_window = 50  # 50 periods per chart
for i in range(0, len(data_for_plot) - chart_window, chart_window):
    chart_data = data_for_plot.iloc[i:i + chart_window]
    chart_filename = f"{chart_image_dir}/chart_{i}.png"
    mpf.plot(chart_data, type='candle', volume=True, style='yahoo', savefig=chart_filename)

# ================== PART 3: CNN-Based Prediction Using Candlestick Charts ==================

# Prepare data for CNN training by loading chart images and assigning labels (up, down, neutral)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    return img

# Assume we have a list of image files and their respective labels
image_files = os.listdir(chart_image_dir)
labels = []  # Replace with your actual labels (0: down, 1: no change, 2: up)

# Load images and labels into arrays
X = []
y = []  # Labels corresponding to each image

for img_file in image_files:
    img = preprocess_image(os.path.join(chart_image_dir, img_file))
    X.append(img)
    y.append(1)  # Example: replace with actual labels

X = np.array(X).reshape(-1, 64, 64, 1)
y = np.array(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Build the CNN model
model_cnn = Sequential()
model_cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dense(3, activation='softmax'))  # 3 classes: up, down, no change

# 2. Compile the model
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Train the model
model_cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 4. Evaluate the model
loss, accuracy = model_cnn.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
