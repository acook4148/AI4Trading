import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Data Collection
def load_data(stock_symbol, start_date, end_date):
    import yfinance as yf
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data['Close']

# Step 2: Data Preprocessing
def preprocess_data(data, look_back=60):
    data = data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Step 3: Model Building
def build_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Training
def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    early_stop = EarlyStopping(monitor='loss', patience=10)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stop])

# Step 5: Prediction
def make_predictions(model, data, look_back, scaler):
    predictions = []
    for i in range(len(data)):
        x_input = data[i].reshape(1, -1)
        x_input = x_input.reshape((1, look_back, 1))
        pred = model.predict(x_input, verbose=0)
        predictions.append(pred[0][0])
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Step 6: Evaluation
def plot_predictions(actual_data, predicted_data):
    plt.plot(actual_data, color='blue', label='Actual Stock Price')
    plt.plot(predicted_data, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Main function to run the prediction
def main(stock_symbol, start_date, end_date):
    data = load_data(stock_symbol, start_date, end_date)
    look_back_num_days = 60
    X, y, scaler = preprocess_data(data, look_back_num_days)
    
    model = build_model()
    train_model(model, X, y)
    
    predictions = make_predictions(model, X, look_back_num_days, scaler)
    plot_predictions(data.values[60:], predictions)

# Example usage
main('AAPL', '2020-01-01', '2023-01-01')
