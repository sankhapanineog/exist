import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Function to generate random time-series data
def generate_random_data():
    np.random.seed(42)
    timestamps = pd.date_range(start="2023-01-01", end="2023-01-10", freq='H')
    values = np.random.normal(loc=0, scale=1, size=len(timestamps))
    data = pd.DataFrame({'timestamp': timestamps, 'value': values})
    return data

# Function to train the model and make predictions
def train_and_predict(data, threshold):
    model = IsolationForest(contamination=threshold)
    predictions = model.fit_predict(data[['value']])
    return predictions

# Streamlit app
def main():
    st.title("Asset Health Prediction App")

    # Generate random data
    data = generate_random_data()

    # Sidebar - Threshold
    st.sidebar.header("Set Threshold")
    threshold = st.sidebar.slider("Select Threshold", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    # Train and predict
    predictions = train_and_predict(data, threshold)

    # Display results
    st.subheader("Raw Data")
    st.write(data)

    st.subheader("Predictions")
    st.write(predictions)

    # Plot data and predictions
    st.subheader("Data and Predictions Plot")
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestamp'], data['value'], label='Data')
    plt.scatter(data['timestamp'][predictions == -1], data['value'][predictions == -1], color='red', label='Unhealthy')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    st.pyplot(plt)

if __name__ == "__main__":
    main()
