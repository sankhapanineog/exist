import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Forward propagation
def forward_propagation(X, parameters, activations):
    caches = []
    A = X

    for i in range(len(parameters) // 2):
        W = parameters[f'W{i + 1}']
        b = parameters[f'b{i + 1}']
        activation = activations[i]

        Z = np.dot(W, A) + b
        A = activation(Z)

        cache = (A, W, b, Z)
        caches.append(cache)

    return A, caches

# Backward propagation
def backward_propagation(AL, Y, caches, activations):
    grads = {}
    L = len(caches)

    dAL = 2 * (AL - Y)

    for i in reversed(range(L)):
        A, W, b, Z = caches[i]
        activation = activations[i]

        dZ = dAL * activation(Z, derivative=True)
        dW = np.dot(dZ, A.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        dAL = np.dot(W.T, dZ)

        grads[f'dW{i + 1}'] = dW
        grads[f'db{i + 1}'] = db

    return grads

# Update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for i in range(L):
        parameters[f'W{i + 1}'] -= learning_rate * grads[f'dW{i + 1}']
        parameters[f'b{i + 1}'] -= learning_rate * grads[f'db{i + 1}']

    return parameters

# Train neural network
def train_neural_network(X, Y, layers, activations, learning_rate, num_iterations):
    np.random.seed(42)
    parameters = initialize_parameters(layers)

    for i in range(num_iterations):
        AL, caches = forward_propagation(X, parameters, activations)
        cost = mean_squared_error(AL, Y)
        grads = backward_propagation(AL, Y, caches, activations)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            st.write(f"Cost after iteration {i}: {cost}")

    return parameters

# Function to make predictions
def predict(X, parameters, activations):
    AL, _ = forward_propagation(X, parameters, activations)
    return AL

# Function to initialize parameters
def initialize_parameters(layers):
    parameters = {}
    L = len(layers)

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layers[l], layers[l - 1]) * 0.01
        parameters[f'b{l}'] = np.zeros((layers[l], 1))

    return parameters

# Streamlit app
def main():
    st.title("Neural Network Health Prediction App")

    # Generate random data
    data = generate_random_data()

    # Prepare data
    X = data['value'].values.reshape(1, -1)
    Y = np.array([[1]])  # Placeholder for future health label, 1 for healthy, 0 for unhealthy

    # Sidebar - Neural Network Configuration
    st.sidebar.header("Neural Network Configuration")
    num_layers = st.sidebar.slider("Number of Layers", min_value=2, max_value=5, value=3)
    layer_sizes = [st.sidebar.slider(f"Layer {i} Size", min_value=1, max_value=100, value=10) for i in range(1, num_layers + 1)]
    activations = [st.sidebar.radio(f"Activation Function Layer {i}", ["ReLU", "Sigmoid"]) for i in range(1, num_layers)]
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    num_iterations = st.sidebar.slider("Number of Iterations", min_value=100, max_value=5000, value=1000, step=100)

    # Map activation function names to functions
    activation_functions = {"ReLU": relu, "Sigmoid": sigmoid}
    activations = [activation_functions[activation] for activation in activations]

    # Train neural network
    parameters = train_neural_network(X, Y, layer_sizes, activations, learning_rate, num_iterations)

    # Make predictions for the original data
    predictions_original = predict(X, parameters, activations)

    # Plot original data
    st.subheader("Original Data Plot")
    fig_original = px.line(data, x='timestamp', y='value', labels={'value': 'Original Data'})
    st.plotly_chart(fig_original)

    # Make predictions for a future timestamp
    future_timestamp = pd.Timestamp("2023-01-15 12:00:00")
    future_value = data.loc[data['timestamp'] == future_timestamp, 'value'].values[0]
    X_future = np.array([[future_value]])

    predictions_future = predict(X_future, parameters, activations)
    health_prediction = 'Healthy' if predictions_future > 0.5 else 'Unhealthy'

    # Plot predicted health for future timestamp
    st.subheader(f"Predicted Health for Future Timestamp ({future_timestamp}): {health_prediction}")
    fig_future = px.line(data, x='timestamp', y='value', labels={'value': 'Original Data'})
    fig_future.add_trace(px.scatter(x=[future_timestamp], y=[future_value], color=['green'], labels={'value': 'Predicted Health'}).data[0])
    st.plotly_chart(fig_future)

if __name__ == "__main__":
    main()
