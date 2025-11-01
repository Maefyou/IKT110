import numpy as np
import plotly.express as px
import numba as nb
from numba import jit

# Numba-optimized functions for speed
@jit(nopython=True)
def predict_numba(theta, bias, xs): 
    """Fast prediction using Numba JIT compilation"""
    return np.dot(xs, theta) + bias

@jit(nopython=True)
def compute_gradients_numba(theta, bias, xs, y):
    """Fast gradient computation using Numba JIT compilation"""
    h = np.dot(xs, theta) + bias
    grad_theta = np.dot(xs.T, (h - y))
    grad_bias = np.sum(h - y)
    return grad_theta, grad_bias

@jit(nopython=True)
def compute_loss_numba(theta, bias, xs, y):
    """Fast loss computation using Numba JIT compilation"""
    h = np.dot(xs, theta) + bias
    return np.sum((h - y)**2)

@jit(nopython=True)
def sgd_step_numba(theta, bias, batch_x, batch_y, learning_rate):
    """Single SGD step optimized with Numba"""
    batch_m = batch_x.shape[0]
    grad_theta, grad_bias = compute_gradients_numba(theta, bias, batch_x, batch_y)
    theta = theta - (learning_rate / batch_m) * grad_theta
    bias = bias - (learning_rate / batch_m) * grad_bias
    return theta, bias

@jit(nopython=True)
def train_sgd_numba(theta, bias, data_x, data_y, learning_rate, batch_size, n_epochs):
    """Full SGD training loop optimized with Numba"""
    m = data_x.shape[0]
    j_history = np.zeros(n_epochs)
    
    for epoch in range(n_epochs):
        # Shuffle the data at the beginning of each epoch
        indices = np.random.permutation(m)
        data_x_shuffled = data_x[indices]
        data_y_shuffled = data_y[indices]
        
        # Process data in mini-batches
        for i in range(0, m, batch_size):
            end_idx = min(i + batch_size, m)
            batch_x = data_x_shuffled[i:end_idx]
            batch_y = data_y_shuffled[i:end_idx]
            
            # Update parameters
            theta, bias = sgd_step_numba(theta, bias, batch_x, batch_y, learning_rate)
        
        # Record loss on full dataset after each epoch
        j_history[epoch] = compute_loss_numba(theta, bias, data_x, data_y)
    
    return theta, bias, j_history


# the dataset (no augmentation - just the features)
data_x = np.array([[0.5], [1.0], [2.0]])
data_y = np.array([[1.0], [1.5], [2.5]])
n_features = data_x.shape[1]

# Hyperparameters
learning_rate = 0.1
batch_size = 2  # mini-batch size
n_epochs = 10

# Initialize parameters
theta = np.zeros((n_features, 1))
bias = 0.0

# Set random seed for reproducibility
np.random.seed(42)

# Run SGD with mini-batches (Numba-optimized - pew pew!)
print("Running Numba-optimized SGD...")
import time
start_time = time.time()

theta, bias, j_history = train_sgd_numba(theta, bias, data_x, data_y, learning_rate, batch_size, n_epochs)

end_time = time.time()
print(f"Training completed in {(end_time - start_time)*1000:.2f} ms")
    
print("\n=== Results ===")
print("theta shape:", theta.shape)
print("theta value:", theta.flatten())
print("bias value:", bias)

# Final loss (L2 error)
j_final = compute_loss_numba(theta, bias, data_x, data_y)
print("\nThe L2 error is: {:.2f}".format(j_final))

# Find the L1 error
y_pred = predict_numba(theta, bias, data_x)
l1_error = np.abs(y_pred - data_y).sum()
print("The L1 error is: {:.2f}".format(l1_error))


# Find the R^2 
# if the data is normalized: use the normalized data not the original data (task 3 hint).
# https://en.wikipedia.org/wiki/Coefficient_of_determination
u = ((data_y - y_pred)**2).sum()
v = ((data_y - data_y.mean())** 2).sum()
print("R^2: {:.2f}".format(1 - (u/v)))

# Plot the result
fig = px.line(y=j_history, x=np.arange(len(j_history)), 
              labels={'x': 'Epoch', 'y': 'Loss'},
              title="J(theta) - Loss History (Numba-Optimized SGD)")
fig.show()
