import numpy as np

# Generate some sample data
X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([1.0, 3.0, 7.0, 13.0, 21.0])

# Initialize the parameters
m = 0
b = 0

# Hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Perform gradient descent
for i in range(num_iterations):
    # Compute the predicted values
    y_pred = m * X + b
    # Calculate the gradients
    grad_m = -2 * np.sum(X * (y - y_pred)) / len(X)
    grad_b = -2 * np.sum(y - y_pred) / len(X)
    
    # Update the parameters
    m = m - learning_rate * grad_m
    b = b - learning_rate * grad_b


# Print the final parameters
print(f"Slope (m): {m:.3f}")
print(f"Intercept (b): {b:.3f}")
