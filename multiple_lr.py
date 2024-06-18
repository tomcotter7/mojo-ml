import numpy as np

X = np.array([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
Y = np.array([5.0, 8.0, 11.0, 14.0, 17.0])

m = np.zeros(X.shape[1])
b = 0

learning_rate = 0.01
num_iterations = 1

for i in range(num_iterations):
    print(f"Iteration {i}:")
    print(f"Slope (m): {m}")
    print(f"Intercept (b): {b: .3f}")
    y_pred = np.dot(X, m) + b
    print(X.T)
    print(f"{np.sum(X.T * (Y - y_pred))}")
    print(f"Error: {Y - y_pred}")
    grad_m = -2 * np.dot(X.T, (Y - y_pred)) / len(X)
    print("Dot prod:", np.dot(X.T, (Y - y_pred)))
    grad_b = -2 * np.sum(Y - y_pred) / len(X)
    
    m = m - learning_rate * grad_m
    b = b - learning_rate * grad_b

    print(f"New intercept: {b}")



print(f"Slope (m): {m}")
print(f"Intercept (b): {b: .3f}")


