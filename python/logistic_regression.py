import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(50)
X, Y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=1, random_state=50)
X = np.concatenate((X, np.ones((100, 1))), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# p(x) = exp(w0 + w1*x1 + w2*x2 + w3*x3) / (1 + exp(w0 + w1*x1 + w2*x2 + w3*x3))
weights = np.random.rand(4)

lr = 0.01
num_iterations = 30000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bce_loss(y_pred, y):
    epsilon = 1e-9
    return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

prev_loss = -100000

for i in range(num_iterations):
    z = np.dot(X_train, weights)
    y_pred = sigmoid(z)
    loss = bce_loss(y_pred, y_train)


    gradient_w = np.dot(X_train.T, (y_pred - y_train)) / len(y_train) + 0.01 * weights / len(y_train)
    gradient_bias = np.mean(y_pred - y_train)
    weights -= lr * gradient_w

    if i > 0 and abs(prev_loss - loss) < 1e-9:
        print("Converged at iteration: ", i)
        break

    prev_loss = loss


print("Weights: ", weights)

y_pred = sigmoid(np.dot(X_test, weights))
y_pred = np.round(y_pred)



print("Accuracy: ", accuracy_score(y_test, y_pred))

# Using sklearn

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

