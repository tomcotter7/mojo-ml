from algorithm import vectorize
from tensors import Tensor, dot

alias type = DType.float64
alias num_features = 2

struct MultipleLR[lr: Scalar[type], features: Int = num_features]:
    
    var intercept: Scalar[type]
    var slope: Tensor[type, 1, features] 

    fn __init__(inout self):
        self.intercept = 0.0
        self.slope = Tensor[type, 1, features]()

    fn _forward(self, xs: Tensor[type, size=features]) -> Tensor[type, 1, xs.rank]:
        var y_pred = dot(xs, self.slope) + self.intercept
        return y_pred

    fn _backward(inout self, xs: Tensor[type, size=features], y_true: Tensor[type, 1, xs.rank]) -> Scalar[type]:
        
        var y_pred = self._forward(xs)
        var grad_m = -2 * dot(xs.T(), (y_true - y_pred)) / xs.rank
        var grad_b = -2 * (y_true - y_pred).sum() / xs.rank

        self.slope = self.slope - (lr * grad_m)
        self.intercept = self.intercept - (lr * grad_b)

        return ((y_true - y_pred) ** 2).sum() / xs.rank

    fn fit(inout self, xs: Tensor[type, size=features], y_true: Tensor[type, 1, xs.rank], epochs: Int):
        
        for _ in range(epochs):
            _ = self._backward(xs, y_true)


def main():

    var X = Tensor[type, 5, num_features]()
    var Y = Tensor[type, 1, 5]()

    X[0, 0] = 1.0
    X[0, 1] = 2.0

    X[1, 0] = 2.0
    X[1, 1] = 3.0

    X[2, 0] = 4.0
    X[2, 1] = 5.0

    X[3, 0] = 6.0
    X[3, 1] = 7.0

    X[4, 0] = 8.0
    X[4, 1] = 9.0

    Y[0, 0] = 5.0
    Y[0, 1] = 8.0
    Y[0, 2] = 11.0
    Y[0, 3] = 14.0
    Y[0, 4] = 17.0


    var mlr = MultipleLR[0.01]()

    mlr.fit(X, Y, 10000)

    print("Intercept:", mlr.intercept)
    print("Slope: ")
    var slope = mlr.slope

    for i in range(slope.rank):
        for j in range(slope.size):
            print(slope[i, j])

    
