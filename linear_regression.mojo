from mishra import Matrix

struct LinearRegression[features: Int, dtype: DType = DType.float64, lr: Scalar[dtype]=0.01]:

    var intercept: Scalar[dtype]
    var slope: Matrix[dtype]

    fn __init__(inout self) raises:
        self.slope = Matrix[dtype](features, 1)
        self.intercept = 0.0

    fn _forward(self, xs: Matrix[dtype]) raises -> Matrix[dtype]:
        
        var y_pred = (xs @ self.slope) + self.intercept
        return y_pred
    
    fn _backward(inout self, xs: Matrix[dtype], y_true: Matrix[dtype]) raises -> Scalar[dtype]:
        
        var y_pred = self._forward(xs)
        var grad_m = (xs.T() @ (y_true - y_pred)) * -2 / xs._rows

        var grad_b = (y_true - y_pred).sum() * -2 / xs._rows

        self.slope = self.slope - (grad_m * lr)
        self.intercept = self.intercept - (lr * grad_b)

        return ((y_true - y_pred) ** 2).sum() / xs._rows


    fn train(inout self, xs: Matrix[dtype], ys: Matrix[dtype], epochs: Int) raises -> Scalar[dtype]:

        if xs._cols != features:
            raise "Must have the same number of features"

        var loss: Scalar[dtype] = 0.0
        
        for e in range(epochs):
            loss = self._backward(xs, ys)
            if e % 100 == 0:
                print("Epoch: ", e, "Loss: ", loss)

        return loss


fn main() raises:

    var lr = LinearRegression[2]()

   
    var xs = Matrix(5, 2)
    xs[0, 0] = 1.0
    xs[0, 1] = 2.0
    xs[1, 0] = 2.0
    xs[1, 1] = 3.0
    xs[2, 0] = 4.0
    xs[2, 1] = 5.0
    xs[3, 0] = 6.0
    xs[3, 1] = 7.0
    xs[4, 0] = 8.0
    xs[4, 1] = 9.0


    var ys = Matrix(5, 1)

    ys[0, 0] = 5.0
    ys[1, 0] = 8.0
    ys[2, 0] = 11.0
    ys[3, 0] = 14.0
    ys[4, 0] = 17.0

    var loss = lr.train(xs, ys, 1000)

    print("Final loss: ", loss)
    
    print(lr.slope)
    print(lr.intercept)


