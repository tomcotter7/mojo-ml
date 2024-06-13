struct SimpleLinearRegression:
    
    var intercept: UnsafePointer[Float32]
    var slope: UnsafePointer[Float32]
    var lr: Float32

    fn __init__(inout self, lr: Float32):
        self.intercept = UnsafePointer[Float32].alloc(1)
        self.slope = UnsafePointer[Float32].alloc(1) 
        self.lr = lr

        initialize_pointee_copy(self.intercept, 0.0)
        initialize_pointee_copy(self.slope, 0.0)

    fn _forward(self, x: Float32) -> Float32:
        return self.intercept[] + self.slope[] * x

    fn _backward(self, x: Float32, y: Float32) -> Float32:
        
        var y_pred: Float32 = self._forward(x)

        var slope_grad = (-2 * x) * (y - y_pred)
        var intercept_grad = -2 * (y - y_pred)

        self.slope[] = self.slope[] - (self.lr * slope_grad)
        self.intercept[] = self.intercept[] - (self.lr * intercept_grad)

        var squared_error: Float32 = (y - y_pred) ** 2
        return squared_error

    fn _fit(self, xs: List[Float32], ys: List[Float32]):

        var mse: Float32 = 0

        for i in range(len(xs)):
            mse += self._backward(xs[i], ys[i])


        print("MSE: ", mse)
        print("Slope: ", self.slope[])
        print("Intercept: ", self.intercept[])

    fn train(self, xs: List[Float32], ys: List[Float32], epochs: Int):

        for _ in range(epochs):
            self._fit(xs, ys)
        
    fn predict(self, x: Float32) -> Float32:
        return self._forward(x)


fn main():
    var slr = SimpleLinearRegression(0.1)
    var xs = List[Float32](0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    var ys = List[Float32](0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

    slr.train(xs, ys, 10)
