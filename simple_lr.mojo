struct SimpleLinearRegression:
    
    var intercept: DTypePointer[DType.float32]
    var slope: DTypePointer[DType.float32]
    var lr: Float32

    fn __init__(inout self, lr: Float32):
        self.intercept = DTypePointer[DType.float32].alloc(1)
        self.slope = DTypePointer[DType.float32].alloc(1) 
        self.lr = lr

        self.intercept.store(0.0)
        self.slope.store(0.0)

    fn _forward(self, xs: SIMD[DType.float32]) -> SIMD[DType.float32]:
        return self.intercept.load() + self.slope.load() * xs

    fn _backward(self, xs: SIMD[DType.float32], ys: SIMD[DType.float32]) -> Float32:

        var ys_pred: SIMD[DType.float32] = self._forward(xs)
        var error: SIMD[DType.float32] = (ys - ys_pred) ** 2
        var mse = error.reduce_add[1]() / len(xs)

        var slope_grad = (-2 * xs) * (ys - ys_pred)
        var intercept_grad = -2 * (ys - ys_pred)

        self.slope.store(self.slope.load() - (self.lr * (slope_grad.reduce_add[1]() / len(xs))))
        self.intercept.store(self.intercept.load() - (self.lr * (intercept_grad.reduce_add[1]() / len(xs))))

        return mse

    fn fit(self, xs: SIMD[DType.float32], ys: SIMD[DType.float32], epochs: Int):
        
        for _ in range(epochs):
            _ = self._backward(xs, ys)
            # var mse = self._backward(xs, ys)
            # print("MSE: ", mse)
            # print("New slope: ", self.slope.load())
            # print("New intercept: ", self.intercept.load())

    fn predict(self, xs: SIMD[DType.float32]) -> SIMD[DType.float32]:
        return self._forward(xs)


fn main():
    var slr = SimpleLinearRegression(lr=0.1)
    var x = SIMD[DType.float32](0.0, 1.0, 2.0, 3.0, 4.0)
    var y = SIMD[DType.float32](0.0, 1.0, 2.0, 3.0, 4.0)

    slr.fit(x, y, 100)

    var x_test = SIMD[DType.float32](5.0, 6.0, 7.0, 8.0, 9.0)
    var y_pred = slr.predict(x_test)
    print("Predictions: ", y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4])
