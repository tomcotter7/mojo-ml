struct SimpleLinearRegression:
    var intercept: Float32
    var slope: Float32
    var lr: Float32

    fn __init__(inout self, lr: Float32):
        self.intercept = 0.0
        self.slope = 0.0
        self.lr = lr

    fn _forward(self, xs: SIMD[DType.float32]) -> SIMD[DType.float32]:
        return self.intercept + self.slope * xs

    fn _backward(inout self, xs: SIMD[DType.float32], ys: SIMD[DType.float32]) -> Float32:

        var ys_pred: SIMD[DType.float32] = self._forward(xs)
        var error: SIMD[DType.float32] = (ys - ys_pred) ** 2
        var mse = error.reduce_add[1]() / len(xs)

        var slope_grad = (-2 * xs) * (ys - ys_pred)
        var intercept_grad = -2 * (ys - ys_pred)

        self.slope -= (self.lr * (slope_grad.reduce_add[1]() / len(xs)))
        self.intercept -= (self.lr * (intercept_grad.reduce_add[1]() / len(xs)))

        return mse

    fn fit(inout self, xs: SIMD[DType.float32], ys: SIMD[DType.float32], epochs: Int):
        
        for _ in range(epochs):
            _ = self._backward(xs, ys)

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
