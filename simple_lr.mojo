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

    fn _forward(self, xs: SIMD[DType.float32]) -> SIMD[DType.float32]:
        return self.intercept[] + self.slope[] * xs

    fn _backward(self, xs: SIMD[DType.float32], ys: SIMD[DType.float32]) -> Float32:

        var ys_pred: SIMD[DType.float32] = self._forward(xs)
        var error: SIMD[DType.float32] = (ys - ys_pred) ** 2
        var mse = error.reduce_add[1]() / len(xs)

        var slope_grad = (-2 * xs) * (ys - ys_pred)
        var intercept_grad = -2 * (ys - ys_pred)

        self.slope[] = self.slope[] - (self.lr * (slope_grad.reduce_add[1]() / len(xs)))
        self.intercept[] = self.intercept[] - (self.lr * (intercept_grad.reduce_add[1]() / len(xs)))

        return mse

    fn fit(self, xs: SIMD[DType.float32], ys: SIMD[DType.float32], epochs: Int):
        
        for _ in range(epochs):
            var mse = self._backward(xs, ys)
            print("MSE: ", mse)
            print("New slope: ", self.slope[])
            print("New intercept: ", self.intercept[])


fn main():
    var slr = SimpleLinearRegression(lr=0.01)
    var x = SIMD[DType.float32](0.0, 1.0, 2.0, 3.0, 4.0)
    var y = SIMD[DType.float32](0.0, 1.0, 2.0, 3.0, 4.0)

    slr.fit(x, y, 10)
