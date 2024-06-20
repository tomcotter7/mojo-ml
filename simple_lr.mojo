from tensors import Tensor

alias type = DType.float64

struct SimpleLR[lr: Scalar[type]]:
    
    var intercept: Scalar[type]
    var slope: Scalar[type]

    fn __init__(inout self):
        self.intercept = 0.0
        self.slope = 0.0

    fn _forward[size: Int](self, xs: Tensor[type, 1, size]) -> Tensor[type, 1, size]:
        var ys = (xs * self.slope) + self.intercept
        return ys


    fn _backward[size: Int](
        inout self,
        xs: Tensor[type, 1, size],
        ys: Tensor[type, 1, size]
    ) -> Scalar[type]:

        var ys_pred = self._forward(xs)
        var error = (ys - ys_pred) ** 2
        var mse = error.sum() / xs.size
        
        var slope_grad = ((-2.0 * xs) * (ys - ys_pred))
        var sg = slope_grad.sum() / xs.size

        var intercept_grad = -2.0 * (ys - ys_pred)
        var ig = intercept_grad.sum() / xs.size

        self.slope -= (lr * sg)
        self.intercept -= (lr * ig)

        return mse

    fn fit[size: Int](
        inout self,
        xs: Tensor[type, 1, size],
        ys: Tensor[type, 1, size],
        epochs: Int
    ):
        
        for _ in range(epochs):
            # print("Iteration ", i + 1, ": m = ", self.slope, ", b = ", self.intercept)
            _ = self._backward(xs, ys)


fn main():

    var xs = Tensor[type, 1, 5]()
    var ys = Tensor[type, 1, 5]()

    xs[0, 0] = 0.0
    xs[0, 1] = 1.0
    xs[0, 2] = 2.0
    xs[0, 3] = 3.0
    xs[0, 4] = 4.0


    ys[0, 0] = 1.0
    ys[0, 1] = 3.0
    ys[0, 2] = 7.0
    ys[0, 3] = 13.0
    ys[0, 4] = 21.0

    var slr = SimpleLR[lr=0.01]()

    slr.fit(xs, ys, 1000)

    print(slr.slope, slr.intercept)
