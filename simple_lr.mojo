from tensors import Rank1Tensor

alias type = DType.float64

struct SimpleLR[lr: Scalar[type]]:
    
    var intercept: Scalar[type]
    var slope: Scalar[type]

    fn __init__(inout self):
        self.intercept = 0.0
        self.slope = 0.0

    fn _forward(self, xs: Rank1Tensor[type]) -> Rank1Tensor[xs.dtype, xs.width]:
        return (xs * self.slope) + self.intercept

    fn _backward(
        inout self,
        xs: Rank1Tensor[type],
        ys: Rank1Tensor[type, xs.width]
    ) -> Scalar[type]:

        var ys_pred = self._forward(xs)
        var error = (ys - ys_pred) ** 2
        var mse = error.sum() / xs._width
        
        var slope_grad = ((-2.0 * xs) * (ys - ys_pred))
        var sg = slope_grad.sum() / xs._width

        var intercept_grad = -2.0 * (ys - ys_pred)
        var ig = intercept_grad.sum() / xs._width

        self.slope -= (lr * sg)
        self.intercept -= (lr * ig)

        return mse

    fn fit(
        inout self,
        xs: Rank1Tensor[type],
        ys: Rank1Tensor[type, xs.width],
        epochs: Int
    ):
        
        for _ in range(epochs):
            # print("Iteration ", i + 1, ": m = ", self.slope, ", b = ", self.intercept)
            _ = self._backward(xs, ys)

    fn predict(self, xs: Rank1Tensor[type]) -> Rank1Tensor[type, xs.width]:
        return self._forward(xs)

fn main():

    var xs = Rank1Tensor[type, 5]()
    var ys = Rank1Tensor[type, 5]()

    xs[0] = 0.0
    xs[1] = 1.0
    xs[2] = 2.0
    xs[3] = 3.0
    xs[4] = 4.0

    ys[0] = 1.0
    ys[1] = 3.0
    ys[2] = 7.0
    ys[3] = 13.0
    ys[4] = 21.0

    var slr = SimpleLR[lr=0.01]()

    slr.fit(xs, ys, 1000)

    print(slr.slope, slr.intercept)
