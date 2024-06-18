from algorithm import vectorize
from tensors import RankNTensor, Rank1Tensor

alias type = DType.float64
alias size = 3

struct MultipleLR[lr: Scalar[type], width: Int = size]:
    
    var intercept: Scalar[type]
    var slope: RankNTensor[type, 1, width] 

    fn __init__(inout self):
        self.intercept = 0.0
        self.slope = RankNTensor[type, 1, width]()

    fn _forward(self, xs: RankNTensor[type, width=width]) -> RankNTensor[type, 1, xs.rank]:

        var result = RankNTensor[type, 1, xs.rank]()

        for i in range(xs.rank):
            @parameter
            fn fw[nelts: Int](n: Int):
                result.store[nelts](0, i, xs.load[nelts](i, n) * self.slope.load[nelts](0, i) + self.intercept)

            vectorize[fw, simdwidthof[type]() * 2, size = size]()
        
        return result

    fn _backward(
        inout self,
        xs: RankNTensor[type, width=width],
        ys: RankNTensor[type, 1, xs.rank]
    ) -> Scalar[type]:

        var ys_pred = self._forward(xs)
        var error = (ys - ys_pred) ** 2
        var diff = ys - ys_pred

        print("Error:")
        for i in range(diff.width):
            print(diff[0, i])

        var mse = error.sum() / xs.rank
        
        var dp = xs * diff
        var dot_product = dp.sum()
        print("Dot Product:")
        for i in range(dp.width):
            print(dp[0, i])
        var slope_grad = (-2.0 * dot_product) / xs.rank
        var sg = lr * slope_grad
        
        var diff_sum = diff.sum()
        var intercept_grad = (-2.0 * diff_sum) / xs.rank
        var ig = lr * intercept_grad


        self.slope = self.slope - sg
        self.intercept -= ig

        print("New intercept: ", self.intercept)

        return mse

    fn fit(
        inout self,
        xs: RankNTensor[type, width=width],
        ys: RankNTensor[type, 1, xs.rank],
        epochs: Int
    ):

        for i in range(epochs):
            print("Iteration ", i)
            for j in range(self.slope.width):
                print("Slope: ", self.slope[0, j])
            print("Intercept: ", self.intercept)
            
            _ = self._backward(xs, ys)



def main():

    var xs = RankNTensor[type, 3, 2]()
    xs[0, 0] = 1.0
    xs[0, 1] = 2.0
    xs[1, 0] = 3.0
    xs[1, 1] = 4.0
    xs[2, 0] = 5.0
    xs[2, 1] = 6.0

    var ys = RankNTensor[type, 1, xs.rank]()
    ys[0, 0] = 3.0
    ys[0, 1] = 5.0
    ys[0, 2] = 7.0
    
    var zs = xs.T()
    var ds = ys.T()
    var cs = zs.matmul(ds)

    for i in range(cs.rank):
        for j in range(cs.width):
            print(cs[i, j])




   
    # var xs = RankNTensor[type, 5, 2]()
    # xs[0, 0] = 1.0
    # xs[0, 1] = 2.0
    # xs[1, 0] = 2.0
    # xs[1, 1] = 3.0
    # xs[2, 0] = 4.0
    # xs[2, 1] = 5.0
    # xs[3, 0] = 6.0
    # xs[3, 1] = 7.0
    # xs[4, 0] = 8.0
    # xs[4, 1] = 9.0

    # var ys = RankNTensor[type, 1, 5]()
    # ys[0, 0] = 5.0
    # ys[0, 1] = 8.0
    # ys[0, 2] = 11.0
    # ys[0, 3] = 14.0
    # ys[0, 4] = 17.0

    # var mlr = MultipleLR[0.01, width=xs.width]()
    # mlr.fit(xs, ys, 1)

    # print("Intercept: ", mlr.intercept)



    





