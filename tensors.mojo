from algorithm import vectorize

# rank is "height" of tensor (number of dimensions)
# size is "width" of tensor (number of elements per dimension)
struct Tensor[dtype: DType, rank: Int, size: Int]:
    
    var _data: DTypePointer[dtype]
    alias simd_width: Int = simdwidthof[dtype]() * 2
    alias shape: (Int, Int) = (rank, size)

    fn __init__(inout self):
        self._data = DTypePointer[dtype].alloc(rank * size)
        memset_zero(self._data, rank * size)

    fn __copyinit__(inout self, other: Tensor[dtype, rank, size]):
        self._data = DTypePointer[dtype].alloc(rank * size)
        memcpy(self._data, other._data, rank * size)

    fn __del__(owned self):
        self._data.free()

    fn __repr__(self) -> String:
        
        var result: String = "["
        for x in range(rank):
            result = result + "["
            for y in range(size):
                var p = self[x, y]
                result = result + String(p)
                if y < size - 1:
                    result = result + ", "
            result = result + "]"
            if x < rank - 1:
                result = result + ",\n"

        result = result + "]"
        return result

    fn __getitem__(self, x: Int, y: Int) -> Scalar[dtype]:
        return self._data.load((x * size) + y)

    fn __setitem__(inout self, x: Int, y: Int, value: Scalar[dtype]):
        self._data.store((x * size) + y, value)
    
    fn load[nelts: Int](self, x: Int, y: Int) -> SIMD[dtype, nelts]:
        return self._data.load[width=nelts]((x * size) + y)

    fn store[nelts: Int](inout self, x: Int, y: Int, value: SIMD[dtype, nelts]):
        self._data.store[width=nelts]((x * size) + y, value)

    # operations

    fn __add__(self, other: Scalar[dtype]) -> Tensor[dtype, rank, size]:
        var result = Tensor[dtype, rank, size]()
        for x in range(rank):
            @parameter
            fn add[nelts: Int](y: Int):
                result.store[nelts](x, y, self.load[nelts](x, y) + other)

            vectorize[add, self.simd_width, size = size]()
        return result

    fn __radd__(self, other: Scalar[dtype]) -> Tensor[dtype, rank, size]:
        return self + other
    
    # only implementing same-shape addition for now
    fn __add__(self, other: Tensor[dtype, rank, size]) -> Tensor[dtype, rank, size]:
        var result = Tensor[dtype, rank, size]()
        for x in range(rank):
            @parameter
            fn add[nelts: Int](y: Int):
                result.store[nelts](x, y, self.load[nelts](x, y) + other.load[nelts](x, y))

            vectorize[add, self.simd_width, size = size]()
        return result

    fn __neg__(self) -> Tensor[dtype, rank, size]:
        var result = Tensor[dtype, rank, size]()
        for x in range(rank):
            @parameter
            fn neg[nelts: Int](y: Int):
                result.store[nelts](x, y, -self.load[nelts](x, y))

            vectorize[neg, self.simd_width, size = size]()
        return result

    fn __sub__(self, other: Scalar[dtype]) -> Tensor[dtype, rank, size]:

        var result = Tensor[dtype, rank, size]()
        for x in range(rank):
            @parameter
            fn sub[nelts: Int](y: Int):
                result.store[nelts](x, y, self.load[nelts](x, y) - other)

            vectorize[sub, self.simd_width, size = size]()
        return result

    fn __rsub__(self, other: Scalar[dtype]) -> Tensor[dtype, rank, size]:
        
        var result = Tensor[dtype, rank, size]()
        for x in range(rank):
            @parameter
            fn sub[nelts: Int](y: Int):
                result.store[nelts](x, y, other - self.load[nelts](x, y))

            vectorize[sub, self.simd_width, size = size]()
        return result

    fn __sub__(self, other: Tensor[dtype, rank, size]) -> Tensor[dtype, rank, size]:
        var result = Tensor[dtype, rank, size]()
        for x in range(rank):
            @parameter
            fn sub[nelts: Int](y: Int):
                result.store[nelts](x, y, self.load[nelts](x, y) - other.load[nelts](x, y))

            vectorize[sub, self.simd_width, size = size]()
        return result

    fn __mul__(self, other: Scalar[dtype]) -> Tensor[dtype, rank, size]:
        
        var result = Tensor[dtype, rank, size]()
        for x in range(rank):
            @parameter
            fn mul[nelts: Int](y: Int):
                result.store[nelts](x, y, self.load[nelts](x, y) * other)

            vectorize[mul, self.simd_width, size = size]()
        return result

    fn __rmul__(self, other: Scalar[dtype]) -> Tensor[dtype, rank, size]:
        return self * other
    
    # element-wise multiplication - hadamard product
    fn __mul__(self, other: Tensor[dtype, rank, size]) -> Tensor[dtype, rank, size]:
        var result = Tensor[dtype, rank, size]()
        for x in range(rank):
            @parameter
            fn mul[nelts: Int](y: Int):
                result.store[nelts](x, y, self.load[nelts](x, y) * other.load[nelts](x, y))

            vectorize[mul, self.simd_width, size = size]()
        return result

    fn __truediv__(self, other: Scalar[dtype]) -> Tensor[dtype, rank, size]:
        
        var result = Tensor[dtype, rank, size]()
        for x in range(rank):
            @parameter
            fn div[nelts: Int](y: Int):
                result.store[nelts](x, y, self.load[nelts](x, y) / other)

            vectorize[div, self.simd_width, size = size]()
        return result

    fn __pow__(self, other: Scalar[dtype]) -> Tensor[dtype, rank, size]:
        
        var result = Tensor[dtype, rank, size]()
        for x in range(rank):
            @parameter
            fn pow[nelts: Int](y: Int):
                result.store[nelts](x, y, self.load[nelts](x, y) ** other)

            vectorize[pow, self.simd_width, size = size]()
        return result

    fn __matmul__[p: Int](self, other: Tensor[dtype, size, p]) -> Tensor[dtype, rank, p]:

        var result = Tensor[dtype, rank, p]()
        # for each item in self, we do a vectorized multiplication with the corresponding row in other
        # and sum the results into the result tensor
        for m in range(self.rank):
            for k in range(self.size):
                @parameter
                fn dot[nelts: Int](n: Int):
                    result.store[nelts](m, n, result.load[nelts](m, n) + self[m, k] * other.load[nelts](k, n))
                vectorize[dot, self.simd_width, size = p]()

        return result

    fn T(self) -> Tensor[dtype, size, rank]:
        var result = Tensor[dtype, size, rank]()
        for x in range(rank):
            for y in range(size):
                result[y, x] = self[x, y]
        return result

    fn sum(self) -> Scalar[dtype]:
        
        var result: Scalar[dtype] = 0
        for x in range(rank):
            for y in range(size):
                result = result + self[x, y]

        return result

fn dot[dtype: DType, size: Int](
    x: Tensor[dtype=dtype, rank=1, size=size],
    y: Tensor[dtype=dtype, rank=1, size=size]
) -> Scalar[x.dtype]:
    return (x * y).sum()

fn dot[dtype: DType, rank_x: Int, size_y: Int](
    x: Tensor[dtype=dtype, rank=rank_x, size=2],
    y: Tensor[dtype=dtype, rank=2, size=size_y]
) -> Tensor[dtype, rank_x, size_y]:
    return x @ y

fn dot[dtype: DType, n: Int](
    x: Tensor[dtype=dtype, rank=2, size=n],
    y: Tensor[dtype=dtype, rank=n, size=2]
) -> Tensor[dtype, 2 , 2]:
    return x @ y

fn dot[dtype: DType](
    x: Tensor[dtype],
    y: Scalar[dtype]
) -> Tensor[dtype, x.rank, x.size]:
    return x * y

fn dot[dtype: DType](
    x: Scalar[dtype],
    y: Tensor[dtype]
) -> Tensor[dtype, y.rank, y.size]:
    return x * y

fn dot[dtype: DType, x_size: Int](
    x: Tensor[dtype, size=x_size],
    y: Tensor[dtype, rank=1, size=x_size]
) -> Tensor[dtype, 1, x.rank]:
    
    var result = Tensor[dtype, 1, x.rank]()
    for i in range(x.rank):
        for j in range(x.size):
            result[0, i] += x[i, j] * y[0, j]

    return result

fn dot[dtype: DType, n: Int](
    x: Tensor[dtype, size=n],
    y: Tensor[dtype, rank=n]
) -> Tensor[dtype, x.rank, y.size]:
    
    return x @ y
