from random import rand
from algorithm import vectorize

struct RankNTensor[dtype: DType, rank: Int, width: Int]:
    
    var _data: DTypePointer[dtype]

    fn __init__(inout self):
        self._data = DTypePointer[dtype].alloc(rank * width)
        memset_zero(self._data, rank * width)

    fn __copyinit__(inout self, other: RankNTensor[dtype, rank, width]):
        self._data = DTypePointer[dtype].alloc(rank * width)
        memcpy(self._data, other._data, rank * width)

    fn __getitem__(self, x: Int, y: Int)-> Scalar[dtype]:

        return self._data.load(x * width + y)

    fn __setitem__(inout self, x: Int, y: Int, value: Scalar[dtype]):
        self._data.store(x * width + y, value)

    fn __del__(owned self):
        self._data.free()

    fn load[nelts: Int](self, x: Int, y: Int) -> SIMD[dtype, nelts]:
        return self._data.load[width=nelts](x * width + y)

    fn store[nelts: Int](inout self, x: Int, y: Int, value: SIMD[dtype, nelts]):
        self._data.store[width=nelts](x * width + y, value)

    fn __add__(
        self,
        other: Scalar[dtype]
    ) -> RankNTensor[dtype, rank, width]:

        var result = RankNTensor[dtype, rank, width]()

        for i in range(rank):
            @parameter
            fn add[nelts: Int](n: Int):
                result.store[nelts](i, n, self.load[nelts](i, n) + other)

            vectorize[add, simdwidthof[dtype]() * 2, size = width]()

        return result

    fn __radd__(
        self,
        other: Scalar[dtype]
    ) -> RankNTensor[dtype, rank, width]:
        
        return self + other

    fn __sub__(
        self,
        other: RankNTensor[dtype, rank, width]
    ) -> RankNTensor[dtype, rank, width]:
        
        var result = RankNTensor[dtype, rank, width]()

        for i in range(rank):
            @parameter
            fn sub[nelts: Int](n: Int):
                result.store[nelts](i, n, self.load[nelts](i, n) - other.load[nelts](i, n))

            vectorize[sub, simdwidthof[dtype]() * 2, size = width]()

        return result

    fn __sub__(
        self,
        other: Scalar[dtype]
    ) -> RankNTensor[dtype, rank, width]:

        var result = RankNTensor[dtype, rank, width]()

        for i in range(rank):
            @parameter
            fn sub[nelts: Int](n: Int):
                result.store[nelts](i, n, self.load[nelts](i, n) - other)

            vectorize[sub, simdwidthof[dtype]() * 2, size = width]()

        return result

    fn __mul__(
        self,
        other: Scalar[dtype]
    ) -> RankNTensor[dtype, rank, width]:

        var result = RankNTensor[dtype, rank, width]()

        for i in range(rank):
            @parameter
            fn mul[nelts: Int](n: Int):
                result.store[nelts](i, n, self.load[nelts](i, n) * other)

            vectorize[mul, simdwidthof[dtype]() * 2, size = width]()

        return result

    fn __mul__(
        self,
        other: RankNTensor[dtype]
    ) -> RankNTensor[dtype, rank, width]:

        var result = RankNTensor[dtype, rank, width]()

        for i in range(rank):
            @parameter
            fn mul[nelts: Int](n: Int):
                result.store[nelts](i, n, self.load[nelts](i, n) * other.load[nelts](i, n))

            vectorize[mul, simdwidthof[dtype]() * 2, size = width]()

        return result
    
    fn __rmul__(
        self,
        other: RankNTensor[dtype]
    ) -> RankNTensor[dtype, rank, width]:
        return self * other

    fn __rmul__(
        self,
        other: Scalar[dtype]
    ) -> RankNTensor[dtype, rank, width]:
        return self * other

    fn __pow__(
        self,
        other: Scalar[dtype]
    ) -> RankNTensor[dtype, rank, width]:

        var result = RankNTensor[dtype, rank, width]()

        for i in range(rank):
            @parameter
            fn pow[nelts: Int](n: Int):
                result.store[nelts](i, n, self.load[nelts](i, n) ** other)

            vectorize[pow, simdwidthof[dtype]() * 2, size = width]()

        return result

    fn sum(inout self) -> Scalar[dtype]:

        var result: Scalar[dtype] = 0.0

        for i in range(rank):
            @parameter
            fn sum[nelts: Int](n: Int):
                result += (self.load[nelts](i, n).reduce_add())

            vectorize[sum, simdwidthof[dtype]() * 2, size = width]()

        return result

    fn T(self) -> RankNTensor[dtype, width, rank]:

        var result = RankNTensor[dtype, width, rank]()

        for i in range(self.rank):
            for j in range(self.width):
                result[j, i] = self[i, j]

        return result

    fn matmul[p: Int](
        self,
        other: RankNTensor[dtype, width, p]
    ) -> RankNTensor[dtype, rank, p]:
        
        var result = RankNTensor[dtype, rank, p]()

        for i in range(result.rank):
            for j in range(result.width):
                for k in range(width):
                    result[i, j] += self[i, k] * other[k, j]
                
        return result

    fn dot(
        self,
        other: RankNTensor[dtype, rank, width]
    ) -> Scalar[dtype]:

        var result: Scalar[dtype] = 0.0

        for i in range(rank):
            for j in range(width):
                result += self[i, j] * other[i, j]

        return result

struct Rank1Tensor[dtype: DType, width: Int]:
    
    var _data: DTypePointer[dtype]

    fn __init__(inout self):
        self._data = DTypePointer[dtype].alloc(width)
        # rand(self._data, width)
        memset_zero(self._data, width)

    fn __copyinit__(inout self, other: Rank1Tensor[dtype, width]):
        self._data = DTypePointer[dtype].alloc(width)
        memcpy(self._data, other._data, width)

    fn __getitem__(self, index: Int) -> Scalar[dtype]:
        return self._data.load(index)

    fn __setitem__(inout self, index: Int, value: Scalar[dtype]):
        self._data.store(index, value)

    fn load[nelts: Int](self, index: Int) -> SIMD[dtype, nelts]:
        return self._data.load[width=nelts](index)

    fn store[nelts: Int](inout self, index: Int, value: SIMD[dtype, nelts]):
        self._data.store[width=nelts](index, value)
    
    fn __del__(owned self):
        self._data.free()

    fn __add__(
        self: Rank1Tensor[dtype, width],
        other: Scalar[dtype]
    ) -> Rank1Tensor[dtype, width]:

        var result = Rank1Tensor[dtype, width]()

        @parameter
        fn add[nelts: Int](n: Int):
            result.store[nelts](n, self.load[nelts](n) + other)

        vectorize[add, simdwidthof[dtype]() * 2, size = width]()

        return result

    # fn __add__(
    #     self: Rank1Tensor[dtype, width],
    #     other: Rank1Tensor[dtype, width]
    # ) -> Rank1Tensor[dtype, width]:

    #     var result = Rank1Tensor[dtype, width]()

    #     @parameter
    #     fn add[nelts: Int](n: Int):
    #         result.store[nelts](n, self.load[nelts](n) + other.load[nelts](n))

    #     vectorize[add, simdwidthof[dtype]() * 2, size = width]()
    #     return result

    fn __radd__(
        self: Rank1Tensor[dtype, width],
        other: Scalar[dtype]
    ) -> Rank1Tensor[dtype, width]:
        return self + other

    # fn __radd__(
    #     self: Rank1Tensor[dtype, width],
    #     other: Rank1Tensor[dtype, width]
    # ) -> Rank1Tensor[dtype, width]:
    #     return self + other

    fn __sub__(
        self,
        other: Scalar[dtype]
    ) -> Rank1Tensor[dtype, width]:

        var result = Rank1Tensor[dtype, width]()

        @parameter
        fn sub[nelts: Int](n: Int):
            result.store[nelts](n, self.load[nelts](n) - other)

        vectorize[sub, simdwidthof[dtype]() * 2, size = width]()
        return result

    fn __sub__(
        self,
        other: Rank1Tensor[dtype, width]
    ) -> Rank1Tensor[dtype, width]:

        var result = Rank1Tensor[dtype, width]()

        @parameter
        fn sub[nelts: Int](n: Int):
            result.store[nelts](n, self.load[nelts](n) - other.load[nelts](n))

        vectorize[sub, simdwidthof[dtype]() * 2, size = width]()
        return result

    fn __rsub__(
        self: Rank1Tensor[dtype, width],
        other: Scalar[dtype]
    ) -> Rank1Tensor[dtype, width]:
        
        var result = Rank1Tensor[dtype, width]()

        @parameter
        fn rsub[nelts: Int](n: Int):
            result.store[nelts](n, other - self.load[nelts](n))

        vectorize[rsub, simdwidthof[dtype]() * 2, size = width]()
        return result

    fn __rsub__(
        self: Rank1Tensor[dtype, width],
        other: Rank1Tensor[dtype, width]
    ) -> Rank1Tensor[dtype, width]:

        var result = Rank1Tensor[dtype, width]()

        @parameter
        fn rsub[nelts: Int](n: Int):
            result.store[nelts](n, other.load[nelts](n) - self.load[nelts](n))

        vectorize[rsub, simdwidthof[dtype]() * 2, size = width]()
        return result

    fn __mul__(
        self: Rank1Tensor[dtype, width],
        other: Scalar[dtype]
    ) -> Rank1Tensor[dtype, width]:
        
        var result = Rank1Tensor[dtype, width]()

        @parameter
        fn mul[nelts: Int](n: Int):
            result.store[nelts](n, self.load[nelts](n) * other)

        vectorize[mul, simdwidthof[dtype]() * 2, size = width]()
        return result

    fn __mul__(
        self: Rank1Tensor[dtype, width],
        other: Rank1Tensor[dtype, width]
    ) -> Rank1Tensor[dtype, width]:

        var result = Rank1Tensor[dtype, width]()

        @parameter
        fn mul[nelts: Int](n: Int):
            result.store[nelts](n, self.load[nelts](n) * other.load[nelts](n))

        vectorize[mul, simdwidthof[dtype]() * 2, size = width]()
        return result

    fn __rmul__(
        self: Rank1Tensor[dtype, width],
        other: Scalar[dtype]
    ) -> Rank1Tensor[dtype, width]:
        return self * other

    fn __rmul__(
        self: Rank1Tensor[dtype, width],
        other: Rank1Tensor[dtype, width]
    ) -> Rank1Tensor[dtype, width]:
        return self * other

    fn __pow__(
        self,
        other: Scalar[dtype]
    ) -> Rank1Tensor[dtype, width]:

        var result = Rank1Tensor[dtype, width]()

        @parameter
        fn pow[nelts: Int](n: Int):
            result.store[nelts](n, self.load[nelts](n) ** other)

        vectorize[pow, simdwidthof[dtype]() * 2, size = width]()
        return result

    fn sum(inout self: Rank1Tensor[dtype, width]) -> Scalar[dtype]:
        
        var result: Scalar[dtype] = 0.0

        @parameter
        fn sum[nelts: Int](n: Int):
            result += (self.load[nelts](n).reduce_add())

        vectorize[sum, simdwidthof[dtype]() * 2, size = width]()
        return result

