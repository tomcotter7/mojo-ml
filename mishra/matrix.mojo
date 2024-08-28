from algorithm import vectorize

struct Matrix[dtype: DType = DType.float64](Stringable, CollectionElement, Sized):

    var _matPtr: DTypePointer[dtype]
    var _rows: Int
    var _cols: Int
    alias simd_width = 4 * simdwidthof[dtype]()

    fn __init__(inout self, rows: Int, cols: Int):
        
        self._rows = rows
        self._cols = cols
        self._matPtr = DTypePointer[dtype].alloc(rows * cols)
        memset_zero[dtype](self._matPtr, self._rows * self._cols)

    @always_inline
    fn __copyinit__(inout self, other: Self):
        
        self._rows = other._rows
        self._cols = other._cols
        self._matPtr = DTypePointer[dtype].alloc(self._rows * self._cols)
        memcpy(self._matPtr, other._matPtr, self._rows * self._cols)

    fn __moveinit__(inout self, owned existing: Self):
        
        self._matPtr = existing._matPtr
        self._rows = existing._rows
        self._cols = existing._cols

        existing._rows = 0
        existing._cols = 0
        existing._matPtr = DTypePointer[dtype]()

    fn __del__(owned self):
        self._matPtr.free()

    fn __setitem__(inout self, row: Int, col: Int, value: Scalar[dtype]) raises:

        if row < 0 or row >= self._rows or col < 0 or col >= self._cols:
            raise "Index out of bounds"
        
        self._matPtr[row * self._cols + col] = value

    fn __getitem__(self, row: Int, col: Int) raises -> Scalar[dtype]:

        if row < 0 or row >= self._rows or col < 0 or col >= self._cols:
            raise "Index out of bounds"

        return self._matPtr[row * self._cols + col]

    fn __add__(self, other: Scalar[dtype]) -> Self:

        var newMat = Matrix[dtype](self._rows, self._cols)
        
        @parameter
        fn add_scalar[simd_width: Int](idx: Int) -> None:
            newMat._matPtr[idx] = self._matPtr[idx] + other

        vectorize[add_scalar, self.simd_width](len(newMat))
        return newMat^

    fn __add__(self, other: Self) raises -> Self:

        if self._rows != other._rows or self._cols != other._cols:
            raise "Matrix dimensions do not match"

        var newMat = Matrix[dtype](self._rows, self._cols)

        @parameter
        fn add_matrix[simd_width: Int](idx: Int) -> None:
            newMat._matPtr[idx] = self._matPtr[idx] + other._matPtr[idx]

        vectorize[add_matrix, self.simd_width](len(newMat))
        return newMat^

    fn __sub__(self, other: Scalar[dtype]) -> Self:
        
        var newMat = Matrix[dtype](self._rows, self._cols)

        @parameter
        fn sub_scalar[simd_width: Int](idx: Int) -> None:
            newMat._matPtr[idx] = self._matPtr[idx] - other

        vectorize[sub_scalar, self.simd_width](len(newMat))

        return newMat^

    fn __sub__(self, other: Self) raises -> Self:

        if self._rows != other._rows or self._cols != other._cols:
            raise "Matrix dimension do not match"
        
        var newMat = Matrix[dtype](self._rows, self._cols)

        @parameter
        fn sub_matrix[simd_width: Int](idx: Int) -> None:
            newMat._matPtr[idx] = self._matPtr[idx] - other._matPtr[idx]
        vectorize[sub_matrix, self.simd_width](len(newMat))

        return newMat^

    fn __mul__(self, other: Scalar[dtype]) -> Self:

        var newMat = Matrix[dtype](self._rows, self._cols)

        @parameter
        fn mul_scalar[simd_width: Int](idx: Int) -> None:
            newMat._matPtr[idx] = self._matPtr[idx] * other

        vectorize[mul_scalar, self.simd_width](len(newMat))
        return newMat^

    fn __mul__(self, other: Self) raises -> Self:
        
        if self._rows != other._rows or self._cols != other._cols:
            raise "Matrix dimensions do not match"

        var newMat = Matrix[dtype](self._rows, other._cols)

        @parameter
        fn mul_matrix[simd_width: Int](idx: Int) -> None:
            newMat._matPtr[idx] = self._matPtr[idx] * other._matPtr[idx]

        vectorize[mul_matrix, self.simd_width](len(newMat))
        return newMat^


    fn __matmul__(self, other: Self) raises -> Self:

        if self._cols != other._rows:
            raise "Matrix dimensions do not match"

        var newMat = Matrix[dtype](self._rows, other._cols)
        
        for i in range(self._rows):
            for j in range(other._cols):
                @parameter
                fn dot_product[simd_width: Int](k: Int) -> None:
                    var x = self._matPtr[i * self._cols + k]
                    var y = other._matPtr[k * other._cols + j]
                    newMat._matPtr[i * newMat._cols + j] +=  x * y
                vectorize[dot_product, self.simd_width](self._cols)

        return newMat^

    fn __pow__(self, other: Scalar[dtype]) -> Self:
        
        var newMat = Matrix[dtype](self._rows, self._cols)

        @parameter
        fn pow_scalar[simd_width: Int](idx: Int) -> None:
            newMat._matPtr[idx] = self._matPtr[idx] ** other

        vectorize[pow_scalar, self.simd_width](len(newMat))
        return newMat^

    fn __truediv__(self, other: Scalar[dtype]) -> Self:
        
        var newMat = Matrix[dtype](self._rows, self._cols)

        @parameter
        fn div_scalar[simd_width: Int](idx: Int) -> None:
            newMat._matPtr[idx] = self._matPtr[idx] / other

        vectorize[div_scalar, self.simd_width](len(newMat))
        return newMat^
        
    fn __len__(self) -> Int:
        
        return self._rows * self._cols

    fn __str__(self) -> String:
        
        var s: String = "["

        for i in range(self._rows):
            s += "["
            for j in range(self._cols):
                try:
                    s += String(self[i, j])
                except e:
                    s += "?"
                if j < self._cols - 1:
                    s += ", "
            if i < self._rows - 1:
                s += "],"
            else:
                s += "]"
        
        s += "]"
        return s

    fn T(self) raises -> Self:
        
        var newMat = Matrix[dtype](self._cols, self._rows)

        for i in range(self._rows):

            @parameter
            fn t_mat[simd_width: Int](j: Int) -> None:
                newMat._matPtr[j * newMat._cols + i] = self._matPtr[i * self._cols + j]

            vectorize[t_mat, self.simd_width](self._cols)

        return newMat^
    
    fn sum(self) raises -> Scalar[dtype]:
        
        var sum : Scalar[dtype] = 0.0

        for i in range(self._rows * self._cols):
            sum += self._matPtr[i]

        return sum


