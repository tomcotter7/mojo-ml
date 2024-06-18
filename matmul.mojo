# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# This sample demonstrates how various systems optimizations can be applied to a
# naive matmul implementation in Mojo to gain significant performance speedups

from random import rand

import benchmark
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from algorithm import parallelize, vectorize
from sys import info
from memory import memset_zero
from python import Python

alias M = 1  # rows of A and C
alias N = 9  # cols of B and C
alias K = 9 # cols of A and rows of B
alias type = DType.float32

# simdwidth of = amount of `type` elements that fit into a single SIMD register
# 2x multiplier will use multiple SIMD registers in parallel where possible
alias nelts = simdwidthof[type]() * 2
alias tile_n = 64  # N must be a multiple of this
alias tile_k = 4  # K must be a multiple of this


struct Matrix[rows: Int, cols: Int]:
    var data: DTypePointer[type]

    # Initialize zeroeing all values
    fn __init__(inout self):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: DTypePointer[type]):
        self.data = data

    ## Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(inout self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)


def run_matmul_python() -> Float64:
    Python.add_to_path(".")
    var pymatmul: PythonObject = Python.import_module("pymatmul")
    var py = Python.import_module("builtins")

    var gflops = pymatmul.benchmark_matmul_python(128, 128, 128).to_float64()
    py.print(py.str("{:<13}{:>8.3f} GFLOPS").format("Python:", gflops))

    return gflops


def run_matmul_numpy() -> Float64:
    var pymatmul: PythonObject = Python.import_module("pymatmul")
    var py = Python.import_module("builtins")

    var gflops = pymatmul.benchmark_matmul_numpy(M, N, K).to_float64()
    py.print(py.str("{:<13}{:>8.3f} GFLOPS").format("Numpy:", gflops))

    return gflops


fn matmul_naive(inout C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


# Using stdlib vectorize function
fn matmul_vectorized(inout C: Matrix, A: Matrix, B: Matrix):

    print(C.cols)
    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                print("Nelts:", nelts)
                print("N: ", n)
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts, size = C.cols]()





fn main() raises:

    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()
    var C = Matrix[M, N]()
    
    print(C[0, 0])
    matmul_vectorized(C, A, B)
    print(C[0, 0])




    
