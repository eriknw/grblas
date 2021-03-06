# grblas
Python wrapper around GraphBLAS

To install, `conda install grblas -c jim22k`. This will also pull in the SuiteSparse `ss_graphblas` compiled C library.

Currently works with SuiteSparse:GraphBLAS, but the goal is to make it work with all implementations of the GraphBLAS spec.
Specifically, GraphBLAST is a high priority target, but no work has been done yet to build the connector.

The approach taken with this library is to follow the C-API specification as closely as possible while making improvements 
allowed with the Python syntax. Because the spec always passes in the output object to be written to, we follow the same, 
which is very different from the way Python normally operates. In fact, many who are familiar with other Python data 
libraries (numpy, pandas, etc) will find it strange to not create new objects for every call.

At the highest level, the approach is to separate output, mask, and accumulator on the left side of the assignment 
operator (=) and put the computation on the right side.

This is an example of how the mapping works:<br>
C call: `GrB_Matrix_mxm(M, mask, accum, semiring, A, B, desc=NULL)`<br>
Python call: `M[mask, accum] = A.mxm(B, semiring)`<br>
_where_
 - accum is `GrB_PLUS_INT64` (in C) and `BinaryOp.PLUS` (in Python)
 - semiring is `GrB_MIN_PLUS_INT64` (in C) and `Semiring.MIN_PLUS` (in Python)

The expression on the right `A.mxm(B)` creates a delayed object which does no computation. Once it is received as 
the value of the `__setitem__` call on `M`, the whole thing is translated into the equivalent GraphBLAS call.

As a convenience, delayed objects have a `.new()` method which can be used to force computation and return a new 
object. This is convenient for quickly writing code, but may create many unnecessary objects if used in a loop. It
also loses the ability to perform accumulation with existing results. For best performance, following the standard 
GraphBLAS approach of (1) creating the object outside the loop and (2) using the object repeatedly within each loop 
is a much better approach, even if it doesn't feel very Pythonic. 

Descriptor flags are set on the appropriate elements to keep logic close to what it affects. Here is the same call 
with descriptor bits set. `ttcr` indicates transpose the first and second matrices, complement the mask, and do a 
replacement on the output.

C call: `GrB_Matrix_mxm(M, mask, accum, A, B, semiring, desc.ttcr)`<br>
Python call: `M[~mask, accum, REPLACE] = A.T.mxm(B.T, semiring)`

The objects receiving the flag operations (A.T, ~mask, etc) are also delayed objects. They hold on to the state but 
do no computation, allowing the correct descriptor bits to be set in a single GraphBLAS call.

**If no mask or accumulator is used, the call looks like this**:<br>
`M[:] = A.mxm(B, semiring)`

Python doesn't allow `__setitem__` on an empty key, so we use the empty slice to indicate "applies to all elements", 
i.e. there is no mask.

# Operations
 - mxm: `M[mask, accum] = A.mxm(B, semiring)`
 - mxv: `w[mask, accum] = A.mxv(v, semiring)`
 - vxm: `w[mask, accum] = v.vxm(B, semiring)`
 - eWiseAdd: `M[mask, accum] = A.ewise_add(B, binaryop)`
 - eWiseMult: `M[mask, accum] = A.ewise_mult(B, binaryop)`
 - extract: 
   + `M[mask, accum] = A.extract[rows, cols]`  # rows and cols are a list or a slice
   + `w[mask, accum] = A.extract[rows, col_index]`  # extract column
   + `w[mask, accum] = A.extract[row_index, cols]`  # extract row
 - assign:
   + `M.assign[rows, cols, mask, accum] = A`  # rows and cols are a list or a slice
   + `M.assign[rows, col_index, mask, accum] = v`  # assign column
   + `M.assign[row_index, cols, mask, accum] = v`  # assign row
   + `M.assign[rows, cols, mask, accum] = s`  # assign scalar
 - apply:
   + `M[mask, accum] = A.apply(unaryop)`
   + `M[mask, accum] = A.apply(binaryop, left=s)`  # bind-first
   + `M[mask, accum] = A.apply(binaryop, right=s)`  # bind-second
 - reduce: 
   + `v[mask, accum] = A.reduce_rows(op)`  # reduce row-wise
   + `v[mask, accum] = A.reduce_columns(op)`  # reduce column-wise
   + `s[accum] = A.reduce_scalar(op)`
   + `s[accum] = v.reduce(op)`
 - transpose: `M[mask, accum] = A.T`
 - kronecker: `M[mask, accum] = A.kronecker(B, binaryop)`
 - elementAssign: `M.element[i, j] = s`
 - elementExtract: `s = M.element[i, j]`

# Creating new Vectors / Matrices
 - new_type: `A = Matrix.new_from_type(dtype, num_rows, num_cols)`
 - dup: `B = Matrix.new_from_existing(A)`
 - build: `A = Matrix.new_from_values([row_indices], [col_indices], [values])`
 - new from delayed:
   - Delayed objects can be used to create a new object using `.new()` method
   - `C = A.mxm(B, semiring).new()`

# Properties
 - size: `size = v.size`
 - nrows: `nrows = M.nrows`
 - ncols: `ncols = M.ncols`
 - nvals: `nvals = M.nvals`
 - extractTuples: `rindices, cindices, vals = M.to_values()`

# Initialization
Because we (will) support multiple GraphBLAS implementations (SuiteSparse, GraphBLAST, etc), users must initialize 
grblas prior to importing anything other than the top-level grblas itself.

```
import grblas
grblas.init('suitesparse')
# Now we can import other items from grblas
from grblas.ops import BinaryOp, UnaryOp
from grblas import Matrix, Vector, Scalar
```

# Performant User Defined Functions
`grblas` requires `numba` which enables compiling user-defined Python functions to native C for use in GraphBLAS.

Example customized UnaryOp:
```
def force_odd(x):
    if x % 2 == 0:
        return x + 1
    return x

UnaryOp.register_new('force_odd', force_odd)

v = Vector.new_from_values([0,1,3], [1,2,3])
w = v.apply(UnaryOp.force_odd).new()
w  # indexes=[0,1,3], values=[1,3,3]
```

Similar methods exist for BinaryOp, Monoid, and Semiring.

# Import/Export connectors to the Python ecosystem
`grblas.io` contains functions for converting to and from:
- numpy arrays and matrices
  - `from_numpy(m)`  (_1-D array becomes Vector, 2-D array or matrix becomes Matrix_)
  - `to_numpy(g, format='array')`
- scipy.sparse matrices
  - `from_scipy_sparse_matrix(m)`
  - `to_scipy_sparse_matrix(m, format='csr')`
- networkx graphs
  - `from_networkx(g)`
  - `to_networkx(g)`

# Attribution
This library borrows some great ideas from [pygraphblas](https://github.com/michelp/pygraphblas),
especially around parsing operator names from SuiteSparse and the concept of a Scalar which the backend
implementation doesn't need to know about.

