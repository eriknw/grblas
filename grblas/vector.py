import types
from functools import partial
from .base import lib, ffi, NULL, GbContainer, GbDelayed
from .scalar import Scalar
from .ops import (OpBase, UnaryOp, BinaryOp, Monoid, Semiring,
                  find_opclass, find_return_type)
from . import dtypes
from .exceptions import check_status, is_error, NoValue


class Vector(GbContainer):
    """
    GraphBLAS Sparse Vector
    High-level wrapper around GrB_Vector type
    """
    can_mask = True

    def __init__(self, gb_obj, dtype):
        super().__init__(gb_obj, dtype)
        self.element = Vector.ElementManipulator(self)
        self.extract = Vector.Extractor(self)
        self.assign = Vector.Assigner(self)

    def __del__(self):
        check_status(lib.GrB_Vector_free(self.gb_obj))
    
    def __repr__(self):
        return f'<Vector {self.nvals}/{self.size}:{self.dtype.name}>'

    def __eq__(self, other):
        # Borrowed this recipe from LAGraph
        if type(other) != self.__class__:
            return False
        if self.dtype != other.dtype:
            return False
        if self.size != other.size:
            return False
        if self.nvals != other.nvals:
            return False
        # Use ewise_mult to compare equality via intersection
        matches = Vector.new_from_type(bool, self.size)
        matches[:] = self.ewise_mult(other, BinaryOp.EQ)
        if matches.nvals != self.nvals:
            return False
        # Check if all results are True
        result = Scalar.new_from_type(bool)
        result[:] = matches.reduce(Monoid.LAND)
        return result.value

    def __len__(self):
        return self.nvals

    @property
    def size(self):
        n = ffi.new('GrB_Index*')
        check_status(lib.GrB_Vector_size(n, self.gb_obj[0]))
        return n[0]

    @property
    def shape(self):
        return (self.size,)

    @property
    def nvals(self):
        n = ffi.new('GrB_Index*')
        check_status(lib.GrB_Vector_nvals(n, self.gb_obj[0]))
        return n[0]

    def clear(self):
        check_status(lib.GrB_Vector_clear(self.gb_obj[0]))
    
    def resize(self, size):
        raise NotImplementedError('Not implemented in GraphBLAS 1.2')
        check_status(lib.GrB_Vector_resize(self.gb_obj[0], size))

    def to_values(self):
        """
        GrB_Vector_extractTuples
        Extract the indices and values as 2 generators
        """
        indices = ffi.new('GrB_Index[]', self.nvals)
        values = ffi.new(f'{self.dtype.c_type}[]', self.nvals)
        n = ffi.new('GrB_Index*')
        n[0] = self.nvals
        func = getattr(lib, f'GrB_Vector_extractTuples_{self.dtype.name}')
        check_status(func(
            indices,
            values,
            n,
            self.gb_obj[0]))
        return (iter(indices), iter(values))

    def rebuild_from_values(self, indices, values, *, dup_op=NULL):
        # TODO: add `size` option once .resize is available
        self.clear()
        if not isinstance(indices, (tuple, list)):
            indices = tuple(indices)
        if not isinstance(values, (tuple, list)):
            values = tuple(values)
        if len(indices) != len(values):
            raise ValueError(f'`indices` and `values` have different lengths '
                             f'{len(indices)} != {len(values)}')
        n = len(indices)
        if n <= 0:
            return
        dup_orig = dup_op
        if dup_op is NULL:
            dup_op = BinaryOp.PLUS
        if isinstance(dup_op, BinaryOp):
            dup_op = dup_op[self.dtype]
        indices = ffi.new('GrB_Index[]', indices)
        values = ffi.new(f'{self.dtype.c_type}[]', values)
        # Push values into w
        func = getattr(lib, f'GrB_Vector_build_{self.dtype.name}')
        check_status(func(
            self.gb_obj[0],
            indices,
            values,
            n,
            dup_op))
        # Check for duplicates when dup_op was not provided
        if dup_orig is NULL and self.nvals < len(values):
            raise ValueError('Duplicate indices found, must provide `dup_op` BinaryOp') 

    @classmethod
    def new_from_type(cls, dtype, size=0):
        """
        GrB_Vector_new
        Create a new empty Vector from the given type and size
        """
        new_vector = ffi.new('GrB_Vector*')
        dtype = dtypes.lookup(dtype)
        check_status(lib.GrB_Vector_new(new_vector, dtype.gb_type, size))
        return cls(new_vector, dtype)

    @classmethod
    def new_from_existing(cls, vector):
        """
        GrB_Vector_dup
        Create a new Vector by duplicating an existing one
        """
        new_vec = ffi.new('GrB_Vector*')
        check_status(lib.GrB_Vector_dup(new_vec, vector.gb_obj[0]))
        return cls(new_vec, vector.dtype)

    @classmethod
    def new_from_values(cls, indices, values, *, size=None, dup_op=NULL, dtype=None):
        """Create a new Vector from the given lists of indices and values.  If
        size is not provided, it is computed from the max index found.
        """
        if not isinstance(indices, (tuple, list)):
            indices = tuple(indices)
        if not isinstance(values, (tuple, list)):
            values = tuple(values)
        if len(values) <= 0:
            raise ValueError('No values provided. Unable to determine type.')
        if dtype is None:
            # Find dtype from any of the values (assumption is they are the same type)
            dtype = type(values[0])
        dtype = dtypes.lookup(dtype)
        # Compute size if not provided
        if size is None:
            if not indices:
                raise ValueError('No indices provided. Unable to infer size.')
            size = max(indices) + 1
        # Create the new vector
        w = cls.new_from_type(dtype, size)
        # Add the data
        w.rebuild_from_values(indices, values, dup_op=dup_op)
        return w

    #########################################################
    # Delayed methods
    # 
    # These return a GbDelayed object which must be passed
    # to __setitem__ to trigger a call to GraphBLAS
    #########################################################

    def ewise_add(self, other, op=NULL):
        """
        GrB_eWiseAdd_Vector

        Result will contain the union of indices from both Vectors
        """
        if not isinstance(other, Vector):
            raise TypeError(f'Expected Vector, found {type(other)}')
        if op is NULL:
            op = BinaryOp.PLUS
        opclass = find_opclass(op)
        if opclass not in ('BinaryOp', 'Monoid', 'Semiring'):
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        func = getattr(lib, f'GrB_eWiseAdd_Vector_{opclass}')
        output_constructor = partial(Vector.new_from_type,
                                     dtype=find_return_type(op, self.dtype, other.dtype),
                                     size=self.size)
        return GbDelayed(func,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         output_constructor=output_constructor)

    def ewise_mult(self, other, op=NULL):
        """
        GrB_eWiseMult_Vector

        Result will contain the intersection of indices from both Vectors
        """
        if not isinstance(other, Vector):
            raise TypeError(f'Expected Vector, found {type(other)}')
        if op is NULL:
            op = BinaryOp.TIMES
        opclass = find_opclass(op)
        if opclass not in ('BinaryOp', 'Monoid', 'Semiring'):
            raise TypeError(f'op must be BinaryOp, Monoid, or Semiring')
        func = getattr(lib, f'GrB_eWiseMult_Vector_{opclass}')
        output_constructor = partial(Vector.new_from_type,
                                     dtype=find_return_type(op, self.dtype, other.dtype),
                                     size=self.size)
        return GbDelayed(func,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         output_constructor=output_constructor)

    def vxm(self, other, op=NULL):
        """
        GrB_vxm
        Vector-Matrix multiplication. Result is a Vector.
        """
        from .matrix import Matrix
        if not isinstance(other, Matrix):
            raise TypeError(f'Expected Matrix, found {type(other)}')
        if op is NULL:
            op = Semiring.PLUS_TIMES
        opclass = find_opclass(op)
        if opclass != 'Semiring':
            raise TypeError(f'op must be Semiring')
        output_constructor = partial(Vector.new_from_type,
                                     dtype=find_return_type(op, self.dtype, other.dtype),
                                     size=other.ncols)
        return GbDelayed(lib.GrB_vxm,
                         [op, self.gb_obj[0], other.gb_obj[0]],
                         bt=other.is_transposed,
                         output_constructor=output_constructor)

    def apply(self, op, left=None, right=None):
        """
        GrB_Vector_apply
        Apply UnaryOp to each element of the calling Vector
        A BinaryOp can also be applied if a scalar is passed in as `left` or `right`,
            effectively converting a BinaryOp into a UnaryOp
        """
        opclass = find_opclass(op)
        if opclass == 'UnaryOp':
            if left is not None or right is not None:
                raise TypeError('Cannot provide `left` or `right` for a UnaryOp')
        elif opclass == 'BinaryOp':
            if left is None and right is None:
                raise TypeError('Must provide either `left` or `right` for a BinaryOp')
            elif left is not None and right is not None:
                raise TypeError('Cannot provide both `left` and `right`')
        else:
            raise TypeError('apply only accepts UnaryOp or BinaryOp')
        output_constructor = partial(Vector.new_from_type,
                                     dtype=find_return_type(op, self.dtype),
                                     size=self.size)
        if opclass == 'UnaryOp':
            return GbDelayed(lib.GrB_Vector_apply,
                             [op, self.gb_obj[0]],
                             output_constructor=output_constructor)
        else:
            raise NotImplementedError('apply with BinaryOp not available in GraphBLAS 1.2')
            # TODO: fill this in once function is available

    def reduce(self, op=NULL):
        """
        GrB_Vector_reduce
        Reduce all values into a scalar
        """
        if op is NULL:
            if self.dtype == bool:
                op = Monoid.LOR
            else:
                op = Monoid.PLUS
        func = getattr(lib, f'GrB_Vector_reduce_{self.dtype.name}')
        output_constructor = partial(Scalar.new_from_type,
                                     dtype=find_return_type(op, self.dtype))
        return GbDelayed(func,
                        [op, self.gb_obj[0]],
                        output_constructor=output_constructor)

    class ElementManipulator:
        def __init__(self, vector):
            self._vector = vector
        
        def __repr__(self):
            return 'VectorElementManipulator'
        
        def _parse_index(self, index):
            if not isinstance(index, int):
                raise TypeError('Index must be an int')
            size = self._vector.size
            if index >= size:
                raise IndexError(f'index={index}, size={size}')
            return index

        def __getitem__(self, index):
            index = self._parse_index(index)
            vec = self._vector
            func = getattr(lib, f'GrB_Vector_extractElement_{vec.dtype}')
            result = ffi.new(f'{vec.dtype.c_type}*')

            err_code = func(result,
                            vec.gb_obj[0],
                            index)
            # Don't raise error for no value, simply return `None`
            if is_error(err_code, NoValue):
                return None
            check_status(err_code)
            return result[0]
        
        def __setitem__(self, index, value):
            index = self._parse_index(index)
            vec = self._vector
            func = getattr(lib, f'GrB_Vector_setElement_{vec.dtype}')
            check_status(func(
                vec.gb_obj[0],
                ffi.cast(vec.dtype.c_type, value),
                index))

        def __delitem__(self, index):
            index = self._parse_index(index)
            raise NotImplementedError('Not available in GraphBLAS 1.2')

    class _Indexer:
        def _parse_index(self, index):
            typ = type(index)
            if typ == int:
                # Single index
                return index, None
            if typ == tuple:
                raise TypeError(f'{self} cannot accept a tuple as index; use slice or list')
            if typ == slice:
                if index == slice(None):
                    # [:] means all indices; use special GrB_ALL indicator
                    return lib.GrB_ALL, self._vector.size
                index = tuple(range(self._vector.size)[index])
            elif typ != list:
                try:
                    index = tuple(index)
                except Exception:
                    raise TypeError()
            return ffi.new('GrB_Index[]', index), len(index)

    class Extractor(_Indexer):
        def __init__(self, vector):
            self._vector = vector
        
        def __repr__(self):
            return 'VectorExtractor'
        
        def __getitem__(self, index):
            index, isize = self._parse_index(index)
            if isize is None:
                raise TypeError('Use v.element[i] to get a single element')
            output_constructor = partial(Vector.new_from_type,
                                         dtype=self._vector.dtype,
                                         size=isize)
            return GbDelayed(lib.GrB_Vector_extract,
                            [self._vector.gb_obj[0], index, isize],
                            output_constructor=output_constructor)
    
    class Assigner(_Indexer):
        def __init__(self, vector):
            self._vector = vector
        
        def __repr__(self):
            return 'VectorAssigner'
        
        def __setitem__(self, keys, other):
            # Note: keys contains assignment index, mask, accum, and REPLACE
            #       index must always come first (if given, otherwise assumes all indexes)
            #       v.assign[:] will be interpreted as all indexes with no mask
            if type(keys) != tuple:
                keys = (keys,)
            try:
                # Try to parse the first item as an index
                index, isize = self._parse_index(keys[0])
                keys = keys[1:]
                if isize is None:
                    # We have a single index; convert to a len=1 list
                    index = [index]
                    isize = 1
            except TypeError:
                # No index given; use the default
                index, isize = self._parse_index(slice(None))
            if not keys:
                # No keys given; use the default
                keys = slice(None)
            else:
                for key in keys:
                    if type(key) in (list, slice):
                        raise TypeError('Assignment indexes must come first')

            if isinstance(other, Scalar):
                other = other.value

            if isinstance(other, (int, float, bool)):
                dtype = self._vector.dtype
                func = getattr(lib, f'GrB_Vector_assign_{dtype.name}')
                scalar = ffi.cast(dtype.c_type, other)
                dval = GbDelayed(func,
                                 [scalar, index, isize])
            elif isinstance(other, Vector):
                dval = GbDelayed(lib.GrB_Vector_assign,
                                 [other.gb_obj[0], index, isize])
            else:
                raise TypeError(f'Unexpected type for assignment value: {type(other)}')
            # Forward the __setitem__ call so it is resolved with mask and accum
            self._vector[keys] = dval
