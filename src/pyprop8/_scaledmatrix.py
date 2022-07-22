import numpy as np

"""
This module implements operations on exponentially-scaled matrices. This allows
stable computation of matrix-matrix and matrix-vector products in cases where elements may
be very large and/or very small.

A general matrix M is represented as (s, A) so that M = exp(s). A 

Further, this module implements computations for a 'stack' of matrices: this is simply a
structure to enable efficient evaluation of many similar calculations. If, for example

s1 = Stack{ A, B, C}   and s2 = Stack{ u, v, w}

then

s1 @ s2 = Stack{ A @ u , B @ v, C @ w}
"""


class ScaledMatrixStack:
    """
    A class to represent a collection (`stack`) of N x M matrices.
    An individual matrix A is stored in the form (s, D) such that A = exp(s).D
    where s is a scalar and D is an N x M matrix.
    """

    def __init__(
        self,
        data=None,
        scale=None,
        nStack=None,
        N=None,
        M=None,
        copy=False,
        dtypeData="float64",
        dtypeScale="float64",
    ):
        """
        Initialise a stack (collection) of exponentially scaled matrices: each matrix A
        is expressed as (s, D) where A=exp(s).D. Usage is either:

        scm = ScaledMatrixStack(data, scale, copy=False)

        to populate the stack using pre-existing data and scale, or

        scm = ScaledMatrixStack(nStack, N, M, dtypeData="float64", dtypeScale="float64")

        to create an empty (zeros) stack of a given dimension and data type.

        Inputs:
        data - array, shape (nStack, N, M): a set of nStack (N x M) matrices, {D1, D2, ...}
        scale - array, shape (nStack,): the exponential scale factor for each of the
                nStack matrices, {s1, s2, ...}. (Use scale = np.zeros(...) if `data`
                is currently unscaled.
        nStack - integer. The number of matrices in the stack.
        N, M  - integers. The dimensions of individual matrices.
        copy - True/False. If True, call .copy() on both `data` and `scale`; if False,
                use the versions as passed in.
        dtypeData - any valid `numpy.dtype` specification. The data type used for `D`.
        dtypeScale - any valid `numpy.dtype` specification. The data type used for `s`.
        """
        if data is None:
            if scale is not None:
                raise ValueError(
                    "Optional argument 'scale' cannot be provided without setting 'data'."
                )
            if nStack is None or N is None or M is None:
                raise ValueError("ScaledMatrixStack dimensions must be specified")
            self.nStack = nStack
            self.shape = (N, M)
            self.M = np.zeros([nStack, N, M], dtype=dtypeData)
            self.scale = np.zeros([nStack], dtype=dtypeScale)
        else:
            if not (nStack is None and N is None and M is None):
                raise ValueError("Do not provide dimensions in conjunction with data")
            try:
                self.nStack, N, M = data.shape
                self.shape = (N, M)
            except AttributeError:
                raise TypeError(
                    "ScaledMatrixStack 'data' argument has invalid type (%s)"
                    % type(data)
                )
            except ValueError:
                raise TypeError(
                    "ScaledMatrixStack 'data' argument appears to have wrong shape"
                )
            if copy:
                self.M = data.copy()
                if scale is None:
                    self.scale = np.zeros(self.nStack, dtype=dtypeScale)
                else:
                    try:
                        if scale.shape != (self.nStack,):
                            raise ValueError(
                                "ScaledMatrixStack 'scale' argument does not have expected shape"
                            )
                        self.scale = scale.copy()
                    except AttributeError:  # Assume it's a scalar
                        self.scale = np.full((self.nStack,), scale)
            else:
                self.M = data
                try:
                    if scale.shape != (self.nStack,):
                        raise ValueError(
                            "ScaledMatrixStack 'scale' argument does not have expected shape"
                        )
                    self.scale = scale
                except AttributeError:
                    if scale is None:
                        scale = 0.0
                    self.scale = np.full((self.nStack,), scale)
            # self.rescale()

    def copy(self, dest=None):
        """Create a copy of a ScaledMatrixStack.
        ```
        scm = ScaledMatrixStack(...)
        new = scm.copy()
        ```
        """

        if dest is None:
            return ScaledMatrixStack(self.M, self.scale, copy=True)
        else:
            np.copyto(dest.M, self.M)
            np.copyto(dest.scale, self.scale)
            return dest

    def rescale(self):
        """
        Update the entries in a ScaledMatrixStack so that each matrix D
        has unit maximum absolute value.
        """
        mx = np.abs(self.M).max((1, 2))
        mx = np.where(mx > 0, mx, 1)
        self.M /= mx.reshape(-1, 1, 1)
        self.scale += np.log(mx)

    def __getitem__(self, key):
        """
        Enable indexing.
        ```
        scm = ScaledMatrixStack(...)
        a = scm[3:7] # <--- Select examples 3-7 from the stack; return as a new stack
        b = scm[3:7, 0:2, 0:2] # <--- Select examples 3-7 from the stack; then select
                               #      only the upper 2x2 portion of each matrix and
                               #      return as new stack.
        ```
        """
        if type(key) is tuple:
            if len(key) == 3:
                return ScaledMatrixStack(
                    data=self.M[key[0], key[1], key[2]], scale=self.scale[key[0]]
                )
            else:
                raise IndexError("Please provide one or three indices")
        else:
            if type(key) is int:
                key = slice(key, key + 1, None)
            return ScaledMatrixStack(data=self.M[key, :, :], scale=self.scale[key])

    def __setitem__(self, key, value):
        """Allow updating of subsets of the stack"""
        if type(value) is ScaledMatrixStack:
            M = value.M
            s = value.scale
        elif type(value) is tuple:
            M, s = value
        else:
            M = value
            s = 0
        if type(key) is tuple:
            if len(key) == 3:
                self.M[key[0], key[1], key[2]] = value
                self.scale[key[0]] = s
            else:
                raise IndexError("Please provide one or three indices")
        else:
            self.M[key, :, :] = M
            self.scale[key] = s

    @property
    def value(self):
        """Convert the stack into a single numpy array."""
        return self.M * np.exp(self.scale).reshape(-1, 1, 1)

    def matmul(self, other, out=None):
        """Matrix multiplication between two stacks.
        ```
        a = ScaledMatrixStack(...) # Set of matrices {A1, A2, A3...}
        b = ScaledMatrixStack(...) # Set of matrices {B1, B2, B3...}

        c = a.matmul(b)  # Set of matrices {A1@B1, A2@B2, A3@B3...}
        ```
        """
        if out is None:
            return ScaledMatrixStack(self.M @ other.M, self.scale + other.scale)
        elif out is self:
            self.M = np.matmul(
                self.M, other.M, out=self.M
            )  # In-place not yet supported...
            self.scale += other.scale
            return self
        elif out is other:
            other.M = np.matmul(self.M, other.M, out=other.M)
            other.scale += self.scale
            return other
        else:
            np.matmul(self.M, other.M, out=out.M)
            np.add(self.scale, other.scale, out=out.scale)
            return out

    def add(self, other, out=None):
        """
        Matrix addition between two stacks.
        ```
        a = ScaledMatrixStack(...) # Set of matrices {A1, A2, A3...}
        b = ScaledMatrixStack(...) # Set of matrices {B1, B2, B3...}

        c = a.add(b)  # Set of matrices {A1+B1, A2+B2, A3+B3...}
        ```
        """
        if out is None:
            maxsc = np.maximum(self.scale, other.scale)
            return ScaledMatrixStack(
                (self.M * np.exp(self.scale - maxsc).reshape(-1, 1, 1))
                + (other.M * np.exp(other.scale - maxsc).reshape(-1, 1, 1)),
                maxsc,
            )
        elif out is self:
            maxsc = np.maximum(self.scale, other.scale)
            self.M *= np.exp(self.scale - maxsc).reshape(-1, 1, 1)
            self.M += other.M * np.exp(other.scale - maxsc).reshape(-1, 1, 1)
            self.scale = maxsc
            return self
        elif out is other:
            maxsc = np.maximum(self.scale, other.scale)
            other.M *= np.exp(other.scale - maxsc).reshape(-1, 1, 1)
            other.M += self.M * np.exp(self.scale - maxsc).reshape(-1, 1, 1)
            other.scale = maxsc
            return other
        else:
            maxsc = np.maximum(self.scale, other.scale, out=out.scale)
            np.multiply(self.M, np.exp(self.scale - maxsc).reshape(-1, 1, 1), out=out.M)
            np.add(
                out.M,
                other.M * np.exp(other.scale - maxsc).reshape(-1, 1, 1),
                out=out.M,
            )
            return out

    def subtract(self, other, out=None):
        """
        Matrix subtraction between two stacks.
        ```
        a = ScaledMatrixStack(...) # Set of matrices {A1, A2, A3...}
        b = ScaledMatrixStack(...) # Set of matrices {B1, B2, B3...}

        c = a.subtract(b)  # Set of matrices {A1-B1, A2-B2, A3-B3...}
        ```
        """
        if out is None:
            maxsc = np.maximum(self.scale, other.scale)
            return ScaledMatrixStack(
                (self.M * np.exp(self.scale - maxsc).reshape(-1, 1, 1))
                - (other.M * np.exp(other.scale - maxsc).reshape(-1, 1, 1)),
                maxsc,
            )
        elif out is self:
            maxsc = np.maximum(self.scale, other.scale)
            self.M *= np.exp(self.scale - maxsc).reshape(-1, 1, 1)
            self.M -= other.M * np.exp(other.scale - maxsc).reshape(-1, 1, 1)
            self.scale = maxsc
            return self
        elif out is other:
            maxsc = np.maximum(self.scale, other.scale)
            other.M *= np.exp(other.scale - maxsc).reshape(-1, 1, 1)
            np.subtract(
                self.M * np.exp(self.scale - maxsc).reshape(-1, 1, 1),
                other.M,
                out=other.M,
            )
            other.scale = maxsc
            return other
        else:
            maxsc = np.maximum(self.scale, other.scale, out=out.scale)
            np.multiply(self.M, np.exp(self.scale - maxsc).reshape(-1, 1, 1), out=out.M)
            np.subtract(
                out.M,
                other.M * np.exp(other.scale - maxsc).reshape(-1, 1, 1),
                out=out.M,
            )
            return out

    def multiply(self, other, out=None):
        """
        Element-wise multiplication between two stacks.
        ```
        a = ScaledMatrixStack(...) # Set of matrices {A1, A2, A3...}
        b = ScaledMatrixStack(...) # Set of matrices {B1, B2, B3...}

        c = a.multiply(b)  # Set of matrices {A1*B1, A2*B2, A3*B3...}
        ```
        """
        if type(other) is float or type(other) is int:
            return self.scalarMultiply(other, out)
        if out is None:
            return ScaledMatrixStack(self.M * other.M, self.scale + other.scale)
        elif out is self:
            self.M *= other.M
            self.scale += other.scale
            return self
        elif out is other:
            other.M *= self.M
            other.scale += self.scale
            return other
        else:
            np.multiply(self.M, other.M, out=out.M)
            np.add(self.scale, other.scale, out=out.scale)
            return out

    def scalarMultiply(self, other, out=None):
        """
        Multiply stack by a scalar.
        """
        if out is None:
            return ScaledMatrixStack(self.M.copy(), self.scale + np.log(other))
        elif out is self:
            self.scale += np.log(other)
            return self
        else:
            np.copyto(out.M, self.M)
            np.add(self.scale, np.log(other), out.scale)
            return out

    def divide(self, other, out=None):
        """
        Element-wise division between two stacks.
        ```
        a = ScaledMatrixStack(...) # Set of matrices {A1, A2, A3...}
        b = ScaledMatrixStack(...) # Set of matrices {B1, B2, B3...}

        c = a.multiply(b)  # Set of matrices {A1/B1, A2/B2, A3/B3...}
        ```
        """
        if type(other) is float or type(other) is int:
            return self.scalarMultiply(self, 1 / other, out)
        if out is None:
            return ScaledMatrixStack(self.M / other.M, self.scale - other.scale)
        elif out is self:
            self.M /= other.M
            self.scale -= other.scale
            return self
        else:
            np.divide(self.M, other.M, out=out.M)
            np.subtract(self.scale, other.scale, out=out.scale)
            return out

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __matmul__(self, other):
        return self.matmul(other)

    def __mul__(self, other):
        return self.multiply(other)

    def __truediv__(self, other):
        return self.divide(other)

    def __imatmul__(self, other):
        return self.matmul(other, out=self)

    def __imul__(self, other):
        return self.multiply(other, out=self)

    def __iadd__(self, other):
        return self.add(other, out=self)

    def __isub__(self, other):
        return self.subtract(other, out=self)

    def __itruediv__(self, other):
        return self.divide(other, out=self)

    def __len__(self):
        return self.nStack


def empty_like(s):
    return ScaledMatrixStack(
        nStack=s.nStack,
        N=s.shape[0],
        M=s.shape[1],
        dtypeData=s.M.dtype,
        dypeScale=s.scale.dtype,
    )
