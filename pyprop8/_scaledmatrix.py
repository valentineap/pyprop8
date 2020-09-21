import numpy as np

class ScaledMatrixStack:
    def __init__(self,data=None,scale=None,nStack=None,N=None,M=None,name=None,copy=False,dtypeData='float64',dtypeScale='float64'):
        self.name = name
        if data is None:
            if scale is not None: raise ValueError("Optional argument 'scale' cannot be provided without setting 'data'.")
            if nStack is None or N is None or M is None: raise ValueError("ScaledMatrixStack dimensions must be specified")
            self.nStack = nStack
            self.shape = (N,M)
            self.M = np.zeros([nStack,N,M],dtype=dtypeData)
            self.scale = np.zeros([nStack],dtype=dtypeScale)
        else:
            if not (nStack is None and N is None and M is None): raise ValueError("Do not provide dimensions in conjunction with data")
            try:
                self.nStack,N,M = data.shape
                self.shape = (N,M)
            except AttributeError:
                raise TypeError("ScaledMatrixStack 'data' argument has invalid type (%s)"%type(data))
            except ValueError:
                raise TypeError("ScaledMatrixStack 'data' argument appears to have wrong shape")
            if copy:
                self.M = data.copy()
                if scale is None:
                    self.scale = np.zeros(self.nStack,dtype=dtypeScale)
                else:
                    try:
                        if scale.shape!=(self.nStack,):
                            raise ValueError("ScaledMatrixStack 'scale' argument does not have expected shape")
                        self.scale = scale.copy()
                    except AttributeError: # Assume it's a scalar
                        self.scale = np.full((self.nStack,),scale)
            else:
                self.M = data
                try:
                    if scale.shape!=(self.nStack,):
                        raise ValueError("ScaledMatrixStack 'scale' argument does not have expected shape")
                    self.scale = scale
                except AttributeError:
                    if scale is None: scale = 0.
                    self.scale = np.full((self.nStack,),scale)
            #self.rescale()
    def copy(self,dest = None):
        if dest is None:
            return ScaledMatrixStack(self.M,self.scale,copy=True)
        else:
            np.copyto(dest.M,self.M)
            np.copyto(dest.scale,self.scale)
            return dest
    def rescale(self):
        mx = np.abs(self.M).max((1,2))
        mx = np.where(mx>0,mx,1)
        self.M/=mx.reshape(-1,1,1)
        self.scale+=np.log(mx)
    def __getitem__(self,key):
        if type(key) is tuple:
            if len(key)==3:
                return ScaledMatrixStack(data = self.M[key[0],key[1],key[2]],scale = self.scale[key[0]])
            else:
                raise IndexError("Please provide one or three indices")
        else:
            if type(key) is int: key = slice(key,key+1,None)
            return ScaledMatrixStack(data = self.M[key,:,:],scale = self.scale[key])
    def __setitem__(self,key,value):
        if type(value) is ScaledMatrixStack:
            M = value.M
            s = value.scale
        elif type(value) is tuple:
            M,s = value
        else:
            M = value
            s = 0
        if type(key) is tuple:
            if len(key)==3:
                self.M[key[0],key[1],key[2]] = value
                self.scale[key[0]] = s
            else:
                raise IndexError("Please provide one or three indices")
        else:
            self.M[key,:,:] = M
            self.scale[key] = s
    @property
    def value(self):
        return self.M*np.exp(self.scale).reshape(-1,1,1)
    def matmul(self,other,out=None):
        if out is None:
            return ScaledMatrixStack(self.M@other.M,self.scale+other.scale)
        elif out is self:
            self.M=np.matmul(self.M,other.M,out=self.M) #In-place not yet supported...
            self.scale+=other.scale
            return self
        elif out is other:
            other.M = np.matmul(self.M,other.M,out=other.M)
            other.scale+=self.scale
            return other
        else:
            np.matmul(self.M,other.M,out=out.M)
            np.add(self.scale,other.scale,out=out.scale)
            return out
    def add(self,other,out=None):
        if out is None:
            maxsc = np.maximum(self.scale,other.scale)
            return ScaledMatrixStack((self.M*np.exp(self.scale-maxsc).reshape(-1,1,1))+(other.M*np.exp(other.scale-maxsc).reshape(-1,1,1)),maxsc)
        elif out is self:
            maxsc = np.maximum(self.scale,other.scale)
            self.M*=np.exp(self.scale-maxsc).reshape(-1,1,1)
            self.M+=(other.M*np.exp(other.scale-maxsc).reshape(-1,1,1))
            self.scale = maxsc
            return self
        elif out is other:
            maxsc = np.maximum(self.scale,other.scale)
            other.M*=np.exp(other.scale-maxsc).reshape(-1,1,1)
            other.M+=(self.M*np.exp(self.scale-maxsc).reshape(-1,1,1))
            other.scale = maxsc
            return other
        else:
            maxsc = np.maximum(self.scale,other.scale,out=out.scale)
            np.multiply(self.M,np.exp(self.scale-maxsc).reshape(-1,1,1),out=out.M)
            np.add(out.M,other.M*np.exp(other.scale-maxsc).reshape(-1,1,1),out=out.M)
            return out
    def subtract(self,other,out=None):
        if out is None:
            maxsc = np.maximum(self.scale,other.scale)
            return ScaledMatrixStack((self.M*np.exp(self.scale-maxsc).reshape(-1,1,1))-(other.M*np.exp(other.scale-maxsc).reshape(-1,1,1)),maxsc)
        elif out is self:
            maxsc = np.maximum(self.scale,other.scale)
            self.M*=np.exp(self.scale-maxsc).reshape(-1,1,1)
            self.M-=(other.M*np.exp(other.scale-maxsc).reshape(-1,1,1))
            self.scale = maxsc
            return self
        elif out is other:
            maxsc = np.maximum(self.scale,other.scale)
            other.M*=np.exp(other.scale-maxsc).reshape(-1,1,1)
            np.subtract(self.M*np.exp(self.scale-maxsc).reshape(-1,1,1),other.M,out=other.M)
            other.scale = maxsc
            return other
        else:
            maxsc = np.maximum(self.scale,other.scale,out=out.scale)
            np.multiply(self.M,np.exp(self.scale-maxsc).reshape(-1,1,1),out=out.M)
            np.subtract(out.M,other.M*np.exp(other.scale-maxsc).reshape(-1,1,1),out=out.M)
            return out
    def multiply(self,other,out=None):
        if type(other) is float or type(other) is int:return self.scalarMultiply(other,out)
        if out is None:
            return ScaledMatrixStack(self.M*other.M,self.scale+other.scale)
        elif out is self:
            self.M*=other.M
            self.scale+=other.scale
            return self
        elif out is other:
            other.M*=self.M
            other.scale+=self.scale
            return other
        else:
            np.multiply(self.M,other.M,out=out.M)
            np.add(self.scale,other.scale,out=out.scale)
            return out
    def scalarMultiply(self,other,out=None):
        if out is None:
            return ScaledMatrixStack(self.M.copy(),self.scale+np.log(other))
        elif out is self:
            self.scale+=np.log(other)
            return self
        else:
            np.copyto(out.M,self.M)
            np.add(self.scale,np.log(other),out.scale)
            return out
    def divide(self,other,out=None):
        if type(other) is float or type(other) is int: return self.scalarMultiply(self,1/other,out)
        if out is None:
            return ScaledMatrixStack(self.M/other.M,self.scale-other.scale)
        elif out is self:
            self.M/=other.M
            self.scale-=other.scale
            return self
        else:
            np.divide(self.M,other.M,out=out.M)
            np.subtract(self.scale,other.scale,out=out.scale)
            return out
    def __add__(self,other):
        return self.add(other)
    def __sub__(self,other):
        return self.subtract(other)
    def __matmul__(self,other):
        return self.matmul(other)
    def __mul__(self,other):
        return self.multiply(other)
    def __truediv__(self,other):
        return self.divide(other)
    def __imatmul__(self,other):
        return self.matmul(other,out=self)
    def __imul__(self,other):
        return self.multiply(other,out=self)
    def __iadd__(self,other):
        return self.add(other,out=self)
    def __isub__(self,other):
        return self.subtract(other,out=self)
    def __itruediv__(self,other):
        return self.divide(other,out=self)
    def __len__(self):
        return self.nStack


def empty_like(s):
    return ScaledMatrixStack(nStack=s.nStack,N=s.shape[0],M=s.shape[1],dtypeData = s.M.dtype,dypeScale=s.scale.dtype)
