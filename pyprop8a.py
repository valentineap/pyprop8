import numpy as np
import datetime
import copy
import scipy.integrate as integ
import scipy.special as spec
import functools

ERRSTATE = 'warn'

def makeMomentTensor(strike,dip,rake,M0,eta,xtr):
    strike_r = np.deg2rad(strike)
    dip_r = np.deg2rad(dip)
    rake_r = np.deg2rad(rake)
    sv = np.array([0.,-np.cos(strike_r),np.sin(strike_r)])
    d = np.array([-np.sin(dip_r),np.cos(dip_r)*np.sin(strike_r),np.cos(dip_r)*np.cos(strike_r)])
    n = np.array([np.cos(dip_r),np.sin(dip_r)*np.sin(strike_r),np.sin(dip_r)*np.cos(strike_r)])
    e = sv*np.cos(rake_r) - d*np.sin(rake_r)
    b = np.cross(e,n)
    t = (e+n)/np.sqrt(2)
    p = (e-n)/np.sqrt(2)
    ev = M0 * np.array([-1-0.5*eta+0.5*xtr,eta,1-0.5*eta+0.5*xtr])
    fmom = np.zeros(6)
    fmom[:3] = ev[0]*p**2 + ev[1]*b**2+ev[2]*t**2
    fmom[3] = ev[0]*p[0]*p[1] + ev[1]*b[0]*b[1]+ev[2]*t[0]*t[1]
    fmom[4] = ev[0]*p[0]*p[2] + ev[1]*b[0]*b[2]+ev[2]*t[0]*t[2]
    fmom[5] = ev[0]*p[1]*p[2] + ev[1]*b[1]*b[2]+ev[2]*t[1]*t[2]
    M = np.array([[fmom[0],fmom[3],fmom[4]],
                       [fmom[3],fmom[1],fmom[5]],
                       [fmom[4],fmom[5],fmom[2]]])
    return M
def rtf2xyz(M):
    M2 = np.zeros_like(M)
    M2[0:2,0:2] = M[1:3,1:3]
    M2[0:2,2] = M[0,1:3]
    M2[2,0:2] = M[1:3,0]
    M2[2,2] = M[0,0]
    return M2


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



class Layer:
    def __init__(self,thickness,vp,vs,rho):
        self.dz = thickness
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.mu = self.rho*self.vs**2
        self.sigma = self.rho*self.vp**2
        self.clearPropagators()
    def setFrequency(self,omega,k):
        self.clearPropagators()
        self.omega = omega
        self.k = k
    def clearPropagators(self):
        self.Z4 = None
        self.iZ4 = None
        self.Z6 = None
        self.iZ6 = None
        self.M4up = None
        self.M4down = None
        self.M6up = None
        self.M6down = None
    def make4DPropagators(self):
        nk = self.k.shape[0]
        self.Z4 = ScaledMatrixStack(nStack=nk,N=4,M=4,dtypeData='complex128',dtypeScale='float64')
        self.iZ4 = ScaledMatrixStack(nStack=nk,N=4,M=4,dtypeData='complex128',dtypeScale='float64')
        self.M4up = ScaledMatrixStack(nStack=nk,N=4,M=4,dtypeData='complex128',dtypeScale='float64')
        self.M4down = ScaledMatrixStack(nStack=nk,N=4,M=4,dtypeData='complex128',dtypeScale='float64')
        kp = self.k>0
        if self.omega==0:
            self.Z4[kp],self.iZ4[kp],self.M4up[kp],self.M4down[kp] = make4DPropagatorsZeroFreq(self.k[kp],self.dz,*self.materialProperties,Z = self.Z4[kp],iZ=self.iZ4[kp],Mup=self.M4up[kp],Mdown=self.M4down[kp])
        else:
            self.Z4[kp],self.iZ4[kp],self.M4up[kp],self.M4down[kp] = make4DPropagators(self.omega,self.k[kp],self.dz,*self.materialProperties,Z = self.Z4[kp],iZ=self.iZ4[kp],Mup=self.M4up[kp],Mdown=self.M4down[kp])
    def make6DPropagators(self):
        nk = self.k.shape[0]
        self.Z6 = ScaledMatrixStack(nStack=nk,N=6,M=6,dtypeData='complex128',dtypeScale='float64')
        self.iZ6 = ScaledMatrixStack(nStack=nk,N=6,M=6,dtypeData='complex128',dtypeScale='float64')
        self.M6up = ScaledMatrixStack(nStack=nk,N=6,M=6,dtypeData='complex128',dtypeScale='float64')
        self.M6down = ScaledMatrixStack(nStack=nk,N=6,M=6,dtypeData='complex128',dtypeScale='float64')
        kp = self.k>0
        if self.omega==0:
            self.Z6[kp],self.iZ6[kp],self.M6up[kp],self.M6down[kp] = make6DPropagatorsZeroFreq(self.k[kp],self.dz,*self.materialProperties,Z = self.Z6[kp],iZ=self.iZ6[kp],Mup=self.M6up[kp],Mdown=self.M6down[kp])
        else:
            self.Z6[kp],self.iZ6[kp],self.M6up[kp],self.M6down[kp] = make6DPropagators(self.omega,self.k[kp],self.dz,*self.materialProperties,Z = self.Z6[kp],iZ=self.iZ6[kp],Mup=self.M6up[kp],Mdown=self.M6down[kp])
    def propup(self,m,minor=False):
        if minor:
            if self.Z6 is None: self.make6DPropagators()
            #return self.Z6@self.M6up@self.iZ6@m
            return self.Z6.matmul(self.M6up.matmul(self.iZ6.matmul(m,out=m),out=m),out=m)
        else:
            if self.Z4 is None:self.make4DPropagators()
            #return self.Z4 @ self.M4up @ self.iZ4 @ m
            return self.Z4.matmul(self.M4up.matmul(self.iZ4.matmul(m,out=m),out=m),out=m)
    def propdown(self,m,minor=False):
        if minor:
            if self.Z6 is None: self.make6DPropagators()
            #return self.Z6 @ self.M6down@self.iZ6@m
            return self.Z6.matmul(self.M6down.matmul(self.iZ6.matmul(m,out=m),out=m),out=m)
        else:
            if self.Z4 is None: self.make4DPropagators()
            return self.Z4 @ self.M4down@self.iZ4@m


    @property
    def materialProperties(self):
        return self.sigma, self.mu, self.rho
    def __repr__(self):
        return "Layer. Thickness: %f\n|             vp: %f\n|             vs: %f\n`----------- rho: %f"%(self.dz,self.vp,self.vs,self.rho)
class Stack:
    def __init__(self,layerlist = None,omega=None,k=None,copylist=True):
        if layerlist is None:
            self.layers = []
        else:
            if copylist:
                self.layers = copy.deepcopy(layerlist)
            else:
                self.layers = layerlist
        self.omega = omega
        self.k = k
        self.iSource = None
        self.iReceiver = None
    def __getitem__(self,key):
        if type(key) is slice:
            return Stack(self.layers[key],self.omega,self.k,copylist=False)
        else:
            return self.layers[key]
    def __setitem__(self,key,value):
        self.layers[key]=value
    def append(self,value):
        self.layers.append(value)
    def setFrequency(self,omega,k):
        for l in self.layers:
            l.setFrequency(omega,k)
        self.omega = omega
        self.k = k
    def clearPropagators(self):
        self.omega= None
        self.k = None
        for l in self.layers:
            l.clearPropagators()
    @property
    def nlayers(self):
        return len(self.layers)
    def propagateDown(self,v,minor):
        for l in self.layers:
            v = l.propdown(v,minor)
        return v
    def propagateUp(self,v,minor):
        for l in self.layers[-1::-1]:
            v = l.propup(v,minor)
        return v
    @property
    def propagatedSurfaceCondition(self):
        try:
            nk = self.k.shape[0]
        except AttributeError:
            nk = 1
        if self.nlayers==0:
            v = freeSurfaceBoundary(nk)
        else:
            if self.layers[0].mu == 0.:
                v = oceanFloorBoundary(self.layers[0].dz,self.omega,self.k,self.layers[0].sigma, self.layers[0].rho)
            else:
                v = self.layers[0].propdown(freeSurfaceBoundary(nk),True)
            if self.nlayers>1:
                v = self[1:].propagateDown(v,True)
        return v
    @property
    def propagatedBasalCondition(self):
        if not np.isinf(self.layers[-1].dz):
            raise ValueError("Basal boundary condition only defined for infinite halfspace")
        v = underlyingHalfspaceBoundary(self.omega,self.k,*self.layers[-1].materialProperties)
        if self.nlayers>1:
            v = self[:-1].propagateUp(v,True)
        return v
    def H(self,omega,k):
        if not self.iReceiver < self.iSource: raise ValueError("Source must lie below receiver")
        #if k==0: return ScaledMatrix(np.zeros([4,4],dtype='complex128'),0,None)
        self.setFrequency(omega,k)
        bcSurface_rec = self[:self.iReceiver].propagatedSurfaceCondition
        bcBase_src = self[self.iSource:].propagatedBasalCondition
        bcBase_rec = self[self.iReceiver:].propagatedBasalCondition
        Delta = makeDelta(bcSurface_rec,bcBase_rec)
        H = ScaledMatrixStack(nStack = k.shape[0], N = 4, M = 4,dtypeData='complex128',dtypeScale='float64')
        H[k>0] = makeN(bcSurface_rec[k>0]).matmul(self[self.iReceiver:self.iSource].propagateUp(makeN(bcBase_src),False)[k>0],out=H[k>0])
        H[k>0]/=Delta[k>0]
        return H
    def _insertLayer(self,z):
        stackz = 0.
        for ilayer,layer in enumerate(self.layers):
            if z<stackz+layer.dz: break
            stackz+=layer.dz
        if z>stackz: # not already at interface
            newlayer = Layer(layer.dz-(z-stackz),layer.vp,layer.vs,layer.rho)
            layer.dz = z-stackz
            self.layers.insert(ilayer+1,newlayer)
            ilayer+=1
        return ilayer
    def insertSource(self,z):
        if z<=0: raise ValueError("Source must be buried.")
        self.iSource = self._insertLayer(z)
    def insertReceiver(self,z):
        if z<0: raise ValueError("Receiver must be at or below surface")
        self.iReceiver = self._insertLayer(z)
    def b(self,omega,k,MT,F):
        return (self.H(omega,k) @ sourceVector(MT,F,k,self.layers[self.iSource].sigma,self.layers[self.iSource].mu)).value
    def trapzKm(self,omega,MT,F,r,kmin=0,kmax=2.04,nk = 1200):
        kk = np.linspace(kmin,kmax,nk)
        y = kk.reshape(nk,1)*self.b(omega,kk,MT,F)[:,0,:]*np.array([spec.jv(np.arange(-2,3),r*k) for k in kk]).reshape(nk,5)
        return integ.trapz(np.where(np.isnan(y),0.,y),dx = kk[1],axis=0)
    def mytrapz(self,omega,MT,F,r,kmax=2.04,nk=1200):
        kk = np.linspace(0,kmax,nk)
        f = self.b(omega,kk,MT,F)[:,0,:]*spec.jv(np.tile(np.arange(-2,3),nk),np.repeat(r*kk,5)).reshape(nk,5)
        return kk[1]*(kk.dot(f) - 0.5*kmax*f[nk-1,:])
    def getIntegrand(self,omega,MT,F,r):
        return lambda k: k*(self.b(omega,k,MT,F) * spec.jv(np.arange(-2,3),k*r))
    def K(self,omega,MT,F,r,phi):
        eimp = np.exp(1j*phi*np.arange(-2,3))
        return integ.quad(lambda k:np.real(k*(self.b(omega,k,MT,F) @ (eimp*spec.jv(np.arange(-2,3),k*r))) [0]),0,np.inf)[0]
    def iK(self,omega,MT,F,r,phi):
        eimp = np.exp(1j*phi*np.arange(-2,3))
        return integ.quad(lambda k:np.imag(k*(self.b(omega,k,MT,F) @ (eimp*spec.jv(np.arange(-2,3),k*r))) [0]),0,np.inf)[0]

def sourceVector(MT,F,k,sigma,mu):
    nk = k.shape[0]
    s = np.zeros([nk,4,5],dtype='complex128')

    s[:,0,2] = MT[2,2]/sigma
    s[:,2,2] = -F[2]
    s[:,3,2] = 0.5*k*(MT[0,0]+MT[1,1]) - k*(sigma-2*mu)*MT[2,2]/sigma
    for sgn in [-1,1]:
        s[:,1,2+sgn] = 0.5*(sgn*MT[0,2] - 1j*MT[1,2])/mu
        s[:,2,2+sgn] = sgn*0.5*k*(MT[0,2]-MT[2,0])+0.5*1j*k*(MT[2,1]-MT[1,2])
        s[:,3,2+sgn] = 0.5*(-sgn*F[0]+1j*F[1])
        s[:,3,2+2*sgn] = 0.25*k*(MT[1,1]-MT[0,0])+sgn*0.25*1j*k*(MT[0,1]+MT[1,0])
    return ScaledMatrixStack(s)

def sourcetimefunc(omega,trise,trupt):
    uu = np.ones(omega.shape,dtype='complex128')
    uxx = np.ones_like(uu)
    uex=np.ones_like(uu)
    wp = omega!=0
    uu[wp] = omega[wp]*trise*1j
    uu[wp] = (1-np.exp(-uu[wp]))/uu[wp]
    uxx[wp] = 1j*omega[wp]*trupt/2
    uex[wp] = np.exp(uxx[wp])
    uxx[wp] = (uex[wp]-1/uex[wp])/(2*uxx[wp])
    return uu*uxx

def makeN(scm):
    m = scm.M
    N = np.zeros([scm.nStack,4,4],dtype='complex128')
    N[:,0,0] = -m[:,1,0]
    N[:,0,1] = -m[:,2,0]
    N[:,0,3] = m[:,0,0]
    N[:,1,0] = -m[:,3,0]
    N[:,1,1] = -m[:,4,0]
    N[:,1,2] = -m[:,0,0]
    N[:,2,1] = -m[:,5,0]
    N[:,2,2] = -m[:,1,0]
    N[:,2,3] = -m[:,3,0]
    N[:,3,0] = m[:,5,0]
    N[:,3,2] = -m[:,2,0]
    N[:,3,3] = -m[:,4,0]
    return ScaledMatrixStack(N,scm.scale)
def makeDelta(scm1,scm2):
    m1 = scm1.M
    m2 = scm2.M
    if not scm1.nStack==scm2.nStack: raise ValueError("Dimension mismatch")
    m = np.zeros([scm1.nStack,1,1],dtype='complex128')
    m[:,0,0] = m1[:,0,0]*m2[:,5,0] - m1[:,1,0]*m2[:,4,0] + m1[:,2,0]*m2[:,3,0] + m1[:,3,0]*m2[:,2,0] - m1[:,4,0]*m2[:,1,0] + m1[:,5,0]*m2[:,0,0]
    return ScaledMatrixStack(m,scm1.scale+scm2.scale)
    # return ScaledMatrix(m1[0]*m2[5] - m1[1]*m2[4] + m1[2]*m2[3] + m1[3]*m2[2] - m1[4]*m2[1] + m1[5]*m2[0],scm1.scale+scm2.scale)






def exphyp(x):
    a = np.real(x)
    b = np.imag(x)
    sgn = np.sign(a)
    t = np.exp(-2*sgn*a)
    return 0.5*np.cos(b)*(1+t)+0.5j*np.sin(b)*sgn*(1-t),0.5*np.cos(b)*sgn*(1-t)+0.5j*np.sin(b)*(1+t),sgn*a#0.5*fac*(1+t),sgn*0.5*fac*(1-t),sc

def make4DPropagatorsZeroFreq(k,dz,sigma,mu,rho,Z = None,iZ = None,Mup = None,Mdown = None):
    # Omega is known to be zero...!
    nk = k.shape[0]
    c,s,scale = exphyp(dz*k)
    # Terms that don't change under h->-h
    exphap_c = np.zeros([nk,4,4],dtype='complex128')
    exphap_c[:,0,0] = c
    exphap_c[:,0,1] = dz*s*rho*(sigma-mu)/(2*sigma*mu)
    exphap_c[:,1,1] = c
    exphap_c[:,2,2] = c
    exphap_c[:,3,2] = dz*s*rho*(sigma-mu)/(2*sigma*mu)
    exphap_c[:,3,3] = c
    exphap_c = ScaledMatrixStack(exphap_c,scale.copy())
    # Terms that flip sign
    exphap_s = np.zeros([nk,4,4],dtype='complex128')
    exphap_s[:,0,2] = rho*(c*dz*k*(mu-sigma)+s*(mu+sigma))/(2*k*mu*sigma)
    exphap_s[:,0,3] = -s
    exphap_s[:,1,2] = -s
    exphap_s[:,2,1] = -s
    exphap_s[:,3,0] = -s
    exphap_s[:,3,1] = rho*(c*dz*k*(mu-sigma)-s*(mu+sigma))/(2*k*mu*sigma)
    exphap_s = ScaledMatrixStack(exphap_s,scale) # Safe not to copy scale as not used again
    if Z is None: Z = ScaledMatrixStack(nStack=nk,N=4,M=4,dtypeData='complex128',dtypeScale='float64')
    rtrho = np.sqrt(rho)
    Z.M[:,0,0] = 1/rtrho
    Z.M[:,1,3] = -1/rtrho
    Z.M[:,2,2] = rtrho
    Z.M[:,2,3] = -2*k*mu/rtrho
    Z.M[:,3,0] = 2*k*mu/rtrho
    Z.M[:,3,1] = rtrho
    Z.scale[:] = 0.

    if iZ is None: iZ = ScaledMatrixStack(nStack=nk,N=4,M=4,dtypeData='complex128',dtypeScale='float64')
    iZ.M[:,0,0] = rtrho
    iZ.M[:,1,0] = -2*k*mu/rtrho
    iZ.M[:,1,3] = 1/rtrho
    iZ.M[:,2,1] = -2*k*mu/rtrho
    iZ.M[:,2,2] = 1/rtrho
    iZ.M[:,3,1] = -rtrho
    iZ.scale[:] = 0.

    Mup = exphap_c.add(exphap_s,out=Mup)
    Mdown = exphap_c.subtract(exphap_s,out=Mdown)
    return Z,iZ,Mup,Mdown

def make6DPropagatorsZeroFreq(k,dz,sigma,mu,rho,Z = None,iZ = None,Mup = None,Mdown = None):
    nk = k.shape[0]
    c,s,scale = exphyp(dz*k)
    # Terms that don't change under h->-h
    mexphap_c = np.zeros([nk,6,6],dtype='complex128')
    mexphap_c[:,0,0] = c**2
    mexphap_c[:,0,5] = s**2
    mexphap_c[:,1,1] = c**2
    mexphap_c[:,1,3] = -s**2*rho*(mu+sigma)
    mexphap_c[:,1,4] = s**2
    mexphap_c[:,2,1] = -s**2*rho*(mu+sigma)
    mexphap_c[:,2,3] =-(s*rho*(mu+sigma))**2
    mexphap_c[:,2,4] = s**2*rho*(mu+sigma)
    mexphap_c[:,4,1] = s**2
    mexphap_c[:,4,3] = s**2*rho*(mu+sigma)
    mexphap_c[:,4,4] = c**2
    mexphap_c[:,5,0] = s**2
    mexphap_c[:,5,5] = c**2
    mexphap_c = ScaledMatrixStack(mexphap_c,2*scale)

    mexphap_c_noscale = np.zeros([nk,6,6],dtype='complex128')
    mexphap_c_noscale[:,2,2] = 2*k*mu*sigma
    mexphap_c_noscale[:,2,3] = -(rho*dz*k*(mu-sigma))**2
    mexphap_c_noscale[:,3,3]= 2*k*mu*sigma


    mexphap_s = np.zeros([nk,6,6],dtype='complex128')
    mexphap_s[:,0,1] = -c*s
    mexphap_s[:,0,3] = -rho*c*s*(mu+sigma)
    mexphap_s[:,0,4] = c*s
    mexphap_s[:,1,0] = -c*s
    mexphap_s[:,1,5] = c*s
    mexphap_s[:,2,0] = -rho*c*s*(mu+sigma)
    mexphap_s[:,2,5] = rho*c*s*(mu+sigma)
    mexphap_s[:,4,0] = c*s
    mexphap_s[:,4,5] = -c*s
    mexphap_s[:,5,1] = c*s
    mexphap_s[:,5,3] = rho*c*s*(mu+sigma)
    mexphap_s[:,5,4] = -c*s
    mexphap_s = ScaledMatrixStack(mexphap_s,2*scale)

    mexphap_s_noscale = np.zeros([nk,6,6],dtype='complex128')
    mexphap_s_noscale[:,0,3] = -rho*dz*k*(mu-sigma)
    mexphap_s_noscale[:,2,0] = rho*dz*k*(mu-sigma)
    mexphap_s_noscale[:,2,5] = rho*dz*k*(mu-sigma)
    mexphap_s_noscale[:,5,3] = -rho*dz*k*(mu-sigma)


    if Z is None: Z = ScaledMatrixStack(nStack=nk,N=6,M=6,dtypeData='complex128',dtypeScale='float64')
    Z.M[:,0,2] = -1/(2*k*mu*rho*sigma)
    Z.M[:,1,1] = 1
    Z.M[:,1,2] = -1/(rho*sigma)
    Z.M[:,2,0] = 1
    Z.M[:,3,5] = 1
    Z.M[:,4,2] = 1/(rho*sigma)
    Z.M[:,4,4] = 1
    Z.M[:,4,1] = -2*k*mu
    Z.M[:,4,2] = 2*k*mu/(rho*sigma)
    Z.M[:,4,3] = -rho
    Z.M[:,4,4] = 2*k*mu
    Z.scale[:] = 0.

    if iZ is None: iZ = ScaledMatrixStack(nStack=nk,N=6,M=6,dtypeData='complex128',dtypeScale='float64')
    iZ.M[:,0,2] = 1
    iZ.M[:,1,0] = -2*k*mu
    iZ.M[:,1,1] = 1
    iZ.M[:,2,0] = -rho/(2*k*mu*sigma)
    iZ.M[:,3,0] = 4*(mu*k)**2/rho
    iZ.M[:,3,1] = -2*k*mu/rho
    iZ.M[:,3,4] = 2*k*mu/rho
    iZ.M[:,3,5] = -1/rho
    iZ.M[:,4,0] = 2*k*mu
    iZ.M[:,4,4] = 1
    iZ.M[:,5,4] = 1
    iZ.scale[:] = 0.

    Mup = mexphap_c+mexphap_s+ScaledMatrixStack(mexphap_c_noscale+mexphap_s_noscale)
    Mdown = mexphap_c-mexphap_s+ScaledMatrixStack(mexphap_c_noscale-mexphap_s_noscale)
    return Z,iZ,Mup,Mdown

def make4DPropagators(omega,k,dz,sigma,mu,rho, Z=None,iZ=None,Mup = None,Mdown=None):
    nk = k.shape[0]
    zmu = np.lib.scimath.sqrt(k**2 - rho*omega**2/mu)
    ximu = zmu/k
    zsig = np.lib.scimath.sqrt(k**2 - rho*omega**2/sigma)
    xisig= zsig/k
    xiprod = ximu*xisig
    cmu,smu,scalemu = exphyp(dz*zmu)
    csig,ssig,scalesig = exphyp(dz*zsig)
    # If h -> -h, we see sign changes on smu, ssig only.

    exphap_s = np.zeros([nk,4,4],dtype='complex128')
    exphap_s[:,0,0] = csig
    exphap_s[:,0,1] = csig*k/omega**2
    exphap_s[:,0,2] = -ssig*zsig/omega**2
    exphap_s[:,2,0] = -ssig*omega**2/zsig
    exphap_s[:,2,1] = -k*ssig/zsig
    exphap_s[:,2,2] = csig
    exphap_s[:,3,0] = -k*ssig/zsig
    exphap_s[:,3,1] = -k**2 * ssig/(zsig*omega**2)
    exphap_s[:,3,2] = k*csig/omega**2
    exphap_s = ScaledMatrixStack(exphap_s,scalesig)

    exphap_m = np.zeros([nk,4,4],dtype='complex128')
    exphap_m[:,0,1] = -k*cmu/omega**2
    exphap_m[:,0,2] = k**2 * smu/(zmu*omega**2)
    exphap_m[:,0,3] = -k*smu/zmu
    exphap_m[:,1,1] = cmu
    exphap_m[:,1,2] = -k*smu/zmu
    exphap_m[:,1,3] = smu*omega**2/zmu
    exphap_m[:,3,1] = smu*zmu/omega**2
    exphap_m[:,3,2] = -k*cmu/omega**2
    exphap_m[:,3,3] = cmu
    exphap_m = ScaledMatrixStack(exphap_m,scalemu)

    Mup = exphap_s.add(exphap_m,out=Mup)
    Mdown = Mup.copy(Mdown)
    np.negative(Mdown.M[:,0:2,2:],Mdown.M[:,0:2,2:])
    np.negative(Mdown.M[:,2:,0:2],Mdown.M[:,2:,0:2])


    rtrho = np.sqrt(rho)

    if Z is None:Z = ScaledMatrixStack(nStack = nk, N = 4, M = 4,dtypeData='complex128',dtypeScale = 'float64')
    Z.M[:,0,0] = 1/rtrho
    Z.M[:,1,3] = -1/rtrho
    Z.M[:,2,2] = rtrho
    Z.M[:,2,3] = -2*mu*k/rtrho
    Z.M[:,3,0] = 2*mu*k/rtrho
    Z.M[:,3,1] = rtrho
    Z.scale[:] = 0.

    if iZ is None: iZ = ScaledMatrixStack(nStack = nk, N = 4, M = 4,dtypeData='complex128',dtypeScale = 'float64')
    iZ.M[:,0,0] = rtrho
    iZ.M[:,1,0] = -2*mu*k/rtrho
    iZ.M[:,1,3] = 1/rtrho
    iZ.M[:,2,1] = -2*mu*k/rtrho
    iZ.M[:,2,2] = 1/rtrho
    iZ.M[:,3,1] = -rtrho
    iZ.scale = 0.

    return Z,iZ,Mup,Mdown

def make6DPropagators(omega,k,dz,sigma,mu,rho, Z=None,iZ=None,Mup = None,Mdown=None):
    nk = k.shape[0]

    zmu = np.lib.scimath.sqrt(k**2 - rho*omega**2/mu)
    ximu = zmu/k
    zsig = np.lib.scimath.sqrt(k**2 - rho*omega**2/sigma)
    xisig= zsig/k
    xiprod = ximu*xisig
    cmu,smu,scalemu = exphyp(dz*zmu)
    csig,ssig,scalesig = exphyp(dz*zsig)
    # If h -> -h, we see sign changes on  X1 and X2 only.

    Pc = cmu*csig
    Ps = smu*ssig
    X1 = cmu*ssig
    X2 = csig*smu

    # Split matrix into term containing cosh/sinh products (which need scaling by exp(scale)) and the rest.
    M1 = np.zeros([nk,6,6],dtype='complex128')

    M1[:,0,0] = Pc
    M1[:,0,1] = -X2
    M1[:,0,2] = X2
    M1[:,0,3] = -X2+xiprod*X1
    M1[:,0,4] = X2
    M1[:,0,5] = -Ps

    M1[:,1,0] = -X1
    M1[:,1,1] = Ps
    M1[:,1,2] = -Ps
    M1[:,1,3] = Ps-Pc*xiprod
    M1[:,1,4] = -Ps
    M1[:,1,5] = X2

    M1[:,2,0] = -X1+xiprod*X2
    M1[:,2,1] = Ps-Pc*xiprod
    M1[:,2,2] = -Ps+Pc*xiprod
    M1[:,2,3] = -2*Pc*xiprod+Ps*(1+xiprod**2)
    M1[:,2,4] = -Ps+Pc*xiprod
    M1[:,2,5] = X2-X1*xiprod

    M1[:,3,0] = X1
    M1[:,3,1] = -Ps
    M1[:,3,2] = Ps
    M1[:,3,3] = -Ps+Pc*xiprod
    M1[:,3,4] = Ps
    M1[:,3,5] = -X2

    M1[:,4,0] = X1
    M1[:,4,1] = -Ps
    M1[:,4,2] = Ps
    M1[:,4,3] = -Ps+Pc*xiprod
    M1[:,4,4] = Ps
    M1[:,4,5] = -X2

    M1[:,5,0] = -Ps
    M1[:,5,1] = X1
    M1[:,5,2] = -X1
    M1[:,5,3] = X1-X2*xiprod
    M1[:,5,4] = -X1
    M1[:,5,5] = Pc
    M1 = ScaledMatrixStack(M1,scalemu+scalesig)


    M2 = np.zeros([nk,6,6],dtype='complex128')
    M2[:,1,1] = xiprod
    M2[:,1,3] = xiprod
    M2[:,2,1] = xiprod
    M2[:,2,3] = 2*xiprod
    M2[:,2,4] = -xiprod
    M2[:,4,3] = -xiprod
    M2[:,4,4] = xiprod
    M2 = ScaledMatrixStack(M2)


    if Z is None: Z = ScaledMatrixStack(nStack=nk,N=6,M=6,dtypeData='complex128',dtypeScale='float64')
    Z.M[:,0,2] = -k**2/(rho*omega**2)
    Z.M[:,1,1] = k
    Z.M[:,1,2] = -2*k**3*mu/(rho*omega**2)
    Z.M[:,2,0] = zsig
    Z.M[:,3,5] = zmu
    Z.M[:,4,2] = 2*k**3*mu/(rho*omega**2)
    Z.M[:,4,4] = k
    Z.M[:,5,1] = -2*k**2*mu
    Z.M[:,5,2] = 4*k**4*mu**2 / (rho*omega**2)
    Z.M[:,5,3] = -rho*omega**2
    Z.M[:,5,4] = 2*k**2*mu
    Z.scale[:] = 0.

    if iZ is None: iZ = ScaledMatrixStack(nStack=nk,N=6,M=6,dtypeData='complex128',dtypeScale='float64')
    iZ.M[:,0,2] = 1/zsig
    iZ.M[:,1,0] = -2*k**2*mu/(zsig*zmu)
    iZ.M[:,1,1] = k/(zsig*zmu)
    iZ.M[:,2,0] = -rho*omega**2/(zsig*zmu)
    iZ.M[:,3,0] = 4*k**4*mu**2/(rho*omega**2 *zsig*zmu)
    iZ.M[:,3,1] = -2*k**3*mu/(rho*omega**2 * zsig*zmu)
    iZ.M[:,3,4] = 2*k**3*mu/(rho*omega**2 * zsig*zmu)
    iZ.M[:,3,5] = -k**2/(rho*omega**2 * zsig*zmu)
    iZ.M[:,4,0] = 2*k**2*mu/(zsig*zmu)
    iZ.M[:,4,4] = k/(zsig*zmu)
    iZ.M[:,5,3] = 1/zmu
    iZ.scale[:] = 0.

    Mup = M1.add(M2,out=Mup)
    np.negative(M1.M[:,0,1:5],M1.M[:,0,1:5])
    np.negative(M1.M[:,5,1:5],M1.M[:,5,1:5])
    np.negative(M1.M[:,1:5,0],M1.M[:,1:5,0])
    np.negative(M1.M[:,1:5,5],M1.M[:,1:5,5])
    Mdown = M1.add(M2,out=Mdown)
    return Z,iZ,Mup,Mdown

def freeSurfaceBoundary(nk):
    m = np.zeros([nk,6,1],dtype='complex128')
    m[:,0,0] = 1
    return ScaledMatrixStack(m)


def oceanFloorBoundary(depth,omega,k,sigma,rho,eps_om=1e-3,eps_k=1e-5):
    nk = k.shape[0]
    zsig = np.lib.scimath.sqrt(k**2 - rho*omega**2/sigma)
    t = np.exp(-2*depth*zsig)
    m = np.zeros([nk,6,1],dtype='complex128')
    m[:,0,0] = 1+t
    if omega!=0: m[:,3,0] = -rho*omega**2*(1-t)/zsig
    return ScaledMatrixStack(m)

def underlyingHalfspaceBoundary(omega,k,sigma,mu,rho,eps_om=1e-3,eps_k=1e-5):
    nk = k.shape[0]
    zsig = np.lib.scimath.sqrt(k**2 - rho*omega**2/sigma)
    zmu = np.lib.scimath.sqrt(k**2 - rho*omega**2/mu)
    xi = np.zeros_like(zsig)
    xi[k==0] = np.sqrt(mu/sigma)
    xi[k>0] = (rho*omega**2/sigma - k[k>0]**2*(1+mu/sigma))/(k[k>0]**2+zsig[k>0]*zmu[k>0])
    m = np.zeros([nk,6,1],dtype='complex128')
    m[:,0,0] = xi/mu
    m[:,1,0] = 2*k*(xi+0.5)
    m[:,2,0] = -zsig
    m[:,3,0] = zmu
    m[:,4,0] = -2*k*(xi+0.5)
    m[:,5,0] = rho*omega**2 - 4*k**2 * mu*(xi+1)
    return ScaledMatrixStack(m)

def loado7(file,irec,icomp,ifreq,nk=1200,ncomp=3,nrecs=20,nfreq=257):
    fp = open(file,'r')
    for i in range(7):
        fp.readline()
    out = np.zeros([nk,5],dtype='complex')
    for ifr in range(ifreq+1):
        for ik in range(nk):
            for ir in range(nrecs):
                for i in range(ncomp):
                    line = fp.readline()
                    if ir==irec and i==icomp and ifreq==ifr:
                        sp = line.split()
                        k=0
                        for j in range(5):
                            out[ik,j] = float(sp[k])+1j*float(sp[k+1])
                            k+=2
    return out

def reorderMT(M,order=[1,2,0]):
    M2 = np.zeros_like(M)
    for io,i in enumerate(order):
        for jo,j in enumerate(order):
            M2[io,jo] = M[i,j]
    return M2

S = Stack([Layer(3.,1.8,0.,1.02),
           Layer(2.,4.5,2.4,2.57),
           Layer(5.,5.8,3.3,2.63),
           Layer(20.,6.5,3.65,2.85),
           Layer(np.inf,8.,4.56,3.34)])
S.insertSource(20.)
S.insertReceiver(3.)
M = makeMomentTensor(340,90,0,2.4E8,0,0)
M2 = rtf2xyz(M)
Mp8 = np.array([[154269088.0,-183850592.0,9.8580646514892578],
                [-183850592.0,-154269120.0,3.5880444049835205],
                [9.8580646514892578,3.5880444049835205,0.]])
F = np.zeros([3])
nfft = 257
alpha = 0.023
dt = 0.5
kk = np.linspace(0,2.04,1200)
rr = np.linspace(10,200,20)
ww = 2*np.pi*np.linspace(0,1,2**8+1)-1j*alpha
