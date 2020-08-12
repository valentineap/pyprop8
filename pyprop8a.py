import numpy as np
import datetime
import copy
# class Settings:
#     fmin = 0.0
#     fmax = 1.0
#     nord = 8
#     kmax = 2.04
#     nk = 1200
#     alpha = 0.023
#     write7 = 0
# class Event:
#     evtime = datetime.datetime(2008,2,14,10,9,29,0)
#     lat = 36.24
#     lon = 21.79
#     depth = 20.0
#     M0 = 2.4e27*10**(-7 - 18 + 6)
#     strike = 340
#     dip = 90
#     rake = 0.
#     eta = 0.
#     xtr = 0.
#     def makeMomentTensor(self):
#         strike_r = np.deg2rad(self.strike)
#         dip_r = np.deg2rad(self.dip)
#         rake_r = np.deg2rad(self.rake)
#         sv = np.array([0.,-np.cos(strike_r),np.sin(strike_r)])
#         d = np.array([-np.sin(dip_r),np.cos(dip_r)*np.sin(strike_r),np.cos(dip_r)*np.cos(strike_r)])
#         n = np.array([np.cos(dip_r),np.sin(dip_r)*np.sin(strike_r),np.sin(dip_r)*np.cos(strike_r)])
#         e = sv*np.cos(rake_r) - d*np.sin(rake_r)
#         b = np.cross(e,n)
#         t = (e+n)/np.sqrt(2)
#         p = (e-n)/np.sqrt(2)
#         ev = self.M0 * np.array([-1-0.5*self.eta+0.5*self.xtr,self.eta,1-0.5*self.eta+0.5*self.xtr])
#         fmom = np.zeros(6)
#         fmom[:3] = ev[0]*p**2 + ev[1]*b**2+ev[2]*t**2
#         fmom[3] = ev[0]*p[0]*p[1] + ev[1]*b[0]*b[1]+ev[2]*t[0]*t[1]
#         fmom[4] = ev[0]*p[0]*p[2] + ev[1]*b[0]*b[2]+ev[2]*t[0]*t[2]
#         fmom[5] = ev[0]*p[1]*p[2] + ev[1]*b[1]*b[2]+ev[2]*t[1]*t[2]
#         self.M = np.array([[fmom[0],fmom[3],fmom[4]],
#                            [fmom[3],fmom[1],fmom[5]],
#                            [fmom[4],fmom[5],fmom[2]]])
#         return self.M
#
# class Output:
#     ifradii = 1
#     rmin  = 10.0
#     rmax = 200.0
#     nradii = 20





class ScaledMatrix:
    def __init__(self, M, scale,rescale = False):
        self.M = M
        self.scale = scale
        if rescale:
            sc = abs(self.M).max()
            if sc == 0.:sc =1
            self.M/=sc
            self.scale+=np.log(sc)
    def __matmul__(self,other):
        M = self.M @ other.M
        scale = abs(M).max()
        if scale==0:scale=1
        return ScaledMatrix(M/scale,self.scale + other.scale + np.log(scale))
    def __mul__(self,other):
        M = self.M*other.M
        scale = max(abs(M))
        if scale==0:scale=1
        return ScaledMatrix(M/scale,self.scale+other.scale+np.log(scale))
    def __add__(self,other):
        scm = max(self.scale,other.scale)
        M = np.exp(self.scale-scm)*self.M + np.exp(other.scale-scm)*other.M
        mabs = abs(M).max()
        return ScaledMatrix(M/mabs,scm+np.log(mabs))
    def __truediv__(self,other):
        return ScaledMatrix(self.M/other.M,self.scale-other.scale)
    def unscale(self):
        return np.exp(self.scale)*self.M
    def __repr__(self):
        return 'exp(%f) x %s'%(self.scale,self.M.__repr__())
    @property
    def shape(self):
        return self.M.shape

class Layer:
    def __init__(self,thickness,vp,vs,rho):
        self.dz = thickness
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.mu = self.rho*self.vs**2
        self.sigma = self.rho*self.vp**2
        self.clearPropagators()
    def clearPropagators(self):
        self.pup = None
        self.pdown = None
        self.mpup=None
        self.mpdown = None
    def makePropagators(self,omega,k,eps_om=1e-3):
        self.pup,self.mpup = propagator2(-self.dz,omega,k,self.sigma,self.mu,self.rho)
        self.pdown,self.mpdown = propagator2(self.dz,omega,k,self.sigma,self.mu,self.rho)
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
        self.omega = omega
        self.k = k
        for l in self.layers:
            l.makePropagators(omega,k)
    def clearPropagators(self):
        self.omega= None
        self.k = None
        for l in self.layers:
            l.clearPropagators()
    @property
    def nlayers(self):
        return len(self.layers)
    def propagateDown(self,v):
        for l in self.layers:
            if v.shape[0] ==4:
                v = l.pdown(v)
            else:
                v = l.mpdown(v)
        return v
    def propagateUp(self,v):
        for l in self.layers[-1::-1]:
            if v.shape[0]==4:
                v = l.pup(v)
            else:
                v = l.mpup(v)
        return v
    @property
    def propagatedSurfaceCondition(self):
        if self.layers[0].mu == 0.:
            v = oceanFloorBoundary(self.layers[0].dz,self.omega,self.k,self.layers[0].sigma, self.layers[0].rho)
        else:
            v = self.layers[0].mpdown(freeSurfaceBoundary())
        if self.nlayers>1:
            v = self[1:].propagateDown(v)
        return v
    @property
    def propagatedBasalCondition(self):
        if not np.isinf(self.layers[-1].dz):
            raise ValueError("Basal boundary condition only defined for infinite halfspace")
        v = underlyingHalfspaceBoundary(self.omega,self.k,*self.layers[-1].materialProperties)
        if self.nlayers>1:
            v = self[:-1].propagateUp(v)
        return v
    def H(self,omega,k):
        if not self.iReceiver < self.iSource: raise ValueError("Source must lie below receiver")
        if k==0: return np.zeros([4,4],dtype='complex128')
        self.setFrequency(omega,k)
        bcSurface_rec = self[:self.iReceiver].propagatedSurfaceCondition
        bcBase_src = self[self.iSource:].propagatedBasalCondition
        bcBase_rec = self[self.iReceiver:].propagatedBasalCondition
        Delta = makeDelta(bcSurface_rec,bcBase_rec)
        P_src_rec = self[self.iReceiver:self.iSource].propagateUp
        H = (makeN(bcSurface_rec) @ P_src_rec(makeN(bcBase_src)))/Delta
        return H.unscale()
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
        return self.H(omega,k) @ sourceVector(MT,F,k,self.layers[self.iSource].sigma,self.layers[self.iSource].mu)



def sourceVector(MT,F,k,sigma,mu):
    s = np.zeros([4,5],dtype='complex128')

    s[0,2] = MT[2,2]/sigma
    s[2,2] = -F[2]
    s[3,2] = 0.5*k*(MT[0,0]+MT[2,2]) - k*(sigma-2*mu)*MT[2,2]/sigma
    for sgn in [-1,1]:
        s[1,2+sgn] = 0.5*(sgn*MT[0,2] - 1j*MT[1,2])/mu
        s[2,2+sgn] = sgn*0.5*k*(MT[0,2]-MT[2,0])+0.5*1j*k*(MT[2,1]-MT[1,2])
        s[3,2+sgn] = 0.5*(-sgn*F[0]+1j*F[1])
        s[3,2+2*sgn] = 0.25*k*(MT[1,1]-MT[0,0])+sgn*0.25*1j*k*(MT[0,1]-MT[1,0])
    return s


def makeN(scm):
    m = scm.M
    N = np.zeros([4,4],dtype='complex128')
    N[0,0] = -m[1]
    N[0,1] = -m[2]
    N[0,3] = m[0]
    N[1,0] = -m[3]
    N[1,1] = -m[4]
    N[1,2] = -m[0]
    N[2,1] = -m[5]
    N[2,2] = -m[1]
    N[2,3] = -m[3]
    N[3,0] = m[5]
    N[3,2] = -m[2]
    N[3,3] = -m[4]
    return ScaledMatrix(N,scm.scale)
def makeDelta(scm1,scm2):
    m1 = scm1.M
    m2 = scm2.M
    return ScaledMatrix(m1[0]*m2[5] - m1[1]*m2[4] + m1[2]*m2[3] + m1[3]*m2[2] - m1[4]*m2[1] + m1[5]*m2[0],scm1.scale+scm2.scale)



# class Model:
#     def __init__(self):
#         self.dz = np.array([3.,2.,5.,20.,1e5])
#         self.interfaces = np.array([3,5,10,30,100030.])
#         self.vp = np.array([1.8,4.5,5.8,6.5,8.0])
#         self.vs = np.array([0.,2.4,3.3,3.65,4.56])
#         self.rho = np.array([1.02,2.57,2.63,2.85,3.34])
#     def insertInterface(self,z,eps = 1e-6):
#         """Insert an interface into the model at depth z.
#            Return the index of the layer that has z as its top surface.
#            If z is within eps of an existing interface, simply
#            return the index of the appropriate layer without inserting.
#         """
#         if z<0 or z>self.interfaces[-1]: raise ValueError("Source depth appears to be outside model domain")
#         source_layer = np.searchsorted(self.interfaces,z,'right')
#         if z-self.interfaces[source_layer-1]<=eps:
#             return source_layer
#         elif self.interfaces[source_layer]-z<=eps:
#             return source_layer+1
#         else:
#             self.interfaces = np.insert(self.interfaces,source_layer,z)
#             self.vp = np.insert(self.vp,source_layer,self.vp[source_layer])
#             self.vs = np.insert(self.vs,source_layer,self.vs[source_layer])
#             self.rho = np.insert(self.rho,source_layer,self.rho[source_layer])
#             return source_layer+1
#     def __repr__(self):
#         out = ['Model:','--------------- z = 0.00 km (Free surface)']
#         for vp,vs,rho,z in zip(self.vp,self.vs,self.rho,self.interfaces):
#             if vs == 0.:
#                 flu = '<-- Fluid'
#             else:
#                 flu = ''
#             out+=['vp = %.2f km/s'%vp, 'vs = %.2f km/s  %s'%(vs,flu), 'rho = %.2f g/cc'%rho,'--------------- z = %.2f km'%z]
#         return '\n'.join(out)


def minors(m):
    order = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    subdet = lambda x,r,c: np.linalg.det(x[np.array(r)[:,np.newaxis],np.array(c)])
    minor = np.zeros([6,6],dtype=m.dtype)
    for i,row in enumerate(order):
        for j,col in enumerate(order):
            minor[i,j] = subdet(m,row,col)
    return minor

def exphyp(x,scale = None):
    sc = abs(x)
    sgn = np.sign(x)
    t = np.exp(-2*sgn*x)
    fac = 1.
    if scale is not None:
        fac = np.exp(sc-scale)
        #print("...>",fac)
        sc = scale
    return 0.5*fac*(1+t),sgn*0.5*fac*(1-t),sc
    #
    # if sc<3:
    #     c = np.exp(-sc)*np.cosh(x)
    #     s = np.exp(-sc)*np.sinh(x)
    # else:
    #     c = 0.5*(1+np.exp(-2*sc))
    #     s = np.sign(x)*0.5*(1-np.exp(-2*sc))
    # return c,s,sc
def propagator2(h,omega,k,sigma,mu,rho):
    if mu==0 or np.isinf(h): return None,None
    zmu = np.lib.scimath.sqrt(k**2 - rho*omega**2/mu)
    ximu = zmu/k
    zsig = np.lib.scimath.sqrt(k**2 - rho*omega**2/sigma)
    xisig= zsig/k
    xiprod = ximu*xisig
    cmu,smu,scalemu = exphyp(h*zmu)
    csig,ssig,scalesig = exphyp(h*zsig)
    Pc = cmu*csig
    Ps = smu*ssig
    X1 = cmu*ssig
    X2 = csig*smu

    exphap_s = ScaledMatrix(np.array([[csig,k*csig/omega**2,-ssig*zsig/omega**2,0],
                                      [0,0,0,0],
                                      [-omega**2*ssig/zsig,-k*ssig/zsig,csig,0],
                                      [-k*ssig/zsig,-(k/omega)**2 *ssig/zsig,k*csig/omega**2,0]]),scalesig,True)
    exphap_c = ScaledMatrix(np.array([[0,-k*cmu/omega**2,(k/omega)**2*smu/zmu,-k*smu/zmu],
                                      [0,cmu,-k*smu/zmu,omega**2*smu/zmu],
                                      [0,0,0,0],
                                      [0,smu*zmu/omega**2,-k*cmu/omega**2,cmu]]),scalemu,True)

    Z =  np.zeros([4,4])
    iZ = np.zeros([4,4])
    rtrho = np.sqrt(rho)
    Z[0,0] = 1/rtrho
    Z[1,3] = -1/rtrho
    Z[2,2] = rtrho
    Z[2,3] = -2*mu*k/rtrho
    Z[3,0] = 2*mu*k/rtrho
    Z[3,1] = rtrho
    Z = ScaledMatrix(Z,0,True)
    iZ[0,0] = rtrho
    iZ[1,0] = -2*mu*k/rtrho
    iZ[1,3] = 1/rtrho
    iZ[2,1] = -2*mu*k/rtrho
    iZ[2,2] = 1/rtrho
    iZ[3,1] = -rtrho
    iZ = ScaledMatrix(iZ,0,True)

    v1 = ScaledMatrix(np.array([1/zsig,k,(k/omega)**2,omega**2,k,zmu]),0,True)
    M1 = ScaledMatrix(np.array([[Pc,-X2,X2,-X2+xiprod*X1,X2,-Ps],
                  [-X1,Ps+xiprod,-Ps,Ps-Pc*xiprod,-Ps,X2],
                  [-X1+xiprod*X2,Ps-Pc*xiprod,-Ps+Pc*xiprod,-2*Pc*xiprod+Ps*(1+xiprod**2),-Ps+Pc*xiprod,X2-X1*xiprod],
                  [X1,-Ps,Ps,-Ps+Pc*xiprod,Ps,-X2],
                  [X1,-Ps,Ps,-Ps+Pc*xiprod,Ps+xiprod,-X2],
                  [-Ps,X1,-X1,X1-X2*xiprod,-X1,Pc]]),scalemu+scalesig,True)
    M2 = ScaledMatrix(np.array([[0,0,0,0,0,0],[0,0,0,xiprod,0,0],[0,xiprod,0,2*xiprod,-xiprod,0],[0,0,0,0,0,0],[0,0,0,-xiprod,0,0],[0,0,0,0,0,0]]),0,True)
    v2 = ScaledMatrix(np.array([zsig,k,(k/omega)**2,omega**2,k,zmu]),0,True)
    mZ = np.zeros([6,6])
    mZi = np.zeros([6,6])
    mZ[0,2] = -1/rho
    mZ[1,1] = 1
    mZ[1,2] = -2*k*mu/rho
    mZ[2,0] = 1
    mZ[3,5] = 1
    mZ[4,2] = 2*k*mu/rho
    mZ[5,1] = -2*k*mu
    mZ[5,2] = (2*k*mu)**2 / rho
    mZ[5,3] = -rho
    mZ[5,4] = 2*k*mu
    mZ = ScaledMatrix(mZ,0,True)
    mZi[0,2] = 1
    mZi[1,0] = -2*k*mu
    mZi[1,1] = 1
    mZi[2,0] = -rho
    mZi[3,0] = (2*k*mu)**2/rho
    mZi[3,1] = -2*k*mu/rho
    mZi[3,4] = 2*k*mu/rho
    mZi[3,5] = -1/rho
    mZi[4,0] = 2*k*mu
    mZi[4,4] = 1
    mZi[5,3] = 1
    mZi = ScaledMatrix(mZi,0,True)
    return lambda m: (Z @ exphap_c @ iZ @ m)+(Z @ exphap_s @ iZ @ m), lambda m: mZ @ (v2*(M1 @ (v1*(mZi @ m))) + v2*(M2 @ (v1*(mZi @ m))))

def makePropagator(h,omega,k,sigma,mu,rho,eps_om = 1e-3,eps_k = 1e-5,dtype = 'complex128'):
    if mu==0 or np.isinf(h): return None,None
    AP = np.zeros([4,4],dtype=dtype)
    Z =  np.zeros([4,4],dtype=dtype)
    iZ = np.zeros([4,4],dtype=dtype)
    rtrho = np.sqrt(rho)
    Z[0,0] = 1/rtrho
    Z[1,3] = -1/rtrho
    Z[2,2] = rtrho
    Z[2,3] = -2*mu*k/rtrho
    Z[3,0] = 2*mu*k/rtrho
    Z[3,1] = rtrho
    iZ[0,0] = rtrho
    iZ[1,0] = -2*mu*k/rtrho
    iZ[1,3] = 1/rtrho
    iZ[2,1] = -2*mu*k/rtrho
    iZ[2,2] = 1/rtrho
    iZ[3,1] = -rtrho

    if abs(omega)>eps_om:
        k2mu = mu*k**2
        k2sig = sigma*k**2
        rw2 = rho*omega**2
        zmu = np.lib.scimath.sqrt(k**2 - rho*omega**2/mu)
        zsig = np.lib.scimath.sqrt(k**2 - rho*omega**2/sigma)

        cmu,smu,scale = exphyp(h*zmu)
        csig,ssig,scale = exphyp(h*zsig,scale)

        # cmu = (np.cosh(h * zmu))
        # csig = (np.cosh(h * zsig))
        # smu = np.sinh(h * zmu)
        # ssig = np.sinh(h * zsig)
        Rmu = (smu/zmu)
        Rsig = (ssig/zsig)
        Pmu = (smu*zmu)
        Psig = (ssig*zsig)
        # print(zmu,zsig,scale)
        # print(cmu,csig,scale)
        # print(smu,ssig)
        # print(Rmu,Rsig,Pmu,Psig)

        AP[0,0] = csig
        AP[0,1] = k*(csig-cmu)/omega**2
        AP[0,2] = (k**2*Rmu - Psig)/omega**2
        AP[0,3] = -k*Rmu

        AP[1,1] = cmu
        AP[1,2] = -k*Rmu
        AP[1,3] = Rmu*omega**2

        AP[2,0] = -Rsig*omega**2
        AP[2,1] = -k*Rsig
        AP[2,2] = csig

        AP[3,0] = -k*Rsig
        AP[3,1] = (Pmu - Rsig*k**2 )/omega**2
        AP[3,2] = k*(csig-cmu)/omega**2
        AP[3,3] = cmu
    else: # Power series expansion to O(omega**2)
        if k > eps_k:
            hk = h*k
            c,s,scale = exphyp(hk)
            # c = np.cosh(hk)
            # s = np.sinh(hk)
            rw2 = rho*omega**2
            tkms = 2*k*mu*sigma
            AP[0,0] = c - h*s*rw2/(2*k*sigma)
            AP[0,1] = -h*s*rho*(mu-sigma)/(2*mu*sigma) + h*(c*hk-s)*(mu**2 - sigma**2)*(rho*omega)**2/(2*tkms**2)
            AP[0,2] = rho*(c*hk*(mu-sigma)+s*(mu+sigma))/tkms - ((hk**2-1)*s*mu**2 - (3+hk**2)*s*sigma**2+c*hk*(mu**2+3*sigma**2))*(rho*omega)**2/(2*k*tkms**2)
            AP[0,3] = -s + (c*hk - s)*rw2/(2*mu*k**2)

            AP[1,1] = c - h*s*rw2/(2*k*mu)
            AP[1,2] = -s + (c*hk - s)*rw2/(2*mu*k**2)
            AP[1,3] = s*omega**2/k

            AP[2,0] = -s*omega**2/k
            AP[2,1] = -s + (c*hk-s)*rw2/(2*sigma*k**2)
            AP[2,2] = c - h*s*rw2/(2*k*sigma)

            AP[3,0] = -s + (c*hk-s)*rw2/(2*sigma*k**2)
            AP[3,1] = rho*(c*hk*(mu-sigma) - s*(mu+sigma))/tkms+ (c*hk*(3*mu**2+sigma**2)-s*((3+hk**2)*mu**2+(1-hk**2)*sigma**2))*(rho*omega)**2/(2*k*tkms**2)
            AP[3,2] = -h*s*rho*(mu-sigma)/(2*mu*sigma) + h*(c*hk-s)*(mu**2-sigma**2)*(rho*omega)**2/(2*tkms**2)
            AP[3,3] = c-h*s*rw2/(2*k*mu)
        else:
            rw2 = rho * omega**2
            hk = h*k
            tms = 2*mu*sigma
            scale=0
            AP[0,0] = 1 + hk**2 / 2 - (h**2 *rho /(2*sigma) + h**4*k**2*rho/(12*sigma))*omega**2
            AP[0,1] = -h**2*k*rho*(mu-sigma)/tms + h**4*k*rho**2*(mu**2 - sigma**2)*omega**2/(6*tms**2)
            AP[0,2] = h*rho/sigma + k**2*rho*(2*h**3*mu - h**3*sigma)/(3*tms)+(k**2 *(h**5/(120*mu**2)-h**5/(40*sigma**2))-h**3/6*sigma**2)*(rho*omega)**2
            AP[0,3] = -hk + h**3*k*rho*omega**2/(6*mu)

            AP[1,1] = 1+hk**2/2-(h**2/(2*mu)+h**4*k**2/(12*mu))*rw2
            AP[1,2] = -hk+h**3*k*rw2/(6*mu)
            AP[1,3] = h*(1+hk**2/6)*omega**2

            AP[2,0] = -h*(1+hk**2/6)*omega**2
            AP[2,1] = hk*(h**2*rw2/(6*sigma)-1)
            AP[2,2] = 1+hk**2/2-h**2*(1+hk**2/6)*rw2

            AP[3,0] = hk*(h**2*rw2/(6*sigma)-1)
            AP[3,1] = h*rho*(-1 + hk**2 *(mu-2*sigma)/(6*sigma)+(1-hk**2*(mu**2-3*sigma**2)/(20*sigma**2))*h**2*rw2/(6*mu))/mu
            AP[3,2] = h**2*k*rho*(mu-sigma)*(h**2*(mu+sigma)*rw2/(12*mu*sigma)-1)/(2*mu*sigma)
            AP[3,3] = 1+ hk**2/2 - h**2*(1+hk**2/6)*rw2/(2*mu)
    #return Z, AP, iZ,scale
    return ScaledMatrix(Z,0,True) @ ScaledMatrix(AP,2*scale,True) @ ScaledMatrix(iZ,0,True), ScaledMatrix(minors(Z),0,True) @ ScaledMatrix(minors(AP),2*scale,True) @ ScaledMatrix(minors(iZ),0,True)

def freeSurfaceBoundary():
    return ScaledMatrix(np.array([1,0,0,0,0,0]),0)
def oceanFloorBoundary(depth,omega,k,sigma,rho,eps_om=1e-3,eps_k=1e-5):
    if k>eps_k:
        zsig = np.lib.scimath.sqrt(k**2 - rho*omega**2/sigma)
        t = np.exp(-2*depth*zsig)
        m = ScaledMatrix(np.array([1+t,0,0,-rho*omega**2*(1-t)/zsig,0,0]),0)
    else:
        if abs(omega)>eps_om:
            zsig = np.lib.scimath.sqrt(-rho*omega**2/sigma)
            t = np.exp(-2*depth*zsig)
            m = ScaledMatrix(np.array([0.5*(1+t) + depth*(1-t)*k**2/(4*zsig),0,0, \
                                        0.5*(t-1)*np.lib.scimath.sqrt(-rho*sigma*omega**2)+sigma*(t-1+depth*zsig*(1+t))*k**2/(4*zsig),0,0]),0)
        else:
            m=ScaledMatrix(np.array([1,0,0,0,0,0]),0)
    return m
def underlyingHalfspaceBoundary(omega,k,sigma,mu,rho,eps_om=1e-3,eps_k=1e-5):
    if k>eps_k:
        zsig = np.lib.scimath.sqrt(k**2 - rho*omega**2/sigma)
        zmu = np.lib.scimath.sqrt(k**2 - rho*omega**2/mu)
        xi = (rho*omega**2/sigma - k**2*(1+mu/sigma))/(k**2+zsig*zmu)
        m =  ScaledMatrix(np.array([xi/mu,2*k*(xi+0.5),-zsig,zmu,-2*k*(xi+0.5), \
                    rho*omega**2 - 4*k**2 * mu*(xi+1)]),0)
    else:
        if abs(omega) > eps_om:
            rms = np.sqrt(mu*sigma)
            m = ScaledMatrix(np.array([-rho/rms + k**2*(mu+sigma-2*rms)/(2*rms*omega**2),
                                        k*rho*(1-2*np.sqrt(mu/sigma)),
                                        np.sqrt(rho/sigma)*1j*omega*(sigma*(k/omega)**2/2 - rho),
                                        np.sqrt(rho/mu)*1j*omega*(-mu*(k/omega)**2/2 + rho),
                                        -k*rho*(1-2*np.sqrt(mu/sigma)),
                                        rho*(4*k**2*mu*(np.sqrt(mu/sigma)-1)+rho*omega**2)
                                        ]))
        else:
            m = ScaledMatrix(np.array([1,0,0,0,0,0]),0)
    return m


S = Stack([Layer(3.,1.8,0.,1.02),
           Layer(2.,4.5,2.4,2.57),
           Layer(5.,5.8,3.3,2.64),
           Layer(20.,6.5,3.65,2.85),
           Layer(np.inf,8.,4.56,3.34)])
S.insertSource(30.)
S.insertReceiver(3.2)
M = np.array([[2.,0.3,0.],
              [0.3,-0.8,0.1],
              [0.,0.1,1.2]])
F = np.zeros([3])
