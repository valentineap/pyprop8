import numpy as np
import scaledmatrix as scm
import scipy.special as spec
import tqdm
import warnings




class PointSource:
    '''
    Object to encapsulate a point source. Source representation
    is a moment tensor and/or a 3-component force vector acting
    at a single point in space and time. Multiple moment tensors
    and force vectors may be specified, in which case calculations
    will be performed for each separately.
    '''
    def __init__(self,lat,lon,dep,Mxyz,F,time):
        self.lat = lat
        self.lon = lon
        self.dep = dep
        self.time = time
        assert len(Mxyz.shape)==len(F.shape),"Mxyz and F should have matching numbers of dimensions"
        assert Mxyz.shape[-2:]==(3,3),"Moment tensor must be (Nx)3x3"
        assert F.shape[-2:]==(3,1),"Force vector must be (Nx)3x1"
        if len(Mxyz.shape)==3:
            assert Mxyz.shape[0]==F.shape[0],"Mxyz and F should have matching first dimension"
            self.n_sources = Mxyz.shape[0]
            self.Mxyz = Mxyz.copy()
            self.F = F.copy()
        elif len(Mxyz.shape)==2:
            self.n_sources = 1
            self.Mxyz = Mxyz.reshape(1,3,3)
            self.F = F.reshape(1,3,1)
        else:
            raise ValueError("Moment tensor should be (Nx)3x3")
class LayeredStructureModel:
    '''Class to represent a layered earth model.'''
    def __init__(self,layers=None,interface_depth_form=False):
        '''
        Construct a model. If `interface_depth_form=True`, `layers` is
        expected to contain information about interface depths and the properties
        immediately below that depth (see `LayeredStructureModel.from_interface_list()`
        for more details). Otherwise, it is expected to contain an ordered sequence of
        layers specified by their thickness and material properties (see
        `LayeredStructureModel.from_layer_list()`).
        '''
        if layers is None:
            self.nlayers=None
            self.dz=None
            self.sigma=None
            self.mu=None
            self.rho = None
        else:
            if interface_depth_form:
                self.from_interface_list(layers)
            else:
                self.from_layer_list(layers)
    def from_layer_list(self,layers):
        '''
        Construct model from list-like structure where:
        i.   Layers are given in order from surface to base;
        ii.  Each layer is specified by four parameters: (thickness, Vp, Vs, density)
        iii. The final entry in the list has thickness 'np.inf' and represents
             the underlying infinite halfspace
        '''
        self.nlayers = len(layers)
        self.dz = np.zeros(self.nlayers)
        self.sigma = np.zeros(self.nlayers)
        self.mu = np.zeros(self.nlayers)
        self.rho = np.zeros(self.nlayers)
        for i,layer in enumerate(layers):
            self.dz[i],vp,vs,rho = layer
            if self.dz[i]<=0: raise ValueError("Layer thickness must be positive")
            if vp<=0: raise ValueError("P-wave velocity must be positive")
            if vs<0: raise ValueError("S-wave velocity must be positive or zero")
            if rho<=0: raise ValueError("Density must be positive")
            self.sigma[i] = rho*vp**2
            self.mu[i] = rho*vs**2
            self.rho[i] = rho
        if not np.isinf(self.dz[-1]): raise ValueError("Model should be terminated by 'infinite-thickness' layer representing underlying halfspace")
    def from_interface_list(self,layers):
        '''
        Construct model from list-like structure where:
        i.   Each entry in the list is specified by four parameters, (z, Vp, Vs, rho),
             where z is an interface depth and (Vp, Vs, rho) represent the properties
             immediately *below* that interface.
        ii.  The list need not be ordered. If there are duplicates in the set of interface depths, the
             first entry in the list will be used; others will be silently discarded.
        iii. There must be one entry with an interface depth of z=0, representing the free surface.
        iv.  The deepest interface is taken as specifying the properties of the underlying infinite half-space.
        '''
        nlayers = len(layers)
        zz = np.zeros(nlayers)
        vp = np.zeros(nlayers)
        vs = np.zeros(nlayers)
        rho = np.zeros(nlayers)
        for i,layer in enumerate(layers):
            zz[i],vp[i],vs[i],rho[i] = layer
        if not np.all(zz>=0): raise ValueError("All interface depths must be positive or zero")
        if not np.all(vp>0): raise ValueError("All Vp values must be positive")
        if not np.all(vs>=0): raise ValueError("All Vs values must be positive or zero")
        if not np.all(rho>0): raise ValueError("All density values must be positive")
        # Discard duplicates
        unique_z, index_z = np.unique(zz,return_index=True)
        zz = zz[index_z]
        vp = vp[index_z]
        vs = vs[index_z]
        rho = rho[index_z]
        sort_order = np.argsort(zz)
        if zz[sort_order[0]]!=0: raise ValueError("Must provide one interface at z=0")
        self.nlayers = len(zz+1)
        self.dz = np.zeros(self.nlayers)
        self.sigma = np.zeros(self.nlayers)
        self.mu = np.zeros(self.nlayers)
        self.rho = np.zeros(self.nlayers)
        for i in range(self.nlayers):
            if i == self.nlayers-1:
                self.dz[i] = np.inf
            else:
                self.dz[i] = zz[sort_order[i+1]]-zz[sort_order[i]]
            self.sigma[i] = rho[sort_order[i]]*vp[sort_order[i]]**2
            self.mu[i] = rho[sort_order[i]]*vs[sort_order[i]]**2
            self.rho[i] = rho[sort_order[i]]
    def with_interfaces(self,*interfaces):
        '''
        Return arrays representing the layered model, with additional interfaces
        inserted at the depths specified in ``*interfaces` (unless there is already
        an interface at that depth). These additional interfaces do not alter the
        material properties of the model (i.e. we end up with layers with duplicate
        properties) and used to ensure that the source and receiver depths coincide
        with an 'interface' as required by the theory.
        '''
        dz = self.dz.copy()
        sigma = self.sigma.copy()
        mu = self.mu.copy()
        rho = self.rho.copy()
        N = dz.shape[0]
        indices = []
        for interface in interfaces:
            z = 0
            for ilayer in range(N):
                if interface<z+dz[ilayer]: break
                z+=dz[ilayer]
            if interface>z:
                dz_ = np.zeros(N+1,dz.dtype)
                dz_[:ilayer] = dz[:ilayer]
                dz_[ilayer] = interface-z
                dz_[ilayer+1] = dz[ilayer]-(interface-z)
                dz_[ilayer+2:] = dz[ilayer+1:]
                sigma_ = np.zeros(N+1,sigma.dtype)
                sigma_[:ilayer+1] = sigma[:ilayer+1]
                sigma_[ilayer+1:] = sigma[ilayer:]
                mu_ = np.zeros(N+1,mu.dtype)
                mu_[:ilayer+1] = mu[:ilayer+1]
                mu_[ilayer+1:] = mu[ilayer:]
                rho_ = np.zeros(N+1,rho.dtype)
                rho_[:ilayer+1] = rho[:ilayer+1]
                rho_[ilayer+1:] = rho[ilayer:]
                for i,n in enumerate(indices):
                    if n>ilayer: indices[i]+=1
                ilayer+=1
                dz = dz_
                sigma = sigma_
                mu = mu_
                rho = rho_
                N+=1
            indices+=[ilayer]
        return tuple([dz,sigma,mu,rho]+indices)
    @property
    def vp(self):
        '''P-wave velocity'''
        return np.sqrt(self.sigma/self.rho)
    @property
    def vs(self):
        '''S-wave velocity'''
        return np.sqrt(self.mu/self.rho)
    def __repr__(self):
        '''Pretty-print model'''
        z = 0
        out = []
        for i in range(self.nlayers):
            out += ['------------------------------------------------------- z = %.2f km\n'%z]
            if self.vs[i]==0:
                out+=  ['  vp =%5.2f km/s       FLUID        rho =%5.2f g/cm^3\n'%(self.vp[i],self.rho[i])]
            else:
                out+=  ['  vp =%5.2f km/s   vs =%5.2f km/s   rho =%5.2f g/cm^3\n'%(self.vp[i],self.vs[i],self.rho[i])]
            z+=self.dz[i]
        return ''.join(out)
    def copy(self):
        '''Return a copy of the current model'''
        m = LayeredStructureModel()
        m.nlayers = self.nlayers
        m.dz = self.dz.copy()
        m.sigma = self.sigma.copy()
        m.mu = self.mu.copy()
        m.rho = self.rho.copy()
        return m




class ReceiverSet:
    def __init__(self):
        pass
    def validate(self):
        if np.any(self.rr<0): raise ValueError("Some receivers appear to be at negative radii")
        if np.any(self.rr>200): warnings.warn("Source-receiver distances exceed 200 km. Flat-earth approximation may not be appropriate. ",RuntimeWarning)


class RegularlyDistributedReceivers(ReceiverSet):
    def __init__(self,rmin,rmax,nr,phimin,phimax,nphi,depth = 0, degrees=True):
        super().__init__()
        self.nr = nr
        self.nphi = nphi
        self.rr = np.linspace(rmin,rmax,nr)
        self.pp = np.linspace(phimin,phimax,nphi)
        self.depth = depth
        if degrees: self.pp = np.deg2rad(self.pp)
    def as_xy(self):
        return np.outer(self.rr,np.cos(self.pp)),np.outer(self.rr,np.sin(self.pp))

class ListOfReceivers(ReceiverSet):
    def __init__(self):
        pass
    def from_xy(self,xx,yy,x0=0,y0=0,depth=0):
        assert xx.shape[0] == yy.shape[0]
        self.nr  = xx.shape[0]
        self.rr = np.zeros(self.nr)
        self.pp = np.zeros(self.nr)
        self.rr = np.sqrt((xx-x0)**2 + (yy-y0)**2)
        self.pp = np.arctan2(yy,xx)
        # for i,xy in enumerate(zip(xx,yy)):
        #     x,y=xy
        #     self.rr[i] = np.sqrt((x-x0)**2 + (y-y0)**2)
        #     self.pp[i] = np.arctan2(y,x)
        self.depth = depth




##################################
### Boundary condition vectors ###
##################################
def freeSurfaceBoundary(nk,sh=False):
    if sh:
        m = np.zeros([nk,2,1],dtype='complex128')
    else:
        m = np.zeros([nk,6,1],dtype='complex128')
    m[:,0,0] = 1
    return scm.ScaledMatrixStack(m)


def oceanFloorBoundary(depth,omega,k,sigma,rho,sh=False):
    nk = k.shape[0]
    if sh:
        m = np.zeros([nk,2,1],dtype='complex128')
        m[:,0,0] = 1
    else:
        zsig = np.lib.scimath.sqrt(k**2 - rho*omega**2/sigma)
        t = np.exp(-2*depth*zsig)
        m = np.zeros([nk,6,1],dtype='complex128')
        m[:,0,0] = 1+t
        if omega!=0: m[:,3,0] = -rho*omega**2*(1-t)/zsig
    return scm.ScaledMatrixStack(m)

def underlyingHalfspaceBoundary(omega,k,sigma,mu,rho,sh=False):
    nk = k.shape[0]

    zmu = np.lib.scimath.sqrt(k**2 - rho*omega**2/mu)
    if sh:
        m = np.zeros([nk,2,1],dtype='complex128')
        m[:,0,0] = 1
        m[:,1,0] = mu*zmu
    else:
        zsig = np.lib.scimath.sqrt(k**2 - rho*omega**2/sigma)
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
    return scm.ScaledMatrixStack(m)

def sourceVector(MT,F,k,sigma,mu):
    nk = k.shape[0]
    s = np.zeros([nk,4,5],dtype='complex128')
    s2 = np.zeros([nk,2,5],dtype='complex128')
    s[:,0,2] = MT[2,2]/sigma
    s[:,2,2] = -F[2]
    s[:,3,2] = 0.5*k*(MT[0,0]+MT[1,1]) - k*(sigma-2*mu)*MT[2,2]/sigma
    s2[:,1,2] =0.5*k*(MT[0,1]-MT[1,0])
    for sgn in [-1,1]:
        s[:,1,2+sgn] = 0.5*(sgn*MT[0,2] - 1j*MT[1,2])/mu
        s[:,2,2+sgn] = sgn*0.5*k*(MT[0,2]-MT[2,0])+0.5*1j*k*(MT[2,1]-MT[1,2])
        s[:,3,2+sgn] = 0.5*(-sgn*F[0]+1j*F[1])
        s[:,3,2+2*sgn] = 0.25*k*(MT[1,1]-MT[0,0])+sgn*0.25*1j*k*(MT[0,1]+MT[1,0])
        s2[:,0,2+sgn] = 0.5*(-sgn*MT[1,2]-1j*MT[0,2])/mu
        s2[:,1,2+sgn] = 0.5*(sgn*F[1]+1j*F[0])
        s2[:,1,2+2*sgn] = 0.25*k*(MT[0,1]+MT[1,0])-sgn*0.25*1j*k*(MT[1,1]-MT[0,0])
    return scm.ScaledMatrixStack(s),scm.ScaledMatrixStack(s2)

def sourceVector_ddep(MT,F,omega,k,sigma,mu,rho):
    lam = sigma-2*mu
    gamma = mu*(3*lam+2*mu)/(lam+2*mu)
    nk = k.shape[0]
    s = np.zeros([nk,4,5],dtype='complex128')
    s2 = np.zeros([nk,2,5],dtype='complex128')
    s[:,0,2] = F[2]/sigma
    s[:,1,2] = -k*(sigma*(MT[0,0]+MT[1,1])-2*(lam+mu)*MT[2,2])/(2*mu*sigma)
    s[:,2,2] = -0.5*k**2*(MT[0,0]+MT[1,1])+(k**2 * lam + rho*omega**2)*MT[2,2]/sigma
    s[:,3,2] = -k*lam*F[2]/sigma
    s2[:,0,2] = k*(-MT[0,1]+MT[1,0])/(2*mu)
    for sgn in [-1,1]:
        s[:,0,2+sgn] = k*((lam+mu)*(-sgn*MT[0,2]+1j*MT[1,2]) + mu*(sgn*MT[2,0]-1j*MT[2,1]))
        s[:,1,2+sgn] = (sgn*F[0]-1j*F[1])/(2*mu)
        s[:,2,2+sgn] = 0.5*k*(sgn*F[0]-1j*F[1])
        s[:,3,2+sgn] = (k**2 * lam*(-sgn*MT[2,0]+1j*MT[2,1])+(sgn*MT[0,2]-1j*MT[1,2])*(k**2 * (lam*mu - (gamma+mu)*sigma+rho*sigma*omega**2))/mu)/(2*sigma)
        s[:,1,2+2*sgn] = k*(MT[0,0] - MT[1,1])/(4*mu)-sgn*1j*k*(MT[0,1]+MT[1,0])/(4*mu)
        s[:,2,2+2*sgn] = 0.25*k**2*(MT[0,0] - MT[1,1])- 0.25*sgn*1j*k**2 *(MT[0,1]+MT[1,0])
        s2[:,0,2+sgn] = (-sgn*F[1]-1j*F[0])/(2*mu)
        s2[:,1,2+sgn] = (k**2 * mu - rho * omega**2)*(sgn*MT[1,2]+1j*MT[0,2])/(2*mu)
        s2[:,0,2+2*sgn] = -k*(MT[0,1]+MT[1,0])/(4*mu) + sgn*1j*k*(MT[1,1]-MT[0,0])/(4*mu)
    return scm.ScaledMatrixStack(s),scm.ScaledMatrixStack(s2)
###################
### Propagation ###
###################

def exphyp(x):
    a = np.real(x)
    b = np.imag(x)
    sgn = np.sign(a)
    t = np.exp(-2*sgn*a)
    return 0.5*np.cos(b)*(1+t)+0.5j*np.sin(b)*sgn*(1-t),0.5*np.cos(b)*sgn*(1-t)+0.5j*np.sin(b)*(1+t),sgn*a

def propagate_general(omega,k,dz,sigma,mu,rho,m2=None,m4=None,m6=None,inplace=True):
    # Propagate the systems in m2/m4/m6 through layer.
    nk = k.shape[0]
    # try:
    #     if omega.shape[0] != nk: raise ValueError('ww should have same dimension as kk')
    # except AttributeError, IndexError:
    #     # omega is a float; continue
    #     pass

    if m4 is not None or m6 is not None:
        zsig = np.lib.scimath.sqrt(k**2 - rho*omega**2/sigma)
        csig,ssig,scalesig = exphyp(dz*zsig)
    zmu = np.lib.scimath.sqrt(k**2 - rho*omega**2/mu)
    cmu,smu,scalemu = exphyp(dz*zmu)
    if m2 is not None:
        M = np.zeros((nk,2,2),dtype='complex128')
        M[:,0,0] = cmu
        M[:,0,1] = smu/(mu*zmu)
        M[:,1,0] = mu*zmu*smu
        M[:,1,1] = cmu
        if inplace:
            out = m2
        else:
            out = None
        m2r = scm.ScaledMatrixStack(M,scalemu.copy()).matmul(m2,out=out)
        del M
    else:
        m2r = None
    if m4 is not None:
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
        M = scm.ScaledMatrixStack(exphap_s,scalesig.copy())
        del exphap_s #Don't need explicit reference; reference in M still exists.

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

        M+=scm.ScaledMatrixStack(exphap_m,scalemu.copy())
        del exphap_m

        rtrho = np.sqrt(rho)

        Z = np.zeros([nk,4,4],dtype='complex128')
        Z[:,0,0] = 1/rtrho
        Z[:,1,3] = -1/rtrho
        Z[:,2,2] = rtrho
        Z[:,2,3] = -2*mu*k/rtrho
        Z[:,3,0] = 2*mu*k/rtrho
        Z[:,3,1] = rtrho

        iZ = np.zeros([nk,4,4],dtype='complex128')
        iZ[:,0,0] = rtrho
        iZ[:,1,0] = -2*mu*k/rtrho
        iZ[:,1,3] = 1/rtrho
        iZ[:,2,1] = -2*mu*k/rtrho
        iZ[:,2,2] = 1/rtrho
        iZ[:,3,1] = -rtrho
        if inplace:
            out = m4
        else:
            out = None
        m4r = scm.ScaledMatrixStack(Z).matmul(M.matmul(scm.ScaledMatrixStack(iZ).matmul(m4,out=out),out=out),out=out)
        del Z,iZ,M
    else:
        m4r = None
    if m6 is not None:
        xiprod = zsig*zmu/k**2
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
        M = scm.ScaledMatrixStack(M1,scalemu+scalesig)
        del M1

        M2 = np.zeros([nk,6,6],dtype='complex128')
        M2[:,1,1] = xiprod
        M2[:,1,3] = xiprod
        M2[:,2,1] = xiprod
        M2[:,2,3] = 2*xiprod
        M2[:,2,4] = -xiprod
        M2[:,4,3] = -xiprod
        M2[:,4,4] = xiprod
        M += scm.ScaledMatrixStack(M2)
        del M2

        Z = np.zeros([nk,6,6],dtype='complex128')
        Z[:,0,2] = -k**2/(rho*omega**2)
        Z[:,1,1] = k
        Z[:,1,2] = -2*k**3*mu/(rho*omega**2)
        Z[:,2,0] = zsig
        Z[:,3,5] = zmu
        Z[:,4,2] = 2*k**3*mu/(rho*omega**2)
        Z[:,4,4] = k
        Z[:,5,1] = -2*k**2*mu
        Z[:,5,2] = 4*k**4*mu**2 / (rho*omega**2)
        Z[:,5,3] = -rho*omega**2
        Z[:,5,4] = 2*k**2*mu


        iZ = np.zeros([nk,6,6],dtype='complex128')
        iZ[:,0,2] = 1/zsig
        iZ[:,1,0] = -2*k**2*mu/(zsig*zmu)
        iZ[:,1,1] = k/(zsig*zmu)
        iZ[:,2,0] = -rho*omega**2/(zsig*zmu)
        iZ[:,3,0] = 4*k**4*mu**2/(rho*omega**2 *zsig*zmu)
        iZ[:,3,1] = -2*k**3*mu/(rho*omega**2 * zsig*zmu)
        iZ[:,3,4] = 2*k**3*mu/(rho*omega**2 * zsig*zmu)
        iZ[:,3,5] = -k**2/(rho*omega**2 * zsig*zmu)
        iZ[:,4,0] = 2*k**2*mu/(zsig*zmu)
        iZ[:,4,4] = k/(zsig*zmu)
        iZ[:,5,3] = 1/zmu
        if inplace:
            out = m6
        else:
            out = None
        m6r = scm.ScaledMatrixStack(Z).matmul(M.matmul(scm.ScaledMatrixStack(iZ).matmul(m6,out=out),out=out),out=out)
    else:
        m6r = None
    return m2r,m4r,m6r

class IndexableNone:
    '''A do-nothing object that nevertheless can be sliced without raising exceptions'''
    def __init__(self):
        pass
    def __getitem__(self,slice):
        return None
    def __setitem__(self,slice,value):
        pass
def propagate(omega,k,dz,sigma,mu,rho,m2=None,m4=None,m6=None,inplace=True):
    if not inplace:
        if m2 is not None:m2 = m2.copy()
        if m4 is not None:m4 = m4.copy()
        if m6 is not None:m6 = m6.copy()
        # and now we can work 'in place'
    if m2 is None: m2 = IndexableNone()
    if m4 is None: m4 = IndexableNone()
    if m6 is None: m6 = IndexableNone()

    ksel = k==0
    m2[ksel] = 0.
    m4[ksel] = 0.
    m6[ksel] = 0

    ksel = k>0
    m2[ksel],m4[ksel],m6[ksel] = propagate_general(omega,k[ksel],dz,sigma,mu,rho,m2[ksel],m4[ksel],m6[ksel],True)
    # Hide the evidence...
    if type(m2) is IndexableNone: m2 = None
    if type(m4) is IndexableNone: m4 = None
    if type(m6) is IndexableNone: m6 = None
    return m2,m4,m6
    # return propagate_general(omega,k,dz,sigma,mu,rho,m2,m4,m6,inplace)
def makeN(s):
    m = s.M
    N = np.zeros([s.nStack,4,4],dtype='complex128')
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
    return scm.ScaledMatrixStack(N,s.scale.copy())
def makeDelta(scm1,scm2,sh=False):
    m1 = scm1.M
    m2 = scm2.M
    if not scm1.nStack==scm2.nStack: raise ValueError("Dimension mismatch")
    m = np.zeros([scm1.nStack,1,1],dtype='complex128')
    if sh:
        m[:,0,0] = m1[:,0,0]*m2[:,1,0] - m1[:,1,0]*m2[:,0,0]
    else:
        m[:,0,0] = m1[:,0,0]*m2[:,5,0] - m1[:,1,0]*m2[:,4,0] + m1[:,2,0]*m2[:,3,0] + m1[:,3,0]*m2[:,2,0] - m1[:,4,0]*m2[:,1,0] + m1[:,5,0]*m2[:,0,0]
    return scm.ScaledMatrixStack(m,scm1.scale+scm2.scale)

def kIntegrationStencil(kmin,kmax,nk):
    kk = np.linspace(kmin,kmax,nk)
    wts = np.full(nk,kk[1]-kk[0])
    wts[0] *= 0.5
    wts[-1] *= 0.5
    return kk,wts

def compute_spectra(structure, source, stations, omegas, derivatives = None, show_progress = True,
                    stencil = kIntegrationStencil, stencil_kwargs = {'kmin':0,'kmax':2.04,'nk':1200} ):
    # Compute spectra for (possibly multiple) receivers located at the
    # same depth on or below the surface.
    stations.validate()
    omegas = np.atleast_1d(omegas)
    nomegas = omegas.shape[0]

    do_derivatives = False
    if derivatives is not None:
        if derivatives.nderivs > 0:
            do_derivatives = True


    nr = stations.nr
    rr_inv = 1/stations.rr
    nsources = source.n_sources

    k,k_wts = stencil(**stencil_kwargs)
    k_wts/=(2*np.pi)

    dz,sigma,mu,rho,isrc,irec = structure.with_interfaces(source.dep,stations.depth)
    assert irec<isrc,"Receivers must be above source"

    # Set up Bessel function arrays
    mm = np.arange(-2,3)
    jv = spec.jv(np.tile(mm,nr*nk),np.outer(k,stations.rr).repeat(5)).reshape(nk,nr,5)
    jvp = spec.jvp(np.tile(mm,nr*nk),np.outer(k,stations.rr).repeat(5)).reshape(nk,nr,5)
    if do_derivatives:
        if derivatives.moment_tensor:
            d_Mxyz = np.array([[[1,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,1]],
                              [[0,1,0],[1,0,0],[0,0,0]],[[0,0,1],[0,0,0],[1,0,0]],[[0,0,0],[0,0,1],[0,1,0]]],dtype='float64')
        if derivatives.force:
            d_F = np.array([[[1],[0],[0]],[[0],[1],[0]],[[0],[0],[1]]],dtype='float64')
        if derivatives.r:
            djvp_dr = spec.jvp(np.tile(mm,nr*nk),np.outer(k,stations.rr).repeat(5),2).reshape(nk,nr,5)*k.reshape(-1,1,1)
    # Allocate output data arrays
    if type(stations) is RegularlyDistributedReceivers:
        spectra = np.zeros([nsources,stations.nr,stations.nphi,3,nomegas],dtype='complex128')
        if do_derivatives: d_spectra = np.zeros([nsources,stations.nr,stations.nphi,derivatives.nderivs,3,nomegas],dtype='complex128')
        ss = slice(None)
        es1 = 'k,ksm,krm,mp->srp'
        es1d = 'k,ksdm,krm,mp->srpd'
        es2 = 'm,ksm,krm,r,k,mp->srp'
        es2d = 'm,ksdm,krm,r,k,mp->srpd'
        es3 = 'k,ksm,krm,m,mp->srp'
    elif type(stations) is ListOfReceivers:
        spectra = np.zeros([nsources,stations.nr,1,3,nomegas],dtype='complex128')
        if do_derivatives: d_spectra = np.zeros([nsources,stations.nr,1,derivatives.nderivs,3,nomegas],dtype='complex128')
        ss = 0
        es1 = 'k,ksm,krm,mr->sr'
        es1d = 'k,ksdm,krm,mr->srd'
        es2 = 'm,ksm,krm,r,k,mr->sr'
        es2d = 'm,ksdm,krm,r,k,mr->srd'
        es3 = 'k,ksm,krm,m,mr->sr'
    else:
        raise NotImplementedError


    if show_progress: t = tqdm.tqdm(total = nomegas)

    plan_1 = False
    plan_1d = False
    plan_2 = False
    plan_2d = False
    plan_3 = False
    determine_optimal_plan=True

    eimphi = np.exp(np.outer(1j*mm,stations.pp))/(2*np.pi)r
    for iom,omega in enumerate(omegas):
        H_psv,H_sh = compute_H_matrices(k[k!=0],omega,dz,sigma,mu,rho,isrc,irec)
        b = np.zeros([nk,nsources,6,5],dtype='complex128')
        for i in range(nsources):
            s_psv,s_sh = sourceVector(source.Mxyz[i,:,:],source.F[i,:,0],k[k!=0],sigma[isrc],mu[isrc])
            b[k!=0,i,:4,:] = (H_psv@s_psv).value
            b[k!=0,i,4:,:] = (H_sh@s_sh).value
        if do_derivatives:
            d_b = np.zeros([nk,nsources,derivatives.nderivs,6,5],dtype='complex128')
            if derivatives.moment_tensor:
                j0 = derivatives.i_mt
                for j in range(6):
                    #print(j)
                    s_psv,s_sh = sourceVector(d_Mxyz[j,:,:],np.zeros([3]),k[k!=0],sigma[isrc],mu[isrc])
                    for i in range(nsources):
                        d_b[k!=0,i,j0+j,:4,:] = (H_psv@s_psv).value
                        d_b[k!=0,i,j0+j,4:,:] = (H_sh@s_sh).value
            if derivatives.force:
                j0 = derivatives.i_f
                for j in range(3):
                    s_psv,s_sh = sourceVector(np.zeros(3,3),d_F[j,:,0],k[k!=0],sigma[isrc],mu[isrc])
                    for i in range(nsources):
                        d_b[k!=0,i,j0+j,:4,:] = (H_psv@s_psv).value
                        d_b[k!=0,i,j0+j,4:,:] = (H_sh@s_sh).value
            if derivatives.depth:
                j0 = derivatives.i_dep
                for i in range(nsources):
                    s_psv,s_sh = sourceVector_ddep(source.Mxyz[i,:,:],source.F[i,:,0],omega,k[k!=0],sigma[isrc],mu[isrc],rho[isrc])
                    d_b[k!=0,i,j0,:4,:] = (H_psv@s_psv).value
                    d_b[k!=0,i,j0,4:,:] = (H_sh@s_sh).value
        del H_psv,H_sh,s_psv,s_sh
        if determine_optimal_plan and iom==0:
            # First time through, determine optimal contraction schemes
            plan_1,_ = np.einsum_path(es1,k*k_wts,b[:,:,1,:],jvp,eimphi)
            plan_2,_ = np.einsum_path(es2,1j*mm,b[:,:,4,:],jv,rr_inv,k_wts,eimphi)
            plan_3,_ = np.einsum_path(es3,k*k_wts,b[:,:,1,:],jvp,1j*mm,eimphi)
            if do_derivatives:
                plan_1d,_ = np.einsum_path(es1d,k*k_wts,d_b[:,:,j0:j0+6,1,:],jvp,eimphi)
                plan_2d,_ = np.einsum_path(es2d,1j*mm,d_b[:,:,j0:j0+6,4,:],jv,rr_inv,k_wts,eimphi)
        spectra[:,:,ss,0,iom] = np.einsum(es1,k*k_wts,b[:,:,1,:],jvp,eimphi,optimize=plan_1)+ \
                        np.einsum(es2,1j*mm,b[:,:,4,:],jv,rr_inv,k_wts,eimphi,optimize=plan_2)
        spectra[:,:,ss,1,iom] = np.einsum(es2,1j*mm,b[:,:,1,:],jv,rr_inv,k_wts,eimphi,optimize=plan_2)-\
                        np.einsum(es1,k*k_wts,b[:,:,4,:],jvp,eimphi,optimize=plan_1)
        spectra[:,:,ss,2,iom] = np.einsum(es1,k*k_wts,b[:,:,0,:],jv,eimphi,optimize=plan_1)
        if do_derivatives:
            if derivatives.moment_tensor:
                j0 = derivatives.i_mt
                d_spectra[:,:,ss,j0:j0+6,0,iom] = np.einsum(es1d,k*k_wts,d_b[:,:,j0:j0+6,1,:],jvp,eimphi,optimize=plan_1d)+ \
                                np.einsum(es2d,1j*mm,d_b[:,:,j0:j0+6,4,:],jv,rr_inv,k_wts,eimphi,optimize=plan_2d)
                d_spectra[:,:,ss,j0:j0+6,1,iom] = np.einsum(es2d,1j*mm,d_b[:,:,j0:j0+6,1,:],jv,rr_inv,k_wts,eimphi,optimize=plan_2d)-\
                                np.einsum(es1d,k*k_wts,d_b[:,:,j0:j0+6,4,:],jvp,eimphi,optimize=plan_1d)
                d_spectra[:,:,ss,j0:j0+6,2,iom] = np.einsum(es1d,k*k_wts,d_b[:,:,j0:j0+6,0,:],jv,eimphi,optimize=plan_1d)
            if derivatives.force:
                j0 = derivatives.i_f
                d_spectra[:,:,ss,j0:j0+3,0,iom] = np.einsum(es1d,k*k_wts,d_b[:,:,j0:j0+3,1,:],jvp,eimphi,optimize=plan_1d)+ \
                                np.einsum(es2d,1j*mm,d_b[:,:,j0:j0+3,4,:],jv,rr_inv,k_wts,eimphi,optimize=plan_2d)
                d_spectra[:,:,ss,j0:j0+3,1,iom] = np.einsum(es2d,1j*mm,d_b[:,:,j0:j0+3,1,:],jv,rr_inv,k_wts,eimphi,optimize=plan_2d)-\
                                np.einsum(es1d,k*k_wts,d_b[:,:,j0:j0+3,4,:],jvp,eimphi,optimize=plan_1d)
                d_spectra[:,:,ss,j0:j0+3,2,iom] = np.einsum(es1d,k*k_wts,d_b[:,:,j0:j0+3,0,:],jv,eimphi,optimize=plan_1d)
            if derivatives.r:
                j0 = derivatives.i_r
                d_spectra[:,:,ss,j0,0,iom] = np.einsum(es1,k*k_wts,b[:,:,1,:],djvp_dr,eimphi,optimize=plan_1) - \
                                        np.einsum(es2,1j*mm,b[:,:,4,:],jv,rr_inv**2,k_wts,eimphi,optimize=plan_2) +\
                                        np.einsum(es2,1j*mm,b[:,:,4,:],jvp,rr_inv,k*k_wts,eimphi,optimize=plan_2)
                d_spectra[:,:,ss,j0,1,iom] = np.einsum(es2,-1j*mm,b[:,:,1,:],jv,rr_inv**2,k_wts,eimphi,optimize=plan_2) +\
                                        np.einsum(es2,1j*mm,b[:,:,1,:],jvp,rr_inv,k*k_wts,eimphi,optimize=plan_2) -\
                                        np.einsum(es1,k*k_wts,b[:,:,4,:],djvp_dr,eimphi,optimize=plan_1)
                d_spectra[:,:,ss,j0,2,iom] = np.einsum(es1,k*k*k_wts,b[:,:,0,:],jvp,eimphi,optimize=plan_1)
            if derivatives.phi:
                j0 = derivatives.i_phi
                d_spectra[:,:,ss,j0,0,iom] = np.einsum(es3,k*k_wts,b[:,:,1,:],jvp,1j*mm,eimphi,optimize=plan_3)+ \
                                np.einsum(es2,-mm*mm,b[:,:,4,:],jv,rr_inv,k_wts,eimphi,optimize=plan_2)
                d_spectra[:,:,ss,j0,1,iom] = np.einsum(es2,mm*mm,b[:,:,1,:],jv,rr_inv,k_wts,eimphi,optimize=plan_2)-\
                                np.einsum(es3,k*k_wts,b[:,:,4,:],jvp,-1j*mm,eimphi,optimize=plan_3)
                d_spectra[:,:,ss,j0,2,iom] = np.einsum(es3,k*k_wts,b[:,:,0,:],jv,-1j*mm,eimphi,optimize=plan_3)
            if derivatives.depth:
                j0 = derivatives.i_dep
                d_spectra[:,:,ss,j0,0,iom] = np.einsum(es1,k*k_wts,d_b[:,:,j0,1,:],jvp,eimphi,optimize=plan_1)+ \
                                np.einsum(es2,1j*mm,d_b[:,:,j0,4,:],jv,rr_inv,k_wts,eimphi,optimize=plan_2)
                d_spectra[:,:,ss,j0,1,iom] = np.einsum(es2,1j*mm,d_b[:,:,j0,1,:],jv,rr_inv,k_wts,eimphi,optimize=plan_2)-\
                                np.einsum(es1,k*k_wts,d_b[:,:,j0,4,:],jvp,eimphi,optimize=plan_1)
                d_spectra[:,:,ss,j0,2,iom] = np.einsum(es1,k*k_wts,d_b[:,:,j0,0,:],jv,eimphi,optimize=plan_1)
            if derivatives.time:
                j0 = derivatives.i_time
                d_spectra[:,:,ss,j0,:,iom] = -1j*omega*spectra[:,:,ss,:,iom]
        if show_progress: t.update(1)
    if show_progress:t.close()
    if do_derivatives:
        return spectra.squeeze(),d_spectra.squeeze()
    else:
        return spectra.squeeze()



def compute_H_matrices(k,omega,dz,sigma,mu,rho,isrc,irec):
    nlayers = dz.shape[0]
    # Propagate surface b/c to receiver
    if mu[0] == 0:
        surface_bc_sh = oceanFloorBoundary(dz[0],omega,k,sigma[0],rho[0],True)
        surface_bc_psv = oceanFloorBoundary(dz[0],omega,k,sigma[0],rho[0],False)
        ibc = 1
    else:
        surface_bc_sh = freeSurfaceBoundary(k.shape[0],True)
        surface_bc_psv = freeSurfaceBoundary(k.shape[0],False)
        ibc = 0
    for i in range(ibc,irec):
        surface_bc_sh, _, surface_bc_psv = propagate(omega,k,-dz[i],sigma[i],mu[i],rho[i],m2=surface_bc_sh,m6=surface_bc_psv)
    # Propagate basal b/c to source depth
    basal_bc_sh = underlyingHalfspaceBoundary(omega,k,sigma[-1],mu[-1],rho[-1],True)
    basal_bc_psv = underlyingHalfspaceBoundary(omega,k,sigma[-1],mu[-1],rho[-1],False)
    for i in range(nlayers-2,isrc-1,-1):
        #print(i,dz[i],sigma[i],mu[i],rho[i])
        basal_bc_sh,_,basal_bc_psv = propagate(omega,k,dz[i],sigma[i],mu[i],rho[i],m2=basal_bc_sh,m6=basal_bc_psv)
    basal_bc_sh_at_src = basal_bc_sh.copy()
    # basal_bc_psv_at_src = basal_bc_psv.copy()
    # Create N and continue to propagate everything up to receiver
    N = makeN(basal_bc_psv)
    for i in range(isrc-1,irec-1,-1):
        basal_bc_sh,N,basal_bc_psv = propagate(omega,k,dz[i],sigma[i],mu[i],rho[i],m2=basal_bc_sh,m4=N,m6=basal_bc_psv)
    # Now assemble H
    H_psv = (makeN(surface_bc_psv) @ N)/makeDelta(surface_bc_psv,basal_bc_psv)
    H_sh = np.zeros([k.shape[0],2,2],dtype='complex128')
    H_sh[:,0,0] = surface_bc_sh.M[:,0,0]*basal_bc_sh_at_src.M[:,1,0]
    H_sh[:,0,1] = - surface_bc_sh.M[:,0,0]*basal_bc_sh_at_src.M[:,0,0]
    H_sh[:,1,0] = surface_bc_sh.M[:,1,0]*basal_bc_sh_at_src.M[:,1,0]
    H_sh[:,1,1] = -surface_bc_sh.M[:,1,0]*basal_bc_sh_at_src.M[:,0,0]
    H_sh = scm.ScaledMatrixStack(H_sh,surface_bc_sh.scale+basal_bc_sh_at_src.scale)/makeDelta(surface_bc_sh,basal_bc_sh,sh=True)
    return H_psv,H_sh


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
def clp(w,w0,w1):
    if np.real(w)<w0:
        return 1.
    elif np.real(w)<w1:
        return 0.5*(1+np.cos(np.pi*(w-w0)/(w1-w0)))
    else:
        return 0

def compute_seismograms(structure, source, stations, nt,dt,alpha,
                        source_time_function=None,pad_frac=0.25,kind ='displacement',
                        return_spectra = False,derivatives=None,show_progress = True,**kwargs):
    npad = int(pad_frac*nt)
    tt = np.arange(nt+npad)*dt
    ww = 2*np.pi*np.fft.rfftfreq(nt+npad,dt)-alpha*1j
    if derivatives is None:
        do_derivatives = False
    else:
        if derivatives.nderivs == 0:
            # Nothing actually turned on...
            do_derivatives = False
        else:
            do_derivatives = True

    spectra = compute_spectra(structure,source,stations,ww,derivatives,show_progress,**kwargs)
    if do_derivatives:
        spectra,d_spectra = spectra

    spec_shape = spectra.shape
    spec_shape_n = len(spec_shape)
    if kind == 'displacement':
        spectra /= 1j*ww.reshape((spec_shape_n-1)*[1]+[-1])
        if do_derivatives:d_spectra /= 1j*ww.reshape((spec_shape_n)*[1]+[-1])
    elif kind == 'velocity':
        pass
    elif kind == 'acceleration':
        spectra *= 1j*ww.reshape((spec_shape_n-1)*[1]+[-1])
        if do_derivatives:d_spectra *= 1j*ww.reshape((spec_shape_n)*[1]+[-1])
    else:
        raise ValueError("Unrecognised seismogram kind '%s'; should be one of 'displacement', 'velocity' or 'acceleration'."%kind)
    if source_time_function is not None:
        stf = np.zeros(ww.shape[0],dtype='complex128')
        for i,w in enumerate(ww):
            stf[i] = source_time_function(w)
        spectra *= stf.reshape((spec_shape_n-1)*[1]+[-1])
        if do_derivatives: d_spectra *= stf.reshape((spec_shape_n)*[1]+[-1])
    # Inverse FFT
    seis = (nt+npad)*np.fft.irfft(spectra)/(2*np.pi)
    # Discard 'padding' and scale by exp(alpha t)
    seis = seis[tuple((spec_shape_n-1)*[slice(None)]+[slice(None,nt)])]*np.exp(alpha*tt[:nt]).reshape((spec_shape_n-1)*[1]+[-1])
    if do_derivatives:
        deriv = (nt+npad)*np.fft.irfft(d_spectra)/(2*np.pi)
        deriv = deriv[tuple((spec_shape_n)*[slice(None)]+[slice(None,nt)])]*np.exp(alpha*tt[:nt]).reshape((spec_shape_n)*[1]+[-1])
    if return_spectra:
        if do_derivatives:
            return tt[:nt],seis,deriv,ww,spectra
        else:
            return tt[:nt],seis,ww,spectra
    else:
        if do_derivatives:
            return tt[:nt],seis,deriv
        else:
            return tt[:nt],seis

class DerivativeSwitches:
    def __init__(self,moment_tensor = False, force = False,
                      r = False, phi = False, depth = False, time = False):
        self.moment_tensor = moment_tensor
        self.force = force
        self.r = r
        self.phi = phi
        self.depth = depth
        self.time = time
    @property
    def nderivs(self):
        n = 0
        if self.moment_tensor: n+=6
        if self.force: n+=3
        if self.r: n+=1
        if self.phi: n+=1
        if self.depth: n+=1
        if self.time: n +=1
        return n
    @property
    def i_mt(self):
        if not self.moment_tensor: return None
        i = 0
        return i
    @property
    def i_f(self):
        if not self.force: return none
        i=0
        if self.moment_tensor: i+=6
        return i
    @property
    def i_r(self):
        if not self.r: return None
        i=0
        if self.moment_tensor: i+=6
        if self.force: i+=3
        return i
    @property
    def i_phi(self):
        if not self.phi: return None
        i=0
        if self.moment_tensor: i+=6
        if self.force: i+=3
        if self.r: i+=1
        return i
    @property
    def i_dep(self):
        if not self.depth: return None
        i=0
        if self.moment_tensor: i+=6
        if self.force: i+=3
        if self.r: i+=1
        if self.phi: i+=1
        return i
    @property
    def i_time(self):
        if not self.time: return None
        i=0
        if self.moment_tensor: i+=6
        if self.force: i+=3
        if self.r: i+=1
        if self.phi: i+=1
        if self.depth: i+=1
        return i
stations = RegularlyDistributedReceivers(20,150,5,0,360,8,depth=3)
model = LayeredStructureModel([(3.,1.8,0.,1.02),(2.,4.5,2.4,2.57),(5.,5.8,3.3,2.63),(20.,6.5,3.65,2.85),(np.inf,8.,4.56,3.34)])
source = PointSource(0,0,20,rtf2xyz(makeMomentTensor(330,90,0,2.4E8,0,0)),np.zeros([3,1]),0)
