import numpy as np
from pyprop8._propagators import *
import scipy.special as spec
HAS_MULTIPROCESSING = False
try:
    from multiprocessing import Pool
    HAS_MULTIPROCESSING = True
except:
    pass


# from tqdm.autonotebook import tqdm edited by MS 2/3/21 due to warning in Anaconda JupyterLab
import warnings

try:
    from tqdm import tqdm
except ImportError:
    print("Unable to import 'tqdm'; progress bars will not be shown.")
    # Create a do-nothing implementation
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, i):
            pass

        def close(self):
            pass


PLANETARY_RADIUS = 6371.0


def gc_dist(lat1, lon1, lat2, lon2, radius=PLANETARY_RADIUS):
    """
    Calculate the great-circle distance between two points on the surface of a
    sphere.

    :param float lat1: Latitude of point 1
    :param float lon1: Longitude of point 1
    :param float lat2: Latitude of point 2
    :param float lon2: Longitude of point 2 (all in radians)
    :param float radius: The radius of the sphere (km).

    :return: The great-circle distance (km).
    :rtype: float
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    u = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * radius * np.arcsin(np.sqrt(u))
    return dist


def gc_azimuth(lat1, lon1, lat2, lon2):
    """
    Azimuth cw from North to go *to* (lat2,lon2) from (lat1,lon1)

    :param float lat1: Latitude of point 1
    :param float lon1: Longitude of point 1
    :param float lat2: Latitude of point 2
    :param float lon2: Longitude of point 2 (all in radians)

    :return: The azimuth (radians).
    :rtype: float

    """
    dlon = lon2 - lon1
    return np.arctan2(
        np.cos(lat2) * np.sin(dlon),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )


class PointSource:
    def __init__(self, x, y, dep, Mxyz, F, time):
        """
        Create an object representing a seismic source. Source mechanism
        represented as the sum of a moment tensor and a force, acting at a
        single point in space. Alternatively N distinct moment tensor/force
        pairs may be specified; in this case, N distinct simulations are to be
        performed.

        :param float x: The x-location of the source.
        :param float y: The y-location of the source. Coordinate system must
            match that used for specifying receivers.
        :param float dep: The depth of the source, km. Must be positive (or
            zero).
        :param numpy.ndarray Mxyz: The moment tensor, expressed relative to a
            Cartesian system. Shape 3x3 or Nx3x3.
        :param numpy.ndarray F: The force vector, expressed relative to a
            Cartesian system. Shape 1x3 or Nx1x3.
        :param float or datetime.datetime time: The event time, expressed either
            as an instance of :py:class:`datetime.datetime`, or as seconds
            relative to some arbitrary epoch.
        """
        self.x = x
        self.y = y
        self.dep = dep
        self.time = time
        assert len(Mxyz.shape) == len(
            F.shape
        ), "Mxyz and F should have matching numbers of dimensions"
        assert Mxyz.shape[-2:] == (3, 3), "Moment tensor must be (Nx)3x3"
        assert F.shape[-2:] == (3, 1), "Force vector must be (Nx)3x1"
        if len(Mxyz.shape) == 3:
            assert (
                Mxyz.shape[0] == F.shape[0]
            ), "Mxyz and F should have matching first dimension"
            self.nsources = Mxyz.shape[0]
            self.Mxyz = Mxyz.copy()
            self.F = F.copy()
        elif len(Mxyz.shape) == 2:
            self.nsources = 1
            self.Mxyz = Mxyz.copy().reshape(1, 3, 3)
            self.F = F.copy().reshape(1, 3, 1)
        else:
            raise ValueError("Moment tensor should be (Nx)3x3")

    def copy(self):
        """
        Make a copy of the current source.
        """
        return PointSource(self.x, self.y, self.dep, self.Mxyz, self.F, self.time)


class LayeredStructureModel:
    def __init__(self, layers=None, interface_depth_form=False):
        """
        Construct an object to represent an earth model.

        :param list or None layers: Description of the layers that comprise the
            model. If ``interface_depth_form=False`` (default), list of tuples
            ``(thickness, vp, vs, rho)`` defining properties of each layer in
            turn, starting from the top; final entry must have thickness
            ``np.inf``. Alternatively if ``interface_depth_form=True``, list of
            tuples ``(ztop, vp, vs, rho)`` defining locations of interfaces
            within the model and the properties immediately *below* that
            interface; in this case one entry must have ``ztop=0``. If ``None``,
            'empty' model is returned.
        :param bool interface_depth_form: Select manner in which ``layers`` is
            specified; see above.

        Each instance provides the following variables, which should be treated
        as read-only:

        :ivar int or None nlayers: The number of layers in the model (including
            the infinite halfspace).
        :ivar numpy.ndarray or None dz: Array of layer thicknesses, from top to
            bottom.
        :ivar numpy.ndarray or None sigma: Array of P-wave moduli in each layer,
            from top to bottom.
        :ivar numpy.ndarray or None mu: Array of S-wave moduli in each layer,
            from top to bottom.
        :ivar numpy.ndarray or None rho: Array of densities in each layer, from
            top to bottom.
        """
        if layers is None:
            self.nlayers = None
            self.dz = None
            self.sigma = None
            self.mu = None
            self.rho = None
        else:
            if interface_depth_form:
                self.from_interface_list(layers)
            else:
                self.from_layer_list(layers)

    def from_layer_list(self, layers):
        """
        Parser used to initialise earth model from list of layer properties.
        Called when class is instantiated with ``interface_depth_form=False``;
        may be called separately to reset all model properties.

        :param list layers: Description of the layers that comprise the model.
            List of tuples ``(thickness, vp, vs, rho)`` defining properties of
            each layer in turn, starting from the top; final entry must have
            thickness ``np.inf`` and represents the underlying infinite
            halfspace.
        """
        self.nlayers = len(layers)
        self.dz = np.zeros(self.nlayers)
        self.sigma = np.zeros(self.nlayers)
        self.mu = np.zeros(self.nlayers)
        self.rho = np.zeros(self.nlayers)
        for i, layer in enumerate(layers):
            self.dz[i], vp, vs, rho = layer
            if self.dz[i] <= 0:
                raise ValueError("Layer thickness must be positive")
            if vp <= 0:
                raise ValueError("P-wave velocity must be positive")
            if vs < 0:
                raise ValueError("S-wave velocity must be positive or zero")
            if rho <= 0:
                raise ValueError("Density must be positive")
            self.sigma[i] = rho * vp**2
            self.mu[i] = rho * vs**2
            self.rho[i] = rho
        if not np.isinf(self.dz[-1]):
            raise ValueError(
                "Model should be terminated by 'infinite-thickness' layer representing underlying halfspace"
            )

    def from_interface_list(self, layers):
        """
        Parser used to initialise earth model from list of interface properties.
        Called when class is instantiated with ``interface_depth_from=True``;
        may be called separately to reset all model properties.

        :param list layers: Description of the layers that comprise the model.
            List of tuples ``(ztop, vp, vs, rho)`` defining locations of
            interfaces within the model and the properties immediately *below*
            that interface; in this case one entry must have ``ztop=0``. The
            list need not be ordered; if any entries exist with duplicate depths
            exist, the first will be used and others silently discarded. The
            deepest interface is taken to specify the properties of the
            underlying infinite halfspace.
        """
        nlayers = len(layers)
        zz = np.zeros(nlayers)
        vp = np.zeros(nlayers)
        vs = np.zeros(nlayers)
        rho = np.zeros(nlayers)
        for i, layer in enumerate(layers):
            zz[i], vp[i], vs[i], rho[i] = layer
        if not np.all(zz >= 0):
            raise ValueError("All interface depths must be positive or zero")
        if not np.all(vp > 0):
            raise ValueError("All Vp values must be positive")
        if not np.all(vs >= 0):
            raise ValueError("All Vs values must be positive or zero")
        if not np.all(rho > 0):
            raise ValueError("All density values must be positive")
        # Discard duplicates
        unique_z, index_z = np.unique(zz, return_index=True)
        zz = zz[index_z]
        vp = vp[index_z]
        vs = vs[index_z]
        rho = rho[index_z]
        sort_order = np.argsort(zz)
        if zz[sort_order[0]] != 0:
            raise ValueError("Must provide one interface at z=0")
        self.nlayers = len(zz + 1)
        self.dz = np.zeros(self.nlayers)
        self.sigma = np.zeros(self.nlayers)
        self.mu = np.zeros(self.nlayers)
        self.rho = np.zeros(self.nlayers)
        for i in range(self.nlayers):
            if i == self.nlayers - 1:
                self.dz[i] = np.inf
            else:
                self.dz[i] = zz[sort_order[i + 1]] - zz[sort_order[i]]
            self.sigma[i] = rho[sort_order[i]] * vp[sort_order[i]] ** 2
            self.mu[i] = rho[sort_order[i]] * vs[sort_order[i]] ** 2
            self.rho[i] = rho[sort_order[i]]

    def with_interfaces(self, *interfaces):
        """
        Return arrays describing the model, with additional (pseudo-)interfaces inserted at any depths passed as ``*args`` (unless there is already an interface at this depth). These additional interfaces do not alter the material properties (i.e. properties are identical above and below the interface), but are used internally to ensure that there is an interface at the source depth, as required for the calculation algorithm.

        :return: tuple ``(dz,sigma,mu,rho,indices,added)`` where:
            - ``dz`` - :py:class:`numpy.ndarray` containing thicknesses of each layer;
            - ``sigma`` - :py:class:`numpy.ndarray` containing P-wave modulus of each layer;
            - ``mu`` - :py:class:`numpy.ndarray` containing S-wave modulus of each layer;
            - ``rho`` - :py:class:`numpy.ndarray` containing density of each layer;
            - ``indices`` - list, possibly empty, containing index of each layer corresponding to an entry in ``*args``.
            - ``added`` - list, possibly empty, indicating whether an entry in ``*args`` required creation of an additional layer (``True``) or if an interface already existed at that depth (``False``).

        """
        dz = self.dz.copy()
        sigma = self.sigma.copy()
        mu = self.mu.copy()
        rho = self.rho.copy()
        N = dz.shape[0]
        indices = []
        pseudo = []
        for interface in interfaces:
            z = 0
            for ilayer in range(N):
                if interface < z + dz[ilayer]:
                    break
                z += dz[ilayer]
            added = False
            if interface > z:
                dz_ = np.zeros(N + 1, dz.dtype)
                dz_[:ilayer] = dz[:ilayer]
                dz_[ilayer] = interface - z
                dz_[ilayer + 1] = dz[ilayer] - (interface - z)
                dz_[ilayer + 2 :] = dz[ilayer + 1 :]
                sigma_ = np.zeros(N + 1, sigma.dtype)
                sigma_[: ilayer + 1] = sigma[: ilayer + 1]
                sigma_[ilayer + 1 :] = sigma[ilayer:]
                mu_ = np.zeros(N + 1, mu.dtype)
                mu_[: ilayer + 1] = mu[: ilayer + 1]
                mu_[ilayer + 1 :] = mu[ilayer:]
                rho_ = np.zeros(N + 1, rho.dtype)
                rho_[: ilayer + 1] = rho[: ilayer + 1]
                rho_[ilayer + 1 :] = rho[ilayer:]
                for i, n in enumerate(indices):
                    if n > ilayer:
                        indices[i] += 1
                ilayer += 1
                dz = dz_
                sigma = sigma_
                mu = mu_
                rho = rho_
                N += 1
                added = True
            indices += [ilayer]
            pseudo += [added]
        return tuple([dz, sigma, mu, rho] + indices + pseudo)

    @property
    def vp(self):
        """
        P-wave velocity in each layer (read-only).

        :type: :py:class:`numpy.ndarray`
        """
        return np.sqrt(self.sigma / self.rho)

    @property
    def vs(self):
        """
        S-wave velocity in each layer (read-only).

        :type: :py:class:`numpy.ndarray`
        """
        return np.sqrt(self.mu / self.rho)

    def __repr__(self):
        """
        Pretty-print model
        :return: An ascii-art rendition of the model
        :rtype: str
        """
        z = 0
        out = []
        for i in range(self.nlayers):
            out += [
                "------------------------------------------------------- z = %.2f km\n"
                % z
            ]
            if self.vs[i] == 0:
                out += [
                    "  vp =%5.2f km/s       FLUID        rho =%5.2f g/cm^3\n"
                    % (self.vp[i], self.rho[i])
                ]
            else:
                out += [
                    "  vp =%5.2f km/s   vs =%5.2f km/s   rho =%5.2f g/cm^3\n"
                    % (self.vp[i], self.vs[i], self.rho[i])
                ]
            z += self.dz[i]
        return "".join(out)

    def copy(self):
        """
        Make a copy of the current model.

        :rtype: :py:class:`LayeredStructureModel`
        """
        m = LayeredStructureModel()
        m.nlayers = self.nlayers
        m.dz = self.dz.copy()
        m.sigma = self.sigma.copy()
        m.mu = self.mu.copy()
        m.rho = self.rho.copy()
        return m


class ReceiverSet:
    """
    Base class for representing receiver locations.
    """

    def __init__(self):
        pass

    def validate(self):
        """
        Performs sanity checks on receivers
        """
        if np.any(self.rr < 0):
            raise ValueError("Some receivers appear to be at negative radii")
        if np.any(self.rr > 200):
            warnings.warn(
                "Source-receiver distances exceed 200 km. Flat-earth approximation may not be appropriate. ",
                RuntimeWarning,
            )

    def copy(self):
        raise NotImplementedError

    @property
    def nDim(self):
        raise NotImplementedError


class RegularlyDistributedReceivers(ReceiverSet):
    def __init__(
        self,
        rmin=30,
        rmax=200,
        nr=18,
        phimin=0,
        phimax=360,
        nphi=36,
        depth=0,
        x0=0,
        y0=0,
        degrees=True,
    ):
        """
        Create a set of receivers distributed regularly in cylindrical polar
        coordinates. Receivers lie on a set of equi-distant concentric circles
        centred on the origin, and at equally-distributed azimuths. This
        regularity enables faster computation and is useful if general
        characterisation of the wavefield is required.

        :param float rmin: Minimum radius to generate (inclusive; km)
        :param float rmax: Maximum radius to generate (inclusive; km)
        :param int nr: Number of radii to generate
        :param float phimin: Minimum angle to generate.
        :param float phimax: Maximum angles to generate. Angles are measured
            counter-clockwise from the x (East) axis, when viewed from above.
            They may be specified in degrees or radians (see `degrees` option).
        :param int nphi: Number of angles to consider
        :param float depth: Depth of burial of receivers (km)
        :param float x0:
        :param float y0: Coordinates of the origin point about which the
            receivers are to be distributed. These are used to verify the
            requirement that the source location coincide with the symmetry
            axis, and to generate appropriate Cartesian coordinates when
            .to_xy() is called.
        :param bool degrees: True if angles are specified in degrees; False if
            in radians.
        """
        super().__init__()
        self.rmin = rmin
        self.rmax = rmax
        self.nr = nr
        self.phimin = phimin
        self.phimax = phimax
        self.nphi = nphi
        self.depth = depth
        self.x0 = x0
        self.y0 = y0
        self.degrees = degrees
        self.rr = None
        self.pp = None

    def copy(self):
        """
        Make a copy of the the current receivers.

        :rtype: RegularlyDistributedReceivers
        """
        r = RegularlyDistributedReceivers(
            self.rmin,
            self.rmax,
            self.nr,
            self.phimin,
            self.phimax,
            self.nphi,
            self.depth,
            self.x0,
            self.y0,
            self.degrees,
        )
        return r

    def generate_rphi(self, event_x, event_y):
        """
        Construct radial and azimuthal grids.

        :param float event_x:
        :param float event_y: Coordinates of the seismic source in global
            Cartesian grid
        """
        if event_x != self.x0 or event_y != self.y0:
            raise ValueError(
                "RegularlyDistributedReceivers may only be used for events located on the central axis."
                "Either (i) Check that x0/y0 parameters of RegularlyDistributedReceivers are set appropriately, or"
                "      (ii) Specify your stations using ListOfReceivers"
            )
        if self.rmax < self.rmin:
            raise ValueError("Minimum radius appears to exceed maximum radius!")
        if self.phimax < self.phimin:
            raise ValueError("Minimum azimuth appears to exceed maximum azimuth!")
        self.rr = np.linspace(self.rmin, self.rmax, self.nr)
        self.pp = np.linspace(self.phimin, self.phimax, self.nphi)
        if self.degrees:
            self.pp = np.deg2rad(self.pp)

    def as_xy(self):
        if self.rr is None or self.pp is None:
            self.generate_rphi(self.x0, self.y0)
        return (self.x0 + np.outer(self.rr, np.cos(self.pp))), (
            self.y0 + np.outer(self.rr, np.sin(self.pp))
        )

    @property
    def nDim(self):
        n = 0
        if self.nr > 1:
            n += 1
        if self.nphi > 1:
            n += 1
        return n

    @property
    def nstations(self):
        """
        The total number of receivers present.

        :type: int
        """
        return self.nr * self.nphi

    def asListOfReceivers(self):
        """
        Convert to ListOfReceivers object.

        :rtype: ListOfReceivers
        """
        xx, yy = self.as_xy()
        return ListOfReceivers(xx.flatten(), yy.flatten(), self.depth)


class ListOfReceivers(ReceiverSet):
    def __init__(self, xx, yy, depth=0, geometry="cartesian"):
        """
        Create a set pf receivers at known locations.

        :param np.ndarray xx:
        :param np.ndarray yy: Arrays containing the x and y coordinates of each
            receiver. See `geometry` parameter for details on interpretation.
        :param float depth: The depth of burial of receivers; zero for surface
            records. Note that all receivers must be at the same depth: if
            multiple depths are required, it will be necessary to split the
            receivers into groups and perform separate simulations for each.
        :param str geometry: If `geometry='cartesian'`, x and y coordinates are
            assumed to be given relative to a Cartesian grid with origin at
            some arbitrary point determined by the user. Alternatively, if
            `geometry='spherical'`, the x and y coordinates are assumed to be
            given in degrees longitude/latitude. Note that the actual
            simulation remains in a Cartesian geometry with a flat-Earth
            approximation.
        """
        assert xx.shape[0] == yy.shape[0]
        self.xx = xx
        self.yy = yy
        self.depth = depth
        self.nr = self.xx.shape[0]
        self.pp = None
        self.rr = None
        self.geometry = geometry

    def generate_rphi(self, event_x, event_y):
        self.nr = self.xx.shape[0]
        if self.geometry == "cartesian":
            self.rr = np.sqrt((self.xx - event_x) ** 2 + (self.yy - event_y) ** 2)
            self.pp = np.arctan2(self.yy - event_y, self.xx - event_x)
        elif self.geometry == "spherical":
            self.rr = gc_dist(
                np.deg2rad(self.yy),
                np.deg2rad(self.xx),
                np.deg2rad(event_y),
                np.deg2rad(event_x),
            )
            self.pp = np.pi / 2 - gc_azimuth(
                np.deg2rad(event_y),
                np.deg2rad(event_x),
                np.deg2rad(self.yy),
                np.deg2rad(self.xx),
            )
        else:
            raise ValueError(
                "Unrecognised value for 'geometry' parameter of ListOfReceivers"
            )

    def as_xy(self):
        return self.xx.reshape(-1, 1), self.yy.reshape(-1, 1)

    @property
    def nDim(self):
        n = 0
        if self.nr > 1:
            n += 1
        return n

    def copy(self):
        """
        Make a copy of the current receivers.

        :rtype: ListOfReceivers
        """
        r = ListOfReceivers(self.xx, self.yy, self.depth, self.geometry)
        return r

    @property
    def nstations(self):
        """
        The total number of receivers present.

        :type: int
        """
        return self.nr


class DerivativeSwitches:
    def __init__(
        self,
        moment_tensor=False,
        force=False,
        r=False,
        phi=False,
        x=False,
        y=False,
        z=False,
        time=False,
        thickness=False,
        structure=None,
    ):
        """
        Object to specify the set of derivatives that are sought.

        :param bool moment_tensor: Compute derivatives wrt six independent
           components of the moment tensor.
        :param bool force: Compute derivatives wrt three components of the force
           vector.
        :param bool r: Compute derivatives wrt source-receiver distance.
        :param bool phi: Compute derivatives wrt source-receiver azimuth.
        :param bool x: Compute derivatives wrt x (east-west) location of source.
        :param bool y: Compute derivatives wrt y (north-south) location of source.
        :param bool z: Compute derivatives wrt z (vertical) location of source.
        :param bool time: Compute derivatives wrt event time.
        :param bool thickness: Compute derivatives wrt layer thicknesses
        :param LayeredStructureModel or None structure: Current model (only
           required when layer-thickness derivatives are sought).
        """
        self.moment_tensor = moment_tensor
        self.force = force
        self.r = r
        self.phi = phi
        self.x = x
        self.y = y
        self.z = z
        self.time = time
        self.thickness = thickness
        self.structure = structure

    @property
    def nderivs(self):
        n = 0
        if self.moment_tensor:
            n += 6
        if self.force:
            n += 3
        if self.r:
            n += 1
        if self.phi:
            n += 1
        if self.x:
            n += 1
        if self.y:
            n += 1
        if self.z:
            n += 1
        if self.time:
            n += 1
        if self.thickness:
            try:
                n += (
                    self.structure.nlayers - 1
                )  # We don't want derivatives wrt the infinite layer
            except AttributeError:
                raise ValueError(
                    "DerivativeSwitches object requires knowledge of earth model to enable structure derivatives. Pass `structure=<model>` at creation or set `.structure` attribute."
                )
        return n

    @property
    def i_mt(self):
        if not self.moment_tensor:
            return None
        i = 0
        return i

    @property
    def i_f(self):
        if not self.force:
            return None
        i = 0
        if self.moment_tensor:
            i += 6
        return i

    @property
    def i_r(self):
        if not self.r:
            return None
        i = 0
        if self.moment_tensor:
            i += 6
        if self.force:
            i += 3
        return i

    @property
    def i_phi(self):
        if not self.phi:
            return None
        i = 0
        if self.moment_tensor:
            i += 6
        if self.force:
            i += 3
        if self.r:
            i += 1
        return i

    @property
    def i_x(self):
        if not self.x:
            return None
        i = 0
        if self.moment_tensor:
            i += 6
        if self.force:
            i += 3
        if self.r:
            i += 1
        if self.phi:
            i += 1
        return i

    @property
    def i_y(self):
        if not self.y:
            return None
        i = 0
        if self.moment_tensor:
            i += 6
        if self.force:
            i += 3
        if self.r:
            i += 1
        if self.phi:
            i += 1
        if self.x:
            i += 1
        return i

    @property
    def i_z(self):
        if not self.z:
            return None
        i = 0
        if self.moment_tensor:
            i += 6
        if self.force:
            i += 3
        if self.r:
            i += 1
        if self.phi:
            i += 1
        if self.x:
            i += 1
        if self.y:
            i += 1
        return i

    @property
    def i_time(self):
        if not self.time:
            return None
        i = 0
        if self.moment_tensor:
            i += 6
        if self.force:
            i += 3
        if self.r:
            i += 1
        if self.phi:
            i += 1
        if self.x:
            i += 1
        if self.y:
            i += 1
        if self.z:
            i += 1
        return i

    @property
    def i_thickness(self):
        if not self.thickness:
            return None
        i = 0
        if self.moment_tensor:
            i += 6
        if self.force:
            i += 3
        if self.r:
            i += 1
        if self.phi:
            i += 1
        if self.x:
            i += 1
        if self.y:
            i += 1
        if self.z:
            i += 1
        if self.time:
            i += 1
        return i


def kIntegrationStencil(kmin, kmax, nk):
    """
    An integration stencil based on the trapezium rule.
    Inputs:
    kmin,kmax - float, upper and lower limits of integration
    nk        - int, number of evaluation points

    Returns:
    ndarray,ndarray - evaluation points and associated weights
    """
    kk = np.linspace(kmin, kmax, nk)
    wts = np.full(nk, kk[1] - kk[0])
    wts[0] *= 0.5
    wts[-1] *= 0.5
    return kk, wts


def compute_spectra(
    structure,
    source,
    stations,
    omegas,
    derivatives=None,
    show_progress=True,
    stencil=kIntegrationStencil,
    stencil_kwargs={"kmin": 0, "kmax": 2.04, "nk": 1200},
    squeeze_outputs=True,
):
    """
    Calculate and return velocity spectra for a given source and earth model at
    specified locations.

    :param LayeredStructureModel structure: The earth model within which
        calculations should be performed.
    :param PointSource source: The source(s) for which calculations should be
        performed
    :param ListOfReceivers or RegularlyDistributedReceivers stations: The
        locations for which spectra should be generated.
    :param numpy.ndarray omegas: Freqencies at which spectrum is to be
        computed. Array may be complex and should have shape (n, ).
    :param DerivativeSwitches or None derivatives: Determines which derivatives
        are computed and returned. See also discussion of return value, below.
    :param bool show_progress: Display progress bars if available.
    :param callable stencil: Function to generate quadrature points/weights for
        performing integral over wavenumber, k. Pass a callable satisfying::

            kk, wts = stencil(**stencil_kwargs)

        where kk represents sampling points and wts the associated quadrature
        weights.
    :param dict stencil_kwargs: Arguments that will be passed to stencil()
    :param bool squeeze_outputs: If true, apply :py:func:`numpy.squeeze` to all
        output arrays to eliminate dimensions of size '1'.

    The output of ``compute_spectra`` depends on the value of the ``derivatives``
    parameter.

    :returns: If ``derivatives = None`` then ``compute_spectra`` returns a single
        array, ``spectra``.Otherwise it returns a tuple of two arrays,
        ``(spectra, deriv)``.

        The shapes of ``spectra`` and ``deriv`` depend on the nature of
        the object passed to ``stations``.

        - If ``stations`` is an instance of :py:class:`~pyprop8.ListOfReceivers`,
          ``spectra`` will have shape
          ``(source.nsources, receivers.nstations, 3, nfreq)``, where
          ``source.nsources`` is the number of moment tensor/source vector pairs
          specified within the ``source`` object, ``receivers.nstations`` is the
          total number of receivers, and ``nfreq`` is the number of frequency
          points at which evaluation was requested (i.e., ``size(omegas)``). The
          third dimension indexes the three components of motion: radial,
          transverse, and vertical. ``deriv`` will have shape
          ``(source.nsources, receivers.nstations,derivatives.nderivs,3,nfreq)``
          where ``derivatives.nderivs`` is the total number of derivative
          components requested within the :py:class:`~pyprop8.DerivativeSwitches`
          object.
        - If ``stations`` is an instance of
          :py:class:`~pyprop8.RegularlyDistributedReceivers`, ``spectra`` will
          have shape ``(source.nsources, receivers.nr, receivers.nphi, 3, nfreq)``,
          where ``receivers.nr`` and ``receivers.nphi`` are the number of grid
          points in the radial and azimuthal directions, respectively.
          ``deriv`` will have shape
          ``(source.nsources, receivers.nr, receivers.nphi, derivatives.nderiv, 3, nfreq)``.

        In all cases, if `squeeze_outputs=True` (default), then
        :py:func:`numpy.squeeze` will be called, discarding dimensions with only
        one entry.


    """
    stations.generate_rphi(source.x, source.y)
    stations.validate()
    omegas = np.atleast_1d(omegas)
    nomegas = omegas.shape[0]

    do_derivatives = False
    if derivatives is not None:
        if derivatives.nderivs > 0:
            do_derivatives = True
        else:
            # The user is set up to expect derivatives so we'll give them something
            # back, but nothing is actually switched on...
            d_spectra = None

    nr = stations.nr
    rr_inv = 1 / stations.rr
    nsources = source.nsources

    k, k_wts = stencil(**stencil_kwargs)
    nk = k.shape[0]
    k_wts /= 2 * np.pi

    dz, sigma, mu, rho, isrc, irec, src_added, rec_added = structure.with_interfaces(
        source.dep, stations.depth
    )
    assert irec < isrc, "Receivers must be above source"

    # Set up Bessel function arrays
    mm = np.arange(-2, 3)
    jv = spec.jv(np.tile(mm, nr * nk), np.outer(k, stations.rr).repeat(5)).reshape(
        nk, nr, 5
    )
    jvp = spec.jvp(np.tile(mm, nr * nk), np.outer(k, stations.rr).repeat(5)).reshape(
        nk, nr, 5
    )
    if do_derivatives:
        if derivatives.moment_tensor:
            d_Mxyz = np.array(
                [
                    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
                    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                ],
                dtype="float64",
            )
        if derivatives.force:
            d_F = np.array(
                [[[1], [0], [0]], [[0], [1], [0]], [[0], [0], [1]]], dtype="float64"
            )
        if derivatives.r or derivatives.x or derivatives.y:
            djvp_dr = spec.jvp(
                np.tile(mm, nr * nk), np.outer(k, stations.rr).repeat(5), 2
            ).reshape(nk, nr, 5) * k.reshape(-1, 1, 1)
    # Allocate output data arrays
    if type(stations) is RegularlyDistributedReceivers:
        spectra = np.zeros(
            [nsources, stations.nr, stations.nphi, 3, nomegas], dtype="complex128"
        )
        if do_derivatives:
            d_spectra = np.zeros(
                [nsources, stations.nr, stations.nphi, derivatives.nderivs, 3, nomegas],
                dtype="complex128",
            )
            if derivatives.x or derivatives.y:
                d_spectra_rphi = np.zeros(
                    [nsources, stations.nr, stations.nphi, 2, 3, nomegas],
                    dtype="complex128",
                )

        ss = slice(None)
        es1 = "k,ksm,krm,mp->srp"
        es1d = "k,ksdm,krm,mp->srpd"
        es2 = "m,ksm,krm,r,k,mp->srp"
        es2d = "m,ksdm,krm,r,k,mp->srpd"
        es3 = "k,ksm,krm,m,mp->srp"
        es4d = "srpcw,rp->srpcw"
    elif type(stations) is ListOfReceivers:
        spectra = np.zeros([nsources, stations.nr, 1, 3, nomegas], dtype="complex128")
        if do_derivatives:
            d_spectra = np.zeros(
                [nsources, stations.nr, 1, derivatives.nderivs, 3, nomegas],
                dtype="complex128",
            )
            if derivatives.x or derivatives.y:
                d_spectra_rphi = np.zeros(
                    [nsources, stations.nr, 1, 2, 3, nomegas], dtype="complex128"
                )
        ss = 0
        es1 = "k,ksm,krm,mr->sr"
        es1d = "k,ksdm,krm,mr->srd"
        es2 = "m,ksm,krm,r,k,mr->sr"
        es2d = "m,ksdm,krm,r,k,mr->srd"
        es3 = "k,ksm,krm,m,mr->sr"
        es4d = "srcw,r->srcw"
    else:
        raise NotImplementedError(
            "Unrecognised receiver object, type: %s" % (type(stations))
        )

    if show_progress:
        t = tqdm(total=nomegas)

    plan_1 = False
    plan_1d = False
    plan_2 = False
    plan_2d = False
    plan_3 = False
    determine_optimal_plan = True

    eimphi = np.exp(np.outer(1j * mm, stations.pp))
    for iom, omega in enumerate(omegas):
        if do_derivatives:
            H = compute_H_matrices(
                k[k != 0],
                omega,
                dz,
                sigma,
                mu,
                rho,
                isrc,
                irec,
                do_derivatives=derivatives.thickness,
            )
            if derivatives.thickness:
                H_psv, H_sh, d_H_psv, d_H_sh = H
            else:
                H_psv, H_sh = H
        else:
            H_psv, H_sh = compute_H_matrices(
                k[k != 0], omega, dz, sigma, mu, rho, isrc, irec
            )

        b = np.zeros([nk, nsources, 6, 5], dtype="complex128")
        for i in range(nsources):
            s_psv, s_sh = sourceVector(
                source.Mxyz[i, :, :],
                source.F[i, :, 0],
                k[k != 0],
                sigma[isrc],
                mu[isrc],
            )
            b[k != 0, i, :4, :] = (H_psv @ s_psv).value
            b[k != 0, i, 4:, :] = (H_sh @ s_sh).value
        if do_derivatives:
            d_b = np.zeros(
                [nk, nsources, derivatives.nderivs, 6, 5], dtype="complex128"
            )
            if derivatives.moment_tensor:
                j0 = derivatives.i_mt
                for j in range(6):
                    # print(j)
                    s_psv, s_sh = sourceVector(
                        d_Mxyz[j, :, :], np.zeros([3]), k[k != 0], sigma[isrc], mu[isrc]
                    )
                    for i in range(nsources):
                        d_b[k != 0, i, j0 + j, :4, :] = (H_psv @ s_psv).value
                        d_b[k != 0, i, j0 + j, 4:, :] = (H_sh @ s_sh).value
            if derivatives.force:
                j0 = derivatives.i_f
                for j in range(3):
                    s_psv, s_sh = sourceVector(
                        np.zeros(3, 3), d_F[j, :, 0], k[k != 0], sigma[isrc], mu[isrc]
                    )
                    for i in range(nsources):
                        d_b[k != 0, i, j0 + j, :4, :] = (H_psv @ s_psv).value
                        d_b[k != 0, i, j0 + j, 4:, :] = (H_sh @ s_sh).value
            if derivatives.z:
                j0 = derivatives.i_z
                for i in range(nsources):
                    s_psv, s_sh = sourceVector_ddep(
                        source.Mxyz[i, :, :],
                        source.F[i, :, 0],
                        omega,
                        k[k != 0],
                        sigma[isrc],
                        mu[isrc],
                        rho[isrc],
                    )
                    d_b[k != 0, i, j0, :4, :] = (H_psv @ s_psv).value
                    d_b[k != 0, i, j0, 4:, :] = (H_sh @ s_sh).value
            if derivatives.thickness:
                if not src_added:
                    warnings.warn(
                        "Thickness derivatives may be inaccurate: source lies exactly at structural interface"
                    )
                j0 = derivatives.i_thickness
                j = 0
                for l in range(
                    dz.shape[0] - 1
                ):  # For all layers except the infinite halfspace...
                    if l == isrc - 1 and src_added:
                        continue  # ...but we're not interested in a fake layer created to represent the source depth...
                    if l == irec - 1 and rec_added:
                        continue  # ...or in a fake layer created to represent the receiver depth...
                    for i in range(nsources):
                        s_psv, s_sh = sourceVector(
                            source.Mxyz[i, :, :],
                            source.F[i, :, 0],
                            k[k != 0],
                            sigma[isrc],
                            mu[isrc],
                        )
                        d_b[k != 0, i, j0 + j, :4, :] = (d_H_psv[l] @ s_psv).value
                        d_b[k != 0, i, j0 + j, 4:, :] = (d_H_sh[l] @ s_sh).value
                        if l < isrc:
                            # Increasing thickness of layer will have side effect of pushing
                            # source infinitesimally deeper. We therefore need to compensate
                            # by making corresponding reduction in thickness of 'pseudo'-layer
                            # above the source.
                            d_b[k != 0, i, j0 + j, :4, :] -= (
                                d_H_psv[isrc - 1] @ s_psv
                            ).value
                            d_b[k != 0, i, j0 + j, 4:, :] -= (
                                d_H_sh[isrc - 1] @ s_sh
                            ).value
                            # However, now we've made the layer containing the source thinner... so
                            # we'd best increase the thickness of the layer below the source (remember, we
                            # split one 'true' layer into two to insert the source).
                            d_b[k != 0, i, j0 + j, :4, :] += (
                                d_H_psv[isrc] @ s_psv
                            ).value
                            d_b[k != 0, i, j0 + j, 4:, :] += (d_H_sh[isrc] @ s_sh).value
                            # None of this is correct if our source lies exactly at a structural
                            # interface -- as we wouldn't create a split layer -- likely rare...
                            # Hence warning message above
                            # Handling this properly would entail creating an additional, infinitesimal layer and re-doing propagation
                        if l < irec and rec_added:
                            # Move receiver 'up' to compensate for thicker layer above.
                            d_b[k != 0, i, j0 + j, :4, :] -= (
                                d_H_psv[irec - 1] @ s_psv
                            ).value
                            d_b[k != 0, i, j0 + j, 4:, :] -= (
                                d_H_sh[irec - 1] @ s_sh
                            ).value
                            # and increase the layer below
                            d_b[k != 0, i, j0 + j, :4, :] += (
                                d_H_psv[irec] @ s_psv
                            ).value
                            d_b[k != 0, i, j0 + j, 4:, :] += (d_H_sh[irec] @ s_sh).value
                            # No warning if receivers lie exactly at a structural interface: this most likely corresponds to
                            # receivers deliberately installed to coincide with interface (e.g. ocean floor).
                    j += 1

        del H_psv, H_sh, s_psv, s_sh
        if do_derivatives:
            if derivatives.thickness:
                del d_H_psv, d_H_sh
        if determine_optimal_plan and iom == 0:
            # First time through, determine optimal contraction schemes
            plan_1, _ = np.einsum_path(es1, k * k_wts, b[:, :, 1, :], jvp, eimphi)
            plan_2, _ = np.einsum_path(
                es2, 1j * mm, b[:, :, 4, :], jv, rr_inv, k_wts, eimphi
            )
            plan_3, _ = np.einsum_path(
                es3, k * k_wts, b[:, :, 1, :], jvp, 1j * mm, eimphi
            )
            if do_derivatives:
                if derivatives.moment_tensor or derivatives.force:
                    plan_1d, _ = np.einsum_path(
                        es1d, k * k_wts, d_b[:, :, 0:6, 1, :], jvp, eimphi
                    )
                    plan_2d, _ = np.einsum_path(
                        es2d, 1j * mm, d_b[:, :, 0:6, 4, :], jv, rr_inv, k_wts, eimphi
                    )
        spectra[:, :, ss, 0, iom] = np.einsum(
            es1, k * k_wts, b[:, :, 1, :], jvp, eimphi, optimize=plan_1
        ) + np.einsum(
            es2, 1j * mm, b[:, :, 4, :], jv, rr_inv, k_wts, eimphi, optimize=plan_2
        )
        spectra[:, :, ss, 1, iom] = np.einsum(
            es2, 1j * mm, b[:, :, 1, :], jv, rr_inv, k_wts, eimphi, optimize=plan_2
        ) - np.einsum(es1, k * k_wts, b[:, :, 4, :], jvp, eimphi, optimize=plan_1)
        spectra[:, :, ss, 2, iom] = np.einsum(
            es1, k * k_wts, b[:, :, 0, :], jv, eimphi, optimize=plan_1
        )
        if do_derivatives:
            if derivatives.moment_tensor:
                j0 = derivatives.i_mt
                d_spectra[:, :, ss, j0 : j0 + 6, 0, iom] = np.einsum(
                    es1d,
                    k * k_wts,
                    d_b[:, :, j0 : j0 + 6, 1, :],
                    jvp,
                    eimphi,
                    optimize=plan_1d,
                ) + np.einsum(
                    es2d,
                    1j * mm,
                    d_b[:, :, j0 : j0 + 6, 4, :],
                    jv,
                    rr_inv,
                    k_wts,
                    eimphi,
                    optimize=plan_2d,
                )
                d_spectra[:, :, ss, j0 : j0 + 6, 1, iom] = np.einsum(
                    es2d,
                    1j * mm,
                    d_b[:, :, j0 : j0 + 6, 1, :],
                    jv,
                    rr_inv,
                    k_wts,
                    eimphi,
                    optimize=plan_2d,
                ) - np.einsum(
                    es1d,
                    k * k_wts,
                    d_b[:, :, j0 : j0 + 6, 4, :],
                    jvp,
                    eimphi,
                    optimize=plan_1d,
                )
                d_spectra[:, :, ss, j0 : j0 + 6, 2, iom] = np.einsum(
                    es1d,
                    k * k_wts,
                    d_b[:, :, j0 : j0 + 6, 0, :],
                    jv,
                    eimphi,
                    optimize=plan_1d,
                )
            if derivatives.force:
                j0 = derivatives.i_f
                d_spectra[:, :, ss, j0 : j0 + 3, 0, iom] = np.einsum(
                    es1d,
                    k * k_wts,
                    d_b[:, :, j0 : j0 + 3, 1, :],
                    jvp,
                    eimphi,
                    optimize=plan_1d,
                ) + np.einsum(
                    es2d,
                    1j * mm,
                    d_b[:, :, j0 : j0 + 3, 4, :],
                    jv,
                    rr_inv,
                    k_wts,
                    eimphi,
                    optimize=plan_2d,
                )
                d_spectra[:, :, ss, j0 : j0 + 3, 1, iom] = np.einsum(
                    es2d,
                    1j * mm,
                    d_b[:, :, j0 : j0 + 3, 1, :],
                    jv,
                    rr_inv,
                    k_wts,
                    eimphi,
                    optimize=plan_2d,
                ) - np.einsum(
                    es1d,
                    k * k_wts,
                    d_b[:, :, j0 : j0 + 3, 4, :],
                    jvp,
                    eimphi,
                    optimize=plan_1d,
                )
                d_spectra[:, :, ss, j0 : j0 + 3, 2, iom] = np.einsum(
                    es1d,
                    k * k_wts,
                    d_b[:, :, j0 : j0 + 3, 0, :],
                    jv,
                    eimphi,
                    optimize=plan_1d,
                )
            if derivatives.r:
                j0 = derivatives.i_r
                d_spectra[:, :, ss, j0, 0, iom] = (
                    np.einsum(
                        es1, k * k_wts, b[:, :, 1, :], djvp_dr, eimphi, optimize=plan_1
                    )
                    - np.einsum(
                        es2,
                        1j * mm,
                        b[:, :, 4, :],
                        jv,
                        rr_inv**2,
                        k_wts,
                        eimphi,
                        optimize=plan_2,
                    )
                    + np.einsum(
                        es2,
                        1j * mm,
                        b[:, :, 4, :],
                        jvp,
                        rr_inv,
                        k * k_wts,
                        eimphi,
                        optimize=plan_2,
                    )
                )
                d_spectra[:, :, ss, j0, 1, iom] = (
                    np.einsum(
                        es2,
                        -1j * mm,
                        b[:, :, 1, :],
                        jv,
                        rr_inv**2,
                        k_wts,
                        eimphi,
                        optimize=plan_2,
                    )
                    + np.einsum(
                        es2,
                        1j * mm,
                        b[:, :, 1, :],
                        jvp,
                        rr_inv,
                        k * k_wts,
                        eimphi,
                        optimize=plan_2,
                    )
                    - np.einsum(
                        es1, k * k_wts, b[:, :, 4, :], djvp_dr, eimphi, optimize=plan_1
                    )
                )
                d_spectra[:, :, ss, j0, 2, iom] = np.einsum(
                    es1, k * k * k_wts, b[:, :, 0, :], jvp, eimphi, optimize=plan_1
                )
            if derivatives.phi:
                j0 = derivatives.i_phi
                d_spectra[:, :, ss, j0, 0, iom] = np.einsum(
                    es3, k * k_wts, b[:, :, 1, :], jvp, 1j * mm, eimphi, optimize=plan_3
                ) + np.einsum(
                    es2,
                    -mm * mm,
                    b[:, :, 4, :],
                    jv,
                    rr_inv,
                    k_wts,
                    eimphi,
                    optimize=plan_2,
                )
                d_spectra[:, :, ss, j0, 1, iom] = np.einsum(
                    es2,
                    -mm * mm,
                    b[:, :, 1, :],
                    jv,
                    rr_inv,
                    k_wts,
                    eimphi,
                    optimize=plan_2,
                ) - np.einsum(
                    es3, k * k_wts, b[:, :, 4, :], jvp, 1j * mm, eimphi, optimize=plan_3
                )
                d_spectra[:, :, ss, j0, 2, iom] = np.einsum(
                    es3, k * k_wts, b[:, :, 0, :], jv, 1j * mm, eimphi, optimize=plan_3
                )
            if derivatives.x or derivatives.y:
                # We need to get the r and phi derivatives one way or another...
                if derivatives.r:
                    d_spectra_rphi[:, :, ss, 0, :, iom] = d_spectra[
                        :, :, ss, derivatives.i_r, :, iom
                    ]
                else:
                    d_spectra_rphi[:, :, ss, 0, 0, iom] = (
                        np.einsum(
                            es1,
                            k * k_wts,
                            b[:, :, 1, :],
                            djvp_dr,
                            eimphi,
                            optimize=plan_1,
                        )
                        - np.einsum(
                            es2,
                            1j * mm,
                            b[:, :, 4, :],
                            jv,
                            rr_inv**2,
                            k_wts,
                            eimphi,
                            optimize=plan_2,
                        )
                        + np.einsum(
                            es2,
                            1j * mm,
                            b[:, :, 4, :],
                            jvp,
                            rr_inv,
                            k * k_wts,
                            eimphi,
                            optimize=plan_2,
                        )
                    )
                    d_spectra_rphi[:, :, ss, 0, 1, iom] = (
                        np.einsum(
                            es2,
                            -1j * mm,
                            b[:, :, 1, :],
                            jv,
                            rr_inv**2,
                            k_wts,
                            eimphi,
                            optimize=plan_2,
                        )
                        + np.einsum(
                            es2,
                            1j * mm,
                            b[:, :, 1, :],
                            jvp,
                            rr_inv,
                            k * k_wts,
                            eimphi,
                            optimize=plan_2,
                        )
                        - np.einsum(
                            es1,
                            k * k_wts,
                            b[:, :, 4, :],
                            djvp_dr,
                            eimphi,
                            optimize=plan_1,
                        )
                    )
                    d_spectra_rphi[:, :, ss, 0, 2, iom] = np.einsum(
                        es1, k * k * k_wts, b[:, :, 0, :], jvp, eimphi, optimize=plan_1
                    )
                if derivatives.phi:
                    d_spectra_rphi[:, :, ss, 1, :, iom] = d_spectra[
                        :, :, ss, derivatives.i_phi, :, iom
                    ]
                else:
                    d_spectra_rphi[:, :, ss, 1, 0, iom] = np.einsum(
                        es3,
                        k * k_wts,
                        b[:, :, 1, :],
                        jvp,
                        1j * mm,
                        eimphi,
                        optimize=plan_3,
                    ) + np.einsum(
                        es2,
                        -mm * mm,
                        b[:, :, 4, :],
                        jv,
                        rr_inv,
                        k_wts,
                        eimphi,
                        optimize=plan_2,
                    )
                    d_spectra_rphi[:, :, ss, 1, 1, iom] = np.einsum(
                        es2,
                        -mm * mm,
                        b[:, :, 1, :],
                        jv,
                        rr_inv,
                        k_wts,
                        eimphi,
                        optimize=plan_2,
                    ) - np.einsum(
                        es3,
                        k * k_wts,
                        b[:, :, 4, :],
                        jvp,
                        1j * mm,
                        eimphi,
                        optimize=plan_3,
                    )
                    d_spectra_rphi[:, :, ss, 1, 2, iom] = np.einsum(
                        es3,
                        k * k_wts,
                        b[:, :, 0, :],
                        jv,
                        1j * mm,
                        eimphi,
                        optimize=plan_3,
                    )
            if derivatives.z:
                j0 = derivatives.i_z
                d_spectra[:, :, ss, j0, 0, iom] = np.einsum(
                    es1, k * k_wts, d_b[:, :, j0, 1, :], jvp, eimphi, optimize=plan_1
                ) + np.einsum(
                    es2,
                    1j * mm,
                    d_b[:, :, j0, 4, :],
                    jv,
                    rr_inv,
                    k_wts,
                    eimphi,
                    optimize=plan_2,
                )
                d_spectra[:, :, ss, j0, 1, iom] = np.einsum(
                    es2,
                    1j * mm,
                    d_b[:, :, j0, 1, :],
                    jv,
                    rr_inv,
                    k_wts,
                    eimphi,
                    optimize=plan_2,
                ) - np.einsum(
                    es1, k * k_wts, d_b[:, :, j0, 4, :], jvp, eimphi, optimize=plan_1
                )
                d_spectra[:, :, ss, j0, 2, iom] = np.einsum(
                    es1, k * k_wts, d_b[:, :, j0, 0, :], jv, eimphi, optimize=plan_1
                )
            if derivatives.time:
                j0 = derivatives.i_time
                d_spectra[:, :, ss, j0, :, iom] = (
                    -1j * omega * spectra[:, :, ss, :, iom]
                )
            if derivatives.thickness:
                j0 = derivatives.i_thickness
                for j in range(structure.nlayers - 1):
                    d_spectra[:, :, ss, j0 + j, 0, iom] = np.einsum(
                        es1,
                        k * k_wts,
                        d_b[:, :, j0 + j, 1, :],
                        jvp,
                        eimphi,
                        optimize=plan_1,
                    ) + np.einsum(
                        es2,
                        1j * mm,
                        d_b[:, :, j0 + j, 4, :],
                        jv,
                        rr_inv,
                        k_wts,
                        eimphi,
                        optimize=plan_2,
                    )
                    d_spectra[:, :, ss, j0 + j, 1, iom] = np.einsum(
                        es2,
                        1j * mm,
                        d_b[:, :, j0 + j, 1, :],
                        jv,
                        rr_inv,
                        k_wts,
                        eimphi,
                        optimize=plan_2,
                    ) - np.einsum(
                        es1,
                        k * k_wts,
                        d_b[:, :, j0 + j, 4, :],
                        jvp,
                        eimphi,
                        optimize=plan_1,
                    )
                    d_spectra[:, :, ss, j0 + j, 2, iom] = np.einsum(
                        es1,
                        k * k_wts,
                        d_b[:, :, j0 + j, 0, :],
                        jv,
                        eimphi,
                        optimize=plan_1,
                    )
        if show_progress:
            t.update(1)
    if show_progress:
        t.close()
    if derivatives is None:
        # Test is NOT on do_derivatives as we want to return two objects
        # whenever a DerivativeSwitches object has been provided -- regardless
        # of whether this actually has anything switched on. If not, d_spectra
        # will be None (see above).
        if squeeze_outputs:
            return spectra[:, :, ss, :, :].squeeze()
        else:
            return spectra[:, :, ss, :, :]
    else:
        if derivatives.x or derivatives.y:
            # xx,yy = stations.as_xy()
            if derivatives.x:
                if type(stations) is RegularlyDistributedReceivers:
                    drdx = np.tile(-np.cos(stations.pp), stations.nr).reshape(
                        stations.nr, stations.nphi
                    )  # Result will be array (nr x nphi)
                    dpdx = np.outer(1 / stations.rr, np.sin(stations.pp))
                elif type(stations) is ListOfReceivers:
                    drdx = -np.cos(stations.pp)
                    dpdx = np.sin(stations.pp) / stations.rr
                else:
                    raise NotImplementedError
                d_spectra[:, :, ss, derivatives.i_x, :, :] = np.einsum(
                    es4d, d_spectra_rphi[:, :, ss, 0, :, :], drdx
                ) + np.einsum(es4d, d_spectra_rphi[:, :, ss, 1, :, :], dpdx)
            if derivatives.y:
                if type(stations) is RegularlyDistributedReceivers:
                    drdy = np.tile(-np.sin(stations.pp), stations.nr).reshape(
                        stations.nr, stations.nphi
                    )
                    dpdy = np.outer(-1 / stations.rr, np.cos(stations.pp))
                elif type(stations) is ListOfReceivers:
                    drdy = -np.sin(stations.pp)
                    dpdy = -np.cos(stations.pp) / stations.rr
                else:
                    raise NotImplementedError
                d_spectra[:, :, ss, derivatives.i_y, :, :] = np.einsum(
                    es4d, d_spectra_rphi[:, :, ss, 0, :, :], drdy
                ) + np.einsum(es4d, d_spectra_rphi[:, :, ss, 1, :, :], dpdy)
            if type(stations) is ListOfReceivers:
                if stations.geometry == "spherical":
                    d_spectra[:, :, ss, derivatives.i_x, :, :] *= (
                        2 * np.pi * PLANETARY_RADIUS * np.cos(np.deg2rad(source.y))
                    ) / 360
                    d_spectra[:, :, ss, derivatives.i_y, :, :] *= (
                        2 * np.pi * PLANETARY_RADIUS
                    ) / 360

        if squeeze_outputs:
            return (
                spectra[:, :, ss, :, :].squeeze(),
                d_spectra[:, :, ss, :, :, :].squeeze(),
            )
        else:
            return spectra[:, :, ss, :, :], d_spectra[:, :, ss, :, :, :]


def compute_seismograms(
    structure,
    source,
    stations,
    nt,
    dt,
    alpha=None,
    source_time_function=None,
    pad_frac=0.5,
    xyz=True,
    derivatives=None,
    show_progress=True,
    squeeze_outputs=True,
    number_of_processes=1,
    **kwargs
):
    """
    Calculate and return displacement seismograms for a given source and earth
    model at specified locations.

    :param LayeredStructureModel structure: The earth model within which
        calculations should be performed.
    :param PointSource source: The source(s) for which calculations should be
        performed
    :param ListOfReceivers or RegularlyDistributedReceivers stations: The
        locations for which seismograms should be generated.
    :param int nt: Number of time-series points to be returned.
    :param float dt: Desired interval (seconds) between successive time series
        points (i.e. the inverse of sampling frequency). Total time series
        length is thus ``(nt-1)*dt`` seconds.
    :param float or None alpha: Shift applied to integration contour to avoid
        singularities on real axis. If ``None``, this is determined according
        to the rule-of-thumb given in O'Toole & Woodhouse (2011).
    :param callable or None source_time_function: Function representing the
        spectrum of a source time-function to be applied to the seismograms.
        Function should take a single argument, representing angular frequency,
        and return a single complex number representing the amplitude of the
        spectrum at that point. If ``None``, no source time-function is applied.
    :param float pad_frac: Simulations are performed for a longer time series,
        which is then truncated to the desired length. This helps improve
        accuracy and stability. ``pad_frac`` determines the additional padding
        used, specified as a proportion of the desired time series length.
    :param bool xyz: If ``True``, the three components of each seismogram will
        correspond to motion relative to Cartesian x/y/z axes. If ``Fales``,
        the three components will be expressed in  polar coordinates:
        radial/transverse/vertical.
    :param DerivativeSwitches or None derivatives: Determines which derivatives
        are computed and returned. See also discussion of return value, below.
    :param bool show_progress: Display progress bars if available.
    :param bool squeeze_outputs: If true, apply :py:func:`numpy.squeeze` to all
        output arrays to eliminate dimensions of size '1'.
    :param int number_of_processes: The number of processes to use while
        performing computations. If >1, the `multiprocessing` module will be
        used, splitting up the frequency band across processs. Due to the cost
        of creating and managing separate processes, more is not always faster.
    :param bool \**kwargs: Any additional keyword options will be passed to
        :py:func:`~pyprop8.compute_spectra`.

    The output of ``compute_seismograms`` depends on value of the ``derivatives``
    parameter.

    :returns: If ``derivatives = None`` then ``compute_spectra`` returns a tuple,
        ``(tt, seis)``. Otherwise ``compute_spectra`` returns a tuple,
        ``(tt, seis, deriv)``.

        Here, ``tt`` is a :py:class:`numpy.ndarray` containing the sequence of time
        points for which the seismograms have been evaluated. It will have shape
        ``(nt,)``.

        The shapes of ``seis`` and ``deriv`` depend on the nature of the object
        passed to ``stations``:

        - If ``stations`` is an instance of :py:class:`~pyprop8.ListOfReceivers`,
          ``seis`` will have shape
          ``(source.nsources, receivers.nstations, 3, nt)``, where
          ``source.nsources`` is the number of moment tensor/source vector pairs
          specified within the ``source`` object, ``receivers.nstations`` is the
          total number of receivers, and ``nt`` is the number of time points at
          which evaluation was requested. The third dimension indexes the three
          components of motion, governed by the ``xyz`` option. ``deriv`` will
          have shape
          ``(source.nsources, receivers.nstations,derivatives.nderivs,3,nt)``
          where ``derivatives.nderivs`` is the total number of derivative
          components requested within the :py:class:`~pyprop8.DerivativeSwitches`
          object.
        - If ``stations`` is an instance of
          :py:class:`~pyprop8.RegularlyDistributedReceivers`, ``seis`` will
          have shape ``(source.nsources, receivers.nr, receivers.nphi, 3, nt)``,
          where ``receivers.nr`` and ``receivers.nphi`` are the number of grid
          points in the radial and azimuthal directions, respectively.
          ``deriv`` will have shape
          ``(source.nsources, receivers.nr, receivers.nphi, derivatives.nderiv, 3, nt)``.

        In all cases, if ``squeeze_outputs=True`` (default), then
        :py:func:`numpy.squeeze` will be called, discarding dimensions with only
        one entry.
    """
    npad = int(pad_frac * nt)
    tt = np.arange(nt + npad) * dt
    if alpha is None:
        # Use 'rule of thumb' given in O'Toole & Woodhouse (2011)
        alpha = np.log(10) / tt[-1]
    ww = 2 * np.pi * np.fft.rfftfreq(nt + npad, dt)
    delta_omega = ww[1]
    ww = ww - alpha * 1j
    if derivatives is None:
        do_derivatives = False
    else:
        if derivatives.nderivs == 0:
            # Nothing actually turned on...
            do_derivatives = False
        else:
            do_derivatives = True
    if number_of_processes==1:
        spectra = compute_spectra(
            structure,
            source,
            stations,
            ww,
            derivatives,
            show_progress,
            squeeze_outputs=False,
            **kwargs
        )
        if derivatives is not None:
            # Test on derivatives, not do_derivatives, as will return d_spectra as None if derivatives provided but all off
            spectra, d_spectra = spectra
    else:
        if not HAS_MULTIPROCESSING:
            raise ModuleNotFoundError("Setting `number_of_processes > 1` requires the `multiprocessing` module, which appears not to be available.")
        # The following is done inside compute_spectra -- need to do it here as 'side effects' don't get copied
        # back to the main process.
        stations.generate_rphi(source.x,source.y)
        stations.validate()
        with Pool(number_of_processes) as pool:
            # Give each thread an evenly-distributed set of frequencies as costs may be frequency-dependent
            qs = [pool.apply_async(compute_spectra,(structure, source, stations, ww[ipool::number_of_processes], derivatives),
                                    {'show_progress':False,'squeeze_outputs':False}|kwargs) for ipool in range(number_of_processes)]
            # Accumulate results
            for ipool,q in enumerate(qs):
                spec_part = q.get()
                if derivatives is not None:
                    spec_part, d_spec_part = spec_part
                if ipool==0:
                    # Allocate arrays
                    shape = list(spec_part.shape)
                    shape[-1] = ww.shape[0]
                    spectra = np.zeros(shape,dtype=spec_part.dtype)
                    if derivatives is not None:
                        shape = list(d_spec_part.shape)
                        shape[-1] = ww.shape[0]
                        d_spectra = np.zeros(shape,dtype=d_spec_part.dtype)
                spectra[...,ipool::number_of_processes] = spec_part
                if derivatives is not None:
                    d_spectra[...,ipool::number_of_processes] = d_spec_part
        pool.join()

    if type(stations) is RegularlyDistributedReceivers:
        ess = "srpcw,w->srpcw"
        essd = "srpdcw,w->srpdcw"
        est = "ut,srpct,t->srpcu"
        estd = "ut,srpdct,t->srpdcu"
    elif type(stations) is ListOfReceivers:
        ess = "srcw,w->srcw"
        essd = "srdcw,w->srdcw"
        est = "ut,srct,t->srcu"
        estd = "ut,srdct,t->srdcu"
    else:
        raise ValueError("Unrecognised receiver object, type: %s" % (type(stations)))
    spec_shape_n = len(spectra.shape)
    ####
    if source_time_function is not None:
        stf = np.zeros(ww.shape[0], dtype="complex128")
        for i, w in enumerate(ww):
            stf[i] = source_time_function(w)
        spectra = np.einsum(ess, spectra, stf)
        if do_derivatives:
            d_spectra = np.einsum(essd, d_spectra, stf)
    if source.time != 0:  # Time shift
        tshift = np.zeros(ww.shape[0], dtype="complex128")
        for i, w in enumerate(ww):
            tshift[i] = np.exp(-1j * w * source.time)
        spectra = np.einsum(ess, spectra, tshift)
        if do_derivatives:
            d_spectra = np.einsum(essd, d_spectra, tshift)
    # if kind == 'displacement':
    # Fourier integration -- transform without 1/(i w) and then integrate
    stencil = np.tril(np.full([nt, nt + npad], dt, dtype="float64"))
    stencil[np.arange(nt), np.arange(nt)] *= 0.5
    stencil[:, 0] *= 0.5
    stencil[0, 0] = 0
    seis = (nt + npad) * delta_omega * np.fft.irfft(spectra, nt + npad) / (2 * np.pi)
    seis = np.einsum(est, stencil, seis, np.exp(alpha * tt))
    if do_derivatives:
        deriv = (
            (nt + npad) * delta_omega * np.fft.irfft(d_spectra, nt + npad) / (2 * np.pi)
        )
        deriv = np.einsum(estd, stencil, deriv, np.exp(alpha * tt))
    # This doesn't seem to be very stable. I wonder if the better way to get
    # velocity is to force the user to do it themselves -- get a time series and then
    # differentiate as required.
    # elif kind == 'velocity':
    #     stencil = np.zeros([nt,nt+npad],dtype='float64')
    #     stencil[np.arange(nt),np.arange(nt)] = 1.
    #     seis = (nt+npad)*delta_omega*np.fft.irfft(spectra)/(2*np.pi)
    #     seis = np.einsum('ut,srpct,t->srpcu',stencil,seis,np.exp(alpha*tt))
    #     if do_derivatives:
    #         deriv = (nt+npad)*delta_omega*np.fft.irfft(d_spectra)/(2*np.pi)
    #         deriv = np.einsum('ut,srpdct,t->srpdcu',stencil,deriv,np.exp(alpha*tt))
    # elif kind == 'acceleration':
    #       this would be as velocity but scaled by another factor of (i w)
    # else:
    #     raise NotImplementedError(kind)
    if xyz:
        # Rotate from (radial/transverse/z to xyz (enz))
        if type(stations) is RegularlyDistributedReceivers:
            rotator = np.zeros([stations.nr, stations.nphi, 3, 3])
            phi = np.tile(stations.pp, stations.nr).reshape(stations.nr, stations.nphi)
            rotator[:, :, 0, 0] = np.cos(phi)
            rotator[:, :, 0, 1] = -np.sin(phi)
            rotator[:, :, 1, 0] = np.sin(phi)
            rotator[:, :, 1, 1] = np.cos(phi)
            rotator[:, :, 2, 2] = 1
            esr = "rpic,srpct->srpit"
            esrd = "rpic,srpdct->srpdit"
        elif type(stations) is ListOfReceivers:
            rotator = np.zeros([stations.nstations, 3, 3])
            rotator[:, 0, 0] = np.cos(stations.pp)
            rotator[:, 0, 1] = -np.sin(stations.pp)
            rotator[:, 1, 0] = np.sin(stations.pp)
            rotator[:, 1, 1] = np.cos(stations.pp)
            rotator[:, 2, 2] = 1
            esr = "ric,srct->srit"
            esrd = "ric,srdct->srdit"
        else:
            raise ValueError(
                "Unrecognised receiver object, type: %s" % (type(stations))
            )

        if do_derivatives:
            # s_xyz = R.s_rtf
            # D[s_xyz] = D[R].s_rtf+R.D[s_rtf]
            # R = [ cos(f) -sin(f) ]
            #     [ sin(f)  cos(f) ]
            # dR/dq = [ -sin(f) df/dq -cos(f) df/dq ]
            #         [  cos(f) df/dq -sin(f) df/dq ]
            deriv = np.einsum(esrd, rotator, deriv)
            if derivatives.x:
                if type(stations) is ListOfReceivers:
                    dpdx = np.sin(stations.pp) / stations.rr
                    deriv[:, :, derivatives.i_x, 0, :] += np.einsum(
                        "srt,r->srt", seis[:, :, 0, :], -dpdx * np.sin(stations.pp)
                    ) + np.einsum(
                        "srt,r->srt", seis[:, :, 1, :], -dpdx * np.cos(stations.pp)
                    )
                    deriv[:, :, derivatives.i_x, 1, :] += np.einsum(
                        "srt,r->srt", seis[:, :, 0, :], dpdx * np.cos(stations.pp)
                    ) + np.einsum(
                        "srt,r->srt", seis[:, :, 1, :], -dpdx * np.sin(stations.pp)
                    )
                else:
                    raise NotImplementedError(
                        "Lateral derivatives not available with RegularlyDistributedReceivers, as\n"
                        "source is required to lie on central axis. Consider using ListOfReceivers."
                    )
            if derivatives.y:
                if type(stations) is ListOfReceivers:
                    dpdy = -np.cos(stations.pp) / stations.rr
                    deriv[:, :, derivatives.i_y, 0, :] += np.einsum(
                        "srt,r->srt", seis[:, :, 0, :], -dpdy * np.sin(stations.pp)
                    ) + np.einsum(
                        "srt,r->srt", seis[:, :, 1, :], -dpdy * np.cos(stations.pp)
                    )
                    deriv[:, :, derivatives.i_y, 1, :] += np.einsum(
                        "srt,r->srt", seis[:, :, 0, :], dpdy * np.cos(stations.pp)
                    ) + np.einsum(
                        "srt,r->srt", seis[:, :, 1, :], -dpdy * np.sin(stations.pp)
                    )
                else:
                    raise NotImplementedError(
                        "Lateral derivatives not available with RegularlyDistributedReceivers, as\n"
                        "source is required to lie on central axis. Consider using ListOfReceivers."
                    )
            if derivatives.phi:
                if type(stations) is ListOfReceivers:
                    deriv[:, :, derivatives.i_phi, 0, :] += np.einsum(
                        "srt,r->srt", seis[:, :, 0, :], -np.sin(stations.pp)
                    ) + np.einsum("srt,r->srt", seis[:, :, 1, :], -np.cos(stations.pp))
                    deriv[:, :, derivatives.i_phi, 1, :] += np.einsum(
                        "srt,r->srt", seis[:, :, 0, :], np.cos(stations.pp)
                    ) + np.einsum("srt,r->srt", seis[:, :, 1, :], -np.sin(stations.pp))
                elif type(stations) is RegularlyDistributedReceivers:
                    deriv[:, :, :, derivatives.i_phi, 0, :] += np.einsum(
                        "srpt,p->srpt", seis[:, :, :, 0, :], -np.sin(stations.pp)
                    ) + np.einsum(
                        "srpt,p->srpt", seis[:, :, :, 1, :], -np.cos(stations.pp)
                    )
                    deriv[:, :, :, derivatives.i_phi, 1, :] += np.einsum(
                        "srpt,p->srpt", seis[:, :, :, 0, :], np.cos(stations.pp)
                    ) + np.einsum(
                        "srpt,p->srpt", seis[:, :, :, 1, :], -np.sin(stations.pp)
                    )
                else:
                    raise ValueError(
                        "Unrecognised receiver object, type: %s" % (type(stations))
                    )

        # Do this rotation *after* derivatives as we need the unrotated seismograms for x/y deriv
        seis = np.einsum(esr, rotator, seis)
    if squeeze_outputs:
        seis = seis.squeeze()
        if do_derivatives:
            deriv = deriv.squeeze()
    if derivatives is None:
        return tt[:nt], seis
    else:
        return tt[:nt], seis, deriv


def compute_static(
    structure,
    source,
    stations,
    los_vector=np.eye(3),
    derivatives=None,
    squeeze_outputs=True,
    **kwargs
):
    """
    Calculate and return static offset measurements for a given source and earth
    model at specified locations.

    :param LayeredStructureModel structure: The earth model within which
        calculations should be performed.
    :param PointSource source: The source(s) for which calculations should be
        performed
    :param ListOfReceivers or RegularlyDistributedReceivers stations: The
        locations for which seismograms should be generated.
    :param numpy.ndarray los_vector: Vector(s) defining 'line(s) of sight'
        along which static displacement should be measured. Should be expressed
        relative to a Cartesian basis and have shape (3) or (3, nlos). Default
        will return displacements relative to Cartesian basis.
    :param DerivativeSwitches or None derivatives: Determines which derivatives
        are computed and returned. See also discussion of return value, below.
    :param bool show_progress: Display progress bars if available.
    :param bool squeeze_outputs: If true, apply :py:func:`numpy.squeeze` to all
        output arrays to eliminate dimensions of size '1'.
    :param bool \**kwargs: Any additional keyword options will be passed to
        :py:func:`~pyprop8.compute_spectra`.

    The output of ``compute_static`` depends on value of the ``derivatives``
    parameter.

    :returns: If ``derivatives = None`` then ``compute_spectra`` returns a
        single array, ``static``. Otherwise ``compute_spectra`` returns a tuple,
        ``(static, deriv)``.

        The shapes of ``static`` and ``deriv`` depend on the nature of the object
        passed to ``stations``:

        - If ``stations`` is an instance of :py:class:`~pyprop8.ListOfReceivers`,
          ``static`` will have shape
          ``(source.nsources, receivers.nstations, nlos)``, where
          ``source.nsources`` is the number of moment tensor/source vector pairs
          specified within the ``source`` object, ``receivers.nstations`` is the
          total number of receivers, and ``nlos`` is the number of lines-of-sight
          for which evaluation was requested. ``deriv`` will have shape
          ``(source.nsources, receivers.nstations,derivatives.nderivs,nlos)``
          where ``derivatives.nderivs`` is the total number of derivative
          components requested within the :py:class:`~pyprop8.DerivativeSwitches`
          object.
        - If ``stations`` is an instance of
          :py:class:`~pyprop8.RegularlyDistributedReceivers`, ``seis`` will
          have shape ``(source.nsources, receivers.nr, receivers.nphi, nlos)``,
          where ``receivers.nr`` and ``receivers.nphi`` are the number of grid
          points in the radial and azimuthal directions, respectively.
          ``deriv`` will have shape
          ``(source.nsources, receivers.nr, receivers.nphi, derivatives.nderiv, nlos)``.

        In all cases, if ``squeeze_outputs=True`` (default), then
        :py:func:`numpy.squeeze` will be called, discarding dimensions with only
        one entry.
    """
    if derivatives is None:
        do_derivatives = False
    else:
        if derivatives.nderivs == 0:
            # Nothing actually turned on...
            do_derivatives = False
        else:
            do_derivatives = True
    spectra = compute_spectra(
        structure,
        source,
        stations,
        np.array([0.0], dtype="complex128"),
        derivatives=derivatives,
        show_progress=False,
        squeeze_outputs=False,
        **kwargs
    )
    if do_derivatives:
        spectra, d_spectra = spectra

    # Output will have some numerical noise in complex domain, but this should not be significant...
    assert (
        np.abs(np.imag(spectra)) / np.abs(spectra)
    ).max() < 1e-5, "Static field should be effectively real"
    spectra = np.real(spectra)
    if do_derivatives:
        d_spectra = np.real(d_spectra)
    if los_vector is not None:
        nlos = len(los_vector.shape)
        if nlos == 1:
            eslos = ("i", "")
        elif nlos == 2:
            eslos = ("ij", "j")
        else:
            raise ValueError("los_vector should be one- or two-dimensional")
        if type(stations) is ListOfReceivers:
            es = "ric,srcw,%s->sr%sw" % eslos
            esd = "ric,srdcw,%s->srd%sw" % eslos
            rotator = np.zeros([stations.nstations, 3, 3])
            rotator[:, 0, 0] = np.cos(stations.pp)
            rotator[:, 0, 1] = -np.sin(stations.pp)
            rotator[:, 1, 0] = np.sin(stations.pp)
            rotator[:, 1, 1] = np.cos(stations.pp)
            rotator[:, 2, 2] = 1
        elif type(stations) is RegularlyDistributedReceivers:
            es = "rpic,srpcw,%s->srp%sw" % eslos
            esd = "rpic,srpdcw,%s->srpd%sw" % eslos
            rotator = np.zeros([stations.nr, stations.nphi, 3, 3])
            phi = np.tile(stations.pp, stations.nr).reshape(stations.nr, stations.nphi)
            rotator[:, :, 0, 0] = np.cos(phi)
            rotator[:, :, 0, 1] = -np.sin(phi)
            rotator[:, :, 1, 0] = np.sin(phi)
            rotator[:, :, 1, 1] = np.cos(phi)
            rotator[:, :, 2, 2] = 1
        else:
            raise ValueError(
                "Unrecognised receiver object, type: %s" % (type(stations))
            )
        los_vector = los_vector / np.linalg.norm(los_vector, axis=0)
        # print(es,rotator.shape,spectra.shape,los_vector.shape)

        if do_derivatives:
            # s_xyz = l.T.R.s_rtf
            # D[s_xyz] = l.T.R.D[s_rtf] + l.T.D[R].s_rtf
            # R = [ cos(f) -sin(f) ]
            #     [ sin(f)  cos(f) ]
            # dR/dq = [ -sin(f) df/dq -cos(f) df/dq ]
            #         [  cos(f) df/dq -sin(f) df/dq ]
            d_spectra = np.einsum(esd, rotator, d_spectra, los_vector)

            if type(stations) is ListOfReceivers:
                if derivatives.x:
                    dRdx = np.zeros([stations.nstations, 3, 3])
                    dRdx[:, 0, 0] = -np.sin(stations.pp) ** 2 / stations.rr
                    dRdx[:, 0, 1] = (
                        -np.cos(stations.pp) * np.sin(stations.pp) / stations.rr
                    )
                    dRdx[:, 1, 0] = (
                        np.cos(stations.pp) * np.sin(stations.pp) / stations.rr
                    )
                    dRdx[:, 1, 1] = -np.sin(stations.pp) ** 2 / stations.rr
                    d_spectra[:, :, derivatives.i_x, :] += np.einsum(
                        "ric,srcw,%s->sr%sw" % eslos, dRdx, spectra, los_vector
                    )
                if derivatives.y:
                    dRdx = np.zeros([stations.nstations, 3, 3])
                    dRdx[:, 0, 0] = (
                        np.sin(stations.pp) * np.cos(stations.pp) / stations.rr
                    )
                    dRdx[:, 0, 1] = np.cos(stations.pp) ** 2 / stations.rr
                    dRdx[:, 1, 0] = -np.cos(stations.pp) ** 2 / stations.rr
                    dRdx[:, 1, 1] = (
                        np.sin(stations.pp) * np.cos(stations.pp) / stations.rr
                    )
                    d_spectra[:, :, derivatives.i_y, :] += np.einsum(
                        "ric,srcw,%s->sr%sw" % eslos, dRdx, spectra, los_vector
                    )
                if derivatives.phi:
                    dRdx = np.zeros([stations.nstations, 3, 3])
                    dRdx[:, 0, 0] = -np.sin(stations.pp)
                    dRdx[:, 0, 1] = -np.cos(stations.pp)
                    dRdx[:, 1, 0] = np.cos(stations.pp)
                    dRdx[:, 1, 1] = -np.sin(stations.pp)
                    d_spectra[:, :, derivatives.i_phi, :] += np.einsum(
                        "ric,srcw,%s->sr%sw" % eslos, dRdx, spectra, los_vector
                    )
            elif type(stations) is RegularlyDistributedReceivers:
                phi = np.tile(stations.pp, stations.nr).reshape(
                    stations.nr, stations.nphi
                )
                rr = (
                    np.tile(stations.rr, stations.nphi)
                    .reshape(stations.nphi, stations.nr)
                    .T
                )
                if derivatives.x:
                    dRdx = np.zeros([stations.nr, stations.nphi, 3, 3])
                    dRdx[:, :, 0, 0] = -np.sin(phi) ** 2 / rr
                    dRdx[:, :, 0, 1] = -np.cos(phi) * np.sin(phi) / rr
                    dRdx[:, :, 1, 0] = np.cos(phi) * np.sin(phi) / rr
                    dRdx[:, :, 1, 1] = -np.sin(phi) ** 2 / rr
                    d_spectra[:, :, derivatives.i_x, :] += np.einsum(
                        "rpic,srpcw,%s->srp%sw" % eslos, dRdx, spectra, los_vector
                    )
                if derivatives.y:
                    dRdx = np.zeros([stations.nr, stations.nphi, 3, 3])
                    dRdx[:, :, 0, 0] = np.sin(phi) * np.cos(phi) / rr
                    dRdx[:, :, 0, 1] = np.cos(phi) ** 2 / rr
                    dRdx[:, :, 1, 0] = -np.cos(phi) ** 2 / rr
                    dRdx[:, :, 1, 1] = np.sin(phi) * np.cos(phi) / rr
                    d_spectra[:, :, derivatives.i_y, :] += np.einsum(
                        "rpic,srpcw,%s->srp%sw" % eslos, dRdx, spectra, los_vector
                    )
                if derivatives.phi:
                    dRdx = np.zeros([stations.nr, stations.nphi, 3, 3])
                    dRdx[:, :, 0, 0] = -np.sin(phi)
                    dRdx[:, :, 0, 1] = -np.cos(phi)
                    dRdx[:, :, 1, 0] = np.cos(phi)
                    dRdx[:, :, 1, 1] = -np.sin(phi)
                    d_spectra[:, :, derivatives.i_phi, :] += np.einsum(
                        "rpic,srpcw,%s->srp%sw" % eslos, dRdx, spectra, los_vector
                    )
            else:
                raise ValueError(
                    "Unrecognised receiver object, type: %s" % (type(stations))
                )
        spectra = np.einsum(es, rotator, spectra, los_vector)
    spectra = spectra.reshape(spectra.shape[:-1])
    if do_derivatives:
        d_spectra = d_spectra.reshape(d_spectra.shape[:-1])
    if squeeze_outputs:
        spectra = spectra.squeeze()
        if do_derivatives:
            d_spectra = d_spectra.squeeze()
    if derivatives is None:
        return spectra
    else:
        return spectra, d_spectra
