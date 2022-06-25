import numpy as np
from pyprop8 import _scaledmatrix as scm

##################################
### Boundary condition vectors ###
##################################
def freeSurfaceBoundary(nk, sh=False):
    # (2011) Eq. 85 (with P-SV system in minor vector form)
    if sh:
        m = np.zeros([nk, 2, 1], dtype="complex128")
    else:
        m = np.zeros([nk, 6, 1], dtype="complex128")
    m[:, 0, 0] = 1
    return scm.ScaledMatrixStack(m)


def oceanFloorBoundary(depth, omega, k, sigma, rho, sh=False):
    # (2011) Eqs. 86 & 88 (P-SV in minor vector form)
    nk = k.shape[0]
    if sh:
        m = np.zeros([nk, 2, 1], dtype="complex128")
        m[:, 0, 0] = 1
    else:
        zsig = np.lib.scimath.sqrt(k**2 - rho * omega**2 / sigma)
        t = np.exp(-2 * depth * zsig)
        m = np.zeros([nk, 6, 1], dtype="complex128")
        m[:, 0, 0] = 1 + t
        if omega != 0:
            m[:, 3, 0] = -rho * omega**2 * (1 - t) / zsig
    return scm.ScaledMatrixStack(m)


def oceanFloorBoundary_deriv(depth, omega, k, sigma, rho, sh=False):
    # Derivative of (2011) Eqs. 86 & 88 wrt ocean depth (i.e. layer thickness)
    nk = k.shape[0]
    if sh:
        m = np.zeros([nk, 2, 1], dtype="complex128")
    else:
        zsig = np.lib.scimath.sqrt(k**2 - rho * omega**2 / sigma)
        t = np.exp(-2 * depth * zsig)
        m = np.zeros([nk, 6, 1], dtype="complex128")
        m[:, 0, 0] = -2 * t * zsig
        if omega != 0:
            m[:, 3, 0] = -2 * rho * t * omega**2
    return scm.ScaledMatrixStack(m)


def underlyingHalfspaceBoundary(omega, k, sigma, mu, rho, sh=False):
    # (2011) Eqs. 89 & 90
    nk = k.shape[0]

    zmu = np.lib.scimath.sqrt(k**2 - rho * omega**2 / mu)
    if sh:
        m = np.zeros([nk, 2, 1], dtype="complex128")
        m[:, 0, 0] = 1
        m[:, 1, 0] = mu * zmu
    else:
        zsig = np.lib.scimath.sqrt(k**2 - rho * omega**2 / sigma)
        # xi <-- (2011) Eq. 91
        xi = np.zeros_like(zsig)
        xi[k == 0] = np.sqrt(mu / sigma)
        xi[k > 0] = (rho * omega**2 / sigma - k[k > 0] ** 2 * (1 + mu / sigma)) / (
            k[k > 0] ** 2 + zsig[k > 0] * zmu[k > 0]
        )
        m = np.zeros([nk, 6, 1], dtype="complex128")
        m[:, 0, 0] = xi / mu
        m[:, 1, 0] = 2 * k * (xi + 0.5)
        m[:, 2, 0] = -zsig
        m[:, 3, 0] = zmu
        m[:, 4, 0] = -2 * k * (xi + 0.5)
        m[:, 5, 0] = rho * omega**2 - 4 * k**2 * mu * (xi + 1)
    return scm.ScaledMatrixStack(m)


def sourceVector(MT, F, k, sigma, mu):
    # (2011) Eqs. 21 & 22.
    nk = k.shape[0]
    s = np.zeros([nk, 4, 5], dtype="complex128")  # P-SV system, eq.21
    s2 = np.zeros([nk, 2, 5], dtype="complex128")  # SH system, eq. 22
    s[:, 0, 2] = MT[2, 2] / sigma
    s[:, 2, 2] = -F[2]
    s[:, 3, 2] = (
        0.5 * k * (MT[0, 0] + MT[1, 1]) - k * (sigma - 2 * mu) * MT[2, 2] / sigma
    )
    s2[:, 1, 2] = 0.5 * k * (MT[0, 1] - MT[1, 0])
    for sgn in [-1, 1]:
        s[:, 1, 2 + sgn] = 0.5 * (sgn * MT[0, 2] - 1j * MT[1, 2]) / mu
        s[:, 2, 2 + sgn] = sgn * 0.5 * k * (MT[0, 2] - MT[2, 0]) + 0.5 * 1j * k * (
            MT[2, 1] - MT[1, 2]
        )
        s[:, 3, 2 + sgn] = 0.5 * (-sgn * F[0] + 1j * F[1])
        s[:, 3, 2 + 2 * sgn] = 0.25 * k * (
            MT[1, 1] - MT[0, 0]
        ) + sgn * 0.25 * 1j * k * (MT[0, 1] + MT[1, 0])
        s2[:, 0, 2 + sgn] = 0.5 * (-sgn * MT[1, 2] - 1j * MT[0, 2]) / mu
        s2[:, 1, 2 + sgn] = 0.5 * (sgn * F[1] + 1j * F[0])
        s2[:, 1, 2 + 2 * sgn] = 0.25 * k * (
            MT[0, 1] + MT[1, 0]
        ) - sgn * 0.25 * 1j * k * (MT[1, 1] - MT[0, 0])
    return scm.ScaledMatrixStack(s), scm.ScaledMatrixStack(s2)


def sourceVector_ddep(MT, F, omega, k, sigma, mu, rho):
    # (2012) Eqs. A37 & A38
    # i.e. derivative of (2011) Eqs. 21 & 22 wrt source depth.
    lam = sigma - 2 * mu
    gamma = mu * (3 * lam + 2 * mu) / (lam + 2 * mu)
    nk = k.shape[0]
    s = np.zeros([nk, 4, 5], dtype="complex128")  # P-SV system, eq. A37
    s2 = np.zeros([nk, 2, 5], dtype="complex128")  # SH system, eq. A38
    s[:, 0, 2] = F[2] / sigma
    s[:, 1, 2] = (
        -k
        * (sigma * (MT[0, 0] + MT[1, 1]) - 2 * (lam + mu) * MT[2, 2])
        / (2 * mu * sigma)
    )
    s[:, 2, 2] = (
        -0.5 * k**2 * (MT[0, 0] + MT[1, 1])
        + (k**2 * lam + rho * omega**2) * MT[2, 2] / sigma
    )
    s[:, 3, 2] = -k * lam * F[2] / sigma
    s2[:, 0, 2] = k * (-MT[0, 1] + MT[1, 0]) / (2 * mu)
    for sgn in [-1, 1]:
        s[:, 0, 2 + sgn] = (
            k
            * (
                (lam + mu) * (-sgn * MT[0, 2] + 1j * MT[1, 2])
                + mu * (sgn * MT[2, 0] - 1j * MT[2, 1])
            )
            / (2 * mu * sigma)
        )
        s[:, 1, 2 + sgn] = (sgn * F[0] - 1j * F[1]) / (2 * mu)
        s[:, 2, 2 + sgn] = 0.5 * k * (sgn * F[0] - 1j * F[1])
        s[:, 3, 2 + sgn] = (
            -sgn * mu * k**2 * (sigma - 2 * mu) * MT[2, 0]
            + (k**2 * mu * (2 * mu - 3 * sigma) + rho * sigma * omega**2)
            * (sgn * MT[2, 0] - 1j * MT[1, 2])
            - 1j * k**2 * mu * (2 * mu - sigma) * MT[2, 1]
        ) / (2 * mu * sigma)
        # s[:,3,2+sgn] = (k**2 * lam*(-sgn*MT[2,0]+1j*MT[2,1])+(sgn*MT[0,2]-1j*MT[1,2])*(k**2 * (lam*mu - (gamma+mu)*sigma+rho*sigma*omega**2))/mu)/(2*sigma)
        s[:, 1, 2 + 2 * sgn] = k * (MT[0, 0] - MT[1, 1]) / (4 * mu) - sgn * 1j * k * (
            MT[0, 1] + MT[1, 0]
        ) / (4 * mu)
        s[:, 2, 2 + 2 * sgn] = 0.25 * k**2 * (
            MT[0, 0] - MT[1, 1]
        ) - 0.25 * sgn * 1j * k**2 * (MT[0, 1] + MT[1, 0])
        s2[:, 0, 2 + sgn] = (-sgn * F[1] - 1j * F[0]) / (2 * mu)
        s2[:, 1, 2 + sgn] = (
            (k**2 * mu - rho * omega**2)
            * (sgn * MT[1, 2] + 1j * MT[0, 2])
            / (2 * mu)
        )
        s2[:, 0, 2 + 2 * sgn] = -k * (MT[0, 1] + MT[1, 0]) / (4 * mu) + sgn * 1j * k * (
            MT[1, 1] - MT[0, 0]
        ) / (4 * mu)
    return scm.ScaledMatrixStack(s), scm.ScaledMatrixStack(s2)


###################
### Propagation ###
###################


def exphyp(x):
    a = np.real(x)
    b = np.imag(x)
    sgn = np.sign(a)
    t = np.exp(-2 * sgn * a)
    return (
        0.5 * np.cos(b) * (1 + t) + 0.5j * np.sin(b) * sgn * (1 - t),
        0.5 * np.cos(b) * sgn * (1 - t) + 0.5j * np.sin(b) * (1 + t),
        sgn * a,
    )


def propagate_zerofreq(k, dz, sigma, mu, rho, m2=None, m4=None, m6=None, inplace=True):
    # See Mathematica notebook for equations
    # Propagator matrices evaluated at w=0
    """Perform propagation through a layer of thickness dz and physical
    properties (sigma, mu, rho) for a stack of minor vectors corresponding to
    spatial wavenumbers given in array k. Special case for zero (temporal) frequency.
    """
    nk = k.shape[0]
    c, s, scale = exphyp(dz * k)
    # Terms that don't change under h->-h
    if m2 is not None:
        # (2011) Eq. 84
        M = np.zeros((nk, 2, 2), dtype="complex128")
        M[:, 0, 0] = c
        M[:, 0, 1] = s / (mu * k)
        M[:, 1, 0] = mu * k * s
        M[:, 1, 1] = c
        if inplace:
            out = m2
        else:
            out = None
        # And do the propagation
        m2r = scm.ScaledMatrixStack(M, scale.copy()).matmul(m2, out=out)
        del M
    else:
        m2r = None
    if m4 is not None:
        # exp( h A' ) (eq. 62 at zero freq)
        exphap = np.zeros([nk, 4, 4], dtype="complex128")
        exphap[:, 0, 0] = c
        exphap[:, 0, 1] = dz * s * rho * (sigma - mu) / (2 * sigma * mu)
        exphap[:, 0, 2] = (
            rho * (c * dz * k * (mu - sigma) + s * (mu + sigma)) / (2 * k * mu * sigma)
        )
        exphap[:, 0, 3] = -s
        exphap[:, 1, 1] = c
        exphap[:, 1, 2] = -s
        exphap[:, 2, 1] = -s
        exphap[:, 2, 2] = c
        exphap[:, 3, 0] = -s
        exphap[:, 3, 1] = (
            rho * (c * dz * k * (mu - sigma) - s * (mu + sigma)) / (2 * k * mu * sigma)
        )
        exphap[:, 3, 2] = dz * s * rho * (sigma - mu) / (2 * sigma * mu)
        exphap[:, 3, 3] = c
        M = scm.ScaledMatrixStack(exphap, scale.copy())
        del exphap
        # (2011) Eq. 55
        Z = np.zeros([nk, 4, 4], dtype="complex128")
        rtrho = np.sqrt(rho)
        Z[:, 0, 0] = 1 / rtrho
        Z[:, 1, 3] = -1 / rtrho
        Z[:, 2, 2] = rtrho
        Z[:, 2, 3] = -2 * k * mu / rtrho
        Z[:, 3, 0] = 2 * k * mu / rtrho
        Z[:, 3, 1] = rtrho
        # Z^{-1}
        iZ = np.zeros([nk, 4, 4], dtype="complex128")
        iZ[:, 0, 0] = rtrho
        iZ[:, 1, 0] = -2 * k * mu / rtrho
        iZ[:, 1, 3] = 1 / rtrho
        iZ[:, 2, 1] = -2 * k * mu / rtrho
        iZ[:, 2, 2] = 1 / rtrho
        iZ[:, 3, 1] = -rtrho
        if inplace:
            out = m4
        else:
            out = None
        m4r = scm.ScaledMatrixStack(Z).matmul(
            M.matmul(scm.ScaledMatrixStack(iZ).matmul(m4, out=out), out=out), out=out
        )
        del Z, iZ, M
    else:
        m4r = None
    if m6 is not None:
        # See Mathematica notebook. exp(h A' ) evaluated at zero frequency
        # and then split into two parts: one containing cosh/sinh terms
        # and the other without (because they need to be rescaled differently)
        # Rescaling of some components then rolled into definition of Z/inv(Z)
        exphap = np.zeros([nk, 6, 6], dtype="complex128")
        exphap[:, 0, 0] = c**2
        exphap[:, 0, 1] = -c * s
        exphap[:, 0, 3] = -rho * c * s * (mu + sigma)
        exphap[:, 0, 4] = c * s
        exphap[:, 0, 5] = -(s**2)

        exphap[:, 1, 0] = -c * s
        exphap[:, 1, 1] = c**2
        exphap[:, 1, 3] = s**2 * rho * (mu + sigma)
        exphap[:, 1, 4] = -(s**2)
        exphap[:, 1, 5] = c * s

        exphap[:, 2, 0] = -rho * c * s * (mu + sigma)
        exphap[:, 2, 1] = s**2 * rho * (mu + sigma)
        exphap[:, 2, 3] = (s * rho * (mu + sigma)) ** 2
        exphap[:, 2, 4] = -(s**2) * rho * (mu + sigma)
        exphap[:, 2, 5] = rho * c * s * (mu + sigma)

        exphap[:, 4, 0] = c * s
        exphap[:, 4, 1] = -(s**2)
        exphap[:, 4, 3] = -(s**2) * rho * (mu + sigma)
        exphap[:, 4, 4] = c**2
        exphap[:, 4, 5] = -c * s

        exphap[:, 5, 0] = -(s**2)
        exphap[:, 5, 1] = c * s
        exphap[:, 5, 3] = rho * c * s * (mu + sigma)
        exphap[:, 5, 4] = -c * s
        exphap[:, 5, 5] = c**2

        exphap_noscale = np.zeros([nk, 6, 6], dtype="complex128")
        exphap_noscale[:, 0, 3] = -rho * dz * k * (mu - sigma)
        exphap_noscale[:, 2, 0] = rho * dz * k * (mu - sigma)
        exphap_noscale[:, 2, 2] = 2 * k * mu * sigma
        exphap_noscale[:, 2, 3] = -((rho * dz * k * (mu - sigma)) ** 2)
        exphap_noscale[:, 2, 5] = rho * dz * k * (mu - sigma)
        exphap_noscale[:, 3, 3] = 2 * k * mu * sigma
        exphap_noscale[:, 5, 3] = -rho * dz * k * (mu - sigma)
        M = scm.ScaledMatrixStack(exphap, 2 * scale.copy()) + scm.ScaledMatrixStack(
            exphap_noscale
        )
        # These are not Z/inv(Z) as defined in (2011) paper -- 
        # we have pulled some scale factors out of exp(h A') and into
        # these matrices.
        Z = np.zeros([nk, 6, 6], dtype="complex128")
        Z[:, 0, 2] = -1 / (2 * k * mu * rho * sigma)
        Z[:, 1, 1] = 1
        Z[:, 1, 2] = -1 / (rho * sigma)
        Z[:, 2, 0] = 1
        Z[:, 3, 5] = 1
        Z[:, 4, 2] = 1 / (rho * sigma)
        Z[:, 4, 4] = 1
        Z[:, 5, 1] = -2 * k * mu
        Z[:, 5, 2] = 2 * k * mu / (rho * sigma)
        Z[:, 5, 3] = -rho
        Z[:, 5, 4] = 2 * k * mu
        # Minors of Z^{-1}
        iZ = np.zeros([nk, 6, 6], dtype="complex128")
        iZ[:, 0, 2] = 1
        iZ[:, 1, 0] = -2 * k * mu
        iZ[:, 1, 1] = 1
        iZ[:, 2, 0] = -rho
        iZ[:, 3, 0] = 2 * (mu * k) / (rho * sigma)
        iZ[:, 3, 1] = -1 / (rho * sigma)
        iZ[:, 3, 4] = 1 / (rho * sigma)
        iZ[:, 3, 5] = -1 / (2 * k * mu * rho * sigma)
        iZ[:, 4, 0] = 2 * k * mu
        iZ[:, 4, 4] = 1
        iZ[:, 5, 3] = 1

        if inplace:
            out = m6
        else:
            out = None
        m6r = scm.ScaledMatrixStack(Z).matmul(
            M.matmul(scm.ScaledMatrixStack(iZ).matmul(m6, out=out), out=out), out=out
        )
    else:
        m6r = None
    return m2r, m4r, m6r


def propagate_zerofreq_deriv(
    k, dz, sigma, mu, rho, m2=None, m4=None, m6=None, inplace=True
):
    nk = k.shape[0]
    c, s, scale = exphyp(dz * k)
    # Terms that don't change under h->-h
    if m2 is not None:
        M = np.zeros((nk, 2, 2), dtype="complex128")
        M[:, 0, 0] = k * s
        M[:, 0, 1] = c / mu
        M[:, 1, 0] = mu * k**2 * c
        M[:, 1, 1] = k * s
        if inplace:
            out = m2
        else:
            out = None
        m2r = scm.ScaledMatrixStack(M, scale.copy()).matmul(m2, out=out)
        del M
    else:
        m2r = None
    if m4 is not None:
        exphap = np.zeros([nk, 4, 4], dtype="complex128")
        exphap[:, 0, 0] = k * s
        exphap[:, 0, 1] = -((c * dz * k + s) * rho * (mu - sigma)) / (2 * mu * sigma)
        exphap[:, 0, 2] = (
            rho * (2 * c * mu + dz * k * s * (mu - sigma)) / (2 * mu * sigma)
        )
        exphap[:, 0, 3] = -c * k

        exphap[:, 1, 1] = k * s
        exphap[:, 1, 2] = -c * k

        exphap[:, 2, 1] = -c * k
        exphap[:, 2, 2] = k * s

        exphap[:, 3, 0] = -c * k
        exphap[:, 3, 1] = (
            rho * (dz * k * s * (mu - sigma) - 2 * c * sigma) / (2 * mu * sigma)
        )
        exphap[:, 3, 2] = -(c * dz * k + s) * rho * (mu - sigma) / (2 * mu * sigma)
        exphap[:, 3, 3] = k * s
        M = scm.ScaledMatrixStack(exphap, scale.copy())
        del exphap
        Z = np.zeros([nk, 4, 4], dtype="complex128")
        rtrho = np.sqrt(rho)
        Z[:, 0, 0] = 1 / rtrho
        Z[:, 1, 3] = -1 / rtrho
        Z[:, 2, 2] = rtrho
        Z[:, 2, 3] = -2 * k * mu / rtrho
        Z[:, 3, 0] = 2 * k * mu / rtrho
        Z[:, 3, 1] = rtrho
        iZ = np.zeros([nk, 4, 4], dtype="complex128")
        iZ[:, 0, 0] = rtrho
        iZ[:, 1, 0] = -2 * k * mu / rtrho
        iZ[:, 1, 3] = 1 / rtrho
        iZ[:, 2, 1] = -2 * k * mu / rtrho
        iZ[:, 2, 2] = 1 / rtrho
        iZ[:, 3, 1] = -rtrho
        if inplace:
            out = m4
        else:
            out = None
        m4r = scm.ScaledMatrixStack(Z).matmul(
            M.matmul(scm.ScaledMatrixStack(iZ).matmul(m4, out=out), out=out), out=out
        )
        del Z, iZ, M
    else:
        m4r = None
    if m6 is not None:
        exphap = np.zeros([nk, 6, 6], dtype="complex128")
        fac = rho * (mu + sigma) / (mu * sigma)
        exphap[:, 0, 0] = 2 * c * k * s
        exphap[:, 0, 1] = -2 * k * s**2
        exphap[:, 0, 3] = -fac * s**2
        exphap[:, 0, 4] = 2 * k * s**2
        exphap[:, 0, 5] = -2 * c * s * k

        exphap[:, 1, 0] = -2 * k * s**2
        exphap[:, 1, 1] = 2 * c * s * k
        exphap[:, 1, 3] = c * fac * s
        exphap[:, 1, 4] = -2 * c * k * s
        exphap[:, 1, 5] = 2 * k * s**2

        exphap[:, 2, 0] = -fac * s**2
        exphap[:, 2, 1] = c * fac * s
        exphap[:, 2, 3] = c * s * fac**2 / (2 * k)
        exphap[:, 2, 4] = -c * fac * s
        exphap[:, 2, 5] = fac * s**2

        exphap[:, 4, 0] = 2 * k * s**2
        exphap[:, 4, 1] = -2 * c * k * s
        exphap[:, 4, 3] = -c * fac * s
        exphap[:, 4, 4] = 2 * c * k * s
        exphap[:, 4, 5] = -2 * k * s**2

        exphap[:, 5, 0] = -2 * c * k * s
        exphap[:, 5, 1] = 2 * k * s**2
        exphap[:, 5, 3] = fac * s**2
        exphap[:, 5, 4] = -2 * k * s**2
        exphap[:, 5, 5] = 2 * c * k * s

        exphap_noscale = np.zeros([nk, 6, 6], dtype="complex128")
        exphap_noscale[:, 0, 1] = -k
        exphap_noscale[:, 0, 3] = -rho / sigma
        exphap_noscale[:, 0, 4] = k
        exphap_noscale[:, 1, 0] = -k
        exphap_noscale[:, 1, 5] = k
        exphap_noscale[:, 2, 0] = -rho / mu
        exphap_noscale[:, 2, 3] = -0.5 * dz * (fac * (mu - sigma) / (mu + sigma)) ** 2
        exphap_noscale[:, 2, 5] = rho / sigma
        exphap_noscale[:, 4, 0] = k
        exphap_noscale[:, 4, 5] = -k
        exphap_noscale[:, 5, 1] = k
        exphap_noscale[:, 5, 3] = rho / mu
        exphap_noscale[:, 5, 4] = -k
        M = scm.ScaledMatrixStack(exphap, 2 * scale.copy()) + scm.ScaledMatrixStack(
            exphap_noscale
        )

        Z = np.zeros([nk, 6, 6], dtype="complex128")
        Z[:, 0, 2] = -1 / rho
        Z[:, 1, 1] = 1
        Z[:, 1, 2] = -2 * k * mu / rho
        Z[:, 2, 0] = 1
        Z[:, 3, 5] = 1
        Z[:, 4, 2] = 2 * k * mu / rho
        Z[:, 4, 4] = 1
        Z[:, 5, 1] = -2 * k * mu
        Z[:, 5, 2] = 4 * k**2 * mu**2 / rho
        Z[:, 5, 3] = -rho
        Z[:, 5, 4] = 2 * k * mu

        iZ = np.zeros([nk, 6, 6], dtype="complex128")
        iZ[:, 0, 2] = 1
        iZ[:, 1, 0] = -2 * k * mu
        iZ[:, 1, 1] = 1
        iZ[:, 2, 0] = -rho
        iZ[:, 3, 0] = 4 * mu**2 * k**2 / rho
        iZ[:, 3, 1] = -2 * k * mu / rho
        iZ[:, 3, 4] = 2 * k * mu / rho
        iZ[:, 3, 5] = -1 / rho
        iZ[:, 4, 0] = 2 * k * mu
        iZ[:, 4, 4] = 1
        iZ[:, 5, 3] = 1

        if inplace:
            out = m6
        else:
            out = None
        m6r = scm.ScaledMatrixStack(Z).matmul(
            M.matmul(scm.ScaledMatrixStack(iZ).matmul(m6, out=out), out=out), out=out
        )
    else:
        m6r = None
    return m2r, m4r, m6r


def propagate(omega, k, dz, sigma, mu, rho, m2=None, m4=None, m6=None, inplace=True):
    # Propagate the systems in m2/m4/m6 through layer.
    if mu == 0:
        raise NotImplementedError(
            "Propagation through fluid layer not currently implemented. If you have an ocean layer, check that your receivers are placed at or below the sea floor."
        )
    if np.any(k == 0):
        raise ValueError("propagate does not handle k==0.")
    if omega == 0:  # Special handler for zero-frequency system
        return propagate_zerofreq(k, dz, sigma, mu, rho, m2, m4, m6, inplace)
    nk = k.shape[0]
    if m4 is not None or m6 is not None:
        zsig = np.lib.scimath.sqrt(k**2 - rho * omega**2 / sigma)
        csig, ssig, scalesig = exphyp(dz * zsig)
    zmu = np.lib.scimath.sqrt(k**2 - rho * omega**2 / mu)
    cmu, smu, scalemu = exphyp(dz * zmu)
    if m2 is not None:
        M = np.zeros((nk, 2, 2), dtype="complex128")
        M[:, 0, 0] = cmu
        M[:, 0, 1] = smu / (mu * zmu)
        M[:, 1, 0] = mu * zmu * smu
        M[:, 1, 1] = cmu
        if inplace:
            out = m2
        else:
            out = None
        m2r = scm.ScaledMatrixStack(M, scalemu.copy()).matmul(m2, out=out)
        del M
    else:
        m2r = None
    if m4 is not None:
        exphap_s = np.zeros([nk, 4, 4], dtype="complex128")
        exphap_s[:, 0, 0] = csig
        exphap_s[:, 0, 1] = csig * k / omega**2
        exphap_s[:, 0, 2] = -ssig * zsig / omega**2
        exphap_s[:, 2, 0] = -ssig * omega**2 / zsig
        exphap_s[:, 2, 1] = -k * ssig / zsig
        exphap_s[:, 2, 2] = csig
        exphap_s[:, 3, 0] = -k * ssig / zsig
        exphap_s[:, 3, 1] = -(k**2) * ssig / (zsig * omega**2)
        exphap_s[:, 3, 2] = k * csig / omega**2
        M = scm.ScaledMatrixStack(exphap_s, scalesig.copy())
        del exphap_s  # Don't need explicit reference; reference in M still exists.

        exphap_m = np.zeros([nk, 4, 4], dtype="complex128")
        exphap_m[:, 0, 1] = -k * cmu / omega**2
        exphap_m[:, 0, 2] = k**2 * smu / (zmu * omega**2)
        exphap_m[:, 0, 3] = -k * smu / zmu
        exphap_m[:, 1, 1] = cmu
        exphap_m[:, 1, 2] = -k * smu / zmu
        exphap_m[:, 1, 3] = smu * omega**2 / zmu
        exphap_m[:, 3, 1] = smu * zmu / omega**2
        exphap_m[:, 3, 2] = -k * cmu / omega**2
        exphap_m[:, 3, 3] = cmu

        M += scm.ScaledMatrixStack(exphap_m, scalemu.copy())
        del exphap_m

        rtrho = np.sqrt(rho)

        Z = np.zeros([nk, 4, 4], dtype="complex128")
        Z[:, 0, 0] = 1 / rtrho
        Z[:, 1, 3] = -1 / rtrho
        Z[:, 2, 2] = rtrho
        Z[:, 2, 3] = -2 * mu * k / rtrho
        Z[:, 3, 0] = 2 * mu * k / rtrho
        Z[:, 3, 1] = rtrho

        iZ = np.zeros([nk, 4, 4], dtype="complex128")
        iZ[:, 0, 0] = rtrho
        iZ[:, 1, 0] = -2 * mu * k / rtrho
        iZ[:, 1, 3] = 1 / rtrho
        iZ[:, 2, 1] = -2 * mu * k / rtrho
        iZ[:, 2, 2] = 1 / rtrho
        iZ[:, 3, 1] = -rtrho
        if inplace:
            out = m4
        else:
            out = None
        m4r = scm.ScaledMatrixStack(Z).matmul(
            M.matmul(scm.ScaledMatrixStack(iZ).matmul(m4, out=out), out=out), out=out
        )
        del Z, iZ, M
    else:
        m4r = None
    if m6 is not None:
        xiprod = zsig * zmu / k**2
        Pc = cmu * csig
        Ps = smu * ssig
        X1 = cmu * ssig
        X2 = csig * smu

        # Split matrix into term containing cosh/sinh products (which need scaling by exp(scale)) and the rest.
        M1 = np.zeros([nk, 6, 6], dtype="complex128")

        M1[:, 0, 0] = Pc
        M1[:, 0, 1] = -X2
        M1[:, 0, 2] = X2
        M1[:, 0, 3] = -X2 + xiprod * X1
        M1[:, 0, 4] = X2
        M1[:, 0, 5] = -Ps

        M1[:, 1, 0] = -X1
        M1[:, 1, 1] = Ps
        M1[:, 1, 2] = -Ps
        M1[:, 1, 3] = Ps - Pc * xiprod
        M1[:, 1, 4] = -Ps
        M1[:, 1, 5] = X2

        M1[:, 2, 0] = -X1 + xiprod * X2
        M1[:, 2, 1] = Ps - Pc * xiprod
        M1[:, 2, 2] = -Ps + Pc * xiprod
        M1[:, 2, 3] = -2 * Pc * xiprod + Ps * (1 + xiprod**2)
        M1[:, 2, 4] = -Ps + Pc * xiprod
        M1[:, 2, 5] = X2 - X1 * xiprod

        M1[:, 3, 0] = X1
        M1[:, 3, 1] = -Ps
        M1[:, 3, 2] = Ps
        M1[:, 3, 3] = -Ps + Pc * xiprod
        M1[:, 3, 4] = Ps
        M1[:, 3, 5] = -X2

        M1[:, 4, 0] = X1
        M1[:, 4, 1] = -Ps
        M1[:, 4, 2] = Ps
        M1[:, 4, 3] = -Ps + Pc * xiprod
        M1[:, 4, 4] = Ps
        M1[:, 4, 5] = -X2

        M1[:, 5, 0] = -Ps
        M1[:, 5, 1] = X1
        M1[:, 5, 2] = -X1
        M1[:, 5, 3] = X1 - X2 * xiprod
        M1[:, 5, 4] = -X1
        M1[:, 5, 5] = Pc
        M = scm.ScaledMatrixStack(M1, scalemu + scalesig)
        del M1

        M2 = np.zeros([nk, 6, 6], dtype="complex128")
        M2[:, 1, 1] = xiprod
        M2[:, 1, 3] = xiprod
        M2[:, 2, 1] = xiprod
        M2[:, 2, 3] = 2 * xiprod
        M2[:, 2, 4] = -xiprod
        M2[:, 4, 3] = -xiprod
        M2[:, 4, 4] = xiprod
        M += scm.ScaledMatrixStack(M2)
        del M2

        Z = np.zeros([nk, 6, 6], dtype="complex128")
        Z[:, 0, 2] = -(k**2) / (rho * omega**2)
        Z[:, 1, 1] = k
        Z[:, 1, 2] = -2 * k**3 * mu / (rho * omega**2)
        Z[:, 2, 0] = zsig
        Z[:, 3, 5] = zmu
        Z[:, 4, 2] = 2 * k**3 * mu / (rho * omega**2)
        Z[:, 4, 4] = k
        Z[:, 5, 1] = -2 * k**2 * mu
        Z[:, 5, 2] = 4 * k**4 * mu**2 / (rho * omega**2)
        Z[:, 5, 3] = -rho * omega**2
        Z[:, 5, 4] = 2 * k**2 * mu

        iZ = np.zeros([nk, 6, 6], dtype="complex128")
        iZ[:, 0, 2] = 1 / zsig
        iZ[:, 1, 0] = -2 * k**2 * mu / (zsig * zmu)
        iZ[:, 1, 1] = k / (zsig * zmu)
        iZ[:, 2, 0] = -rho * omega**2 / (zsig * zmu)
        iZ[:, 3, 0] = 4 * k**4 * mu**2 / (rho * omega**2 * zsig * zmu)
        iZ[:, 3, 1] = -2 * k**3 * mu / (rho * omega**2 * zsig * zmu)
        iZ[:, 3, 4] = 2 * k**3 * mu / (rho * omega**2 * zsig * zmu)
        iZ[:, 3, 5] = -(k**2) / (rho * omega**2 * zsig * zmu)
        iZ[:, 4, 0] = 2 * k**2 * mu / (zsig * zmu)
        iZ[:, 4, 4] = k / (zsig * zmu)
        iZ[:, 5, 3] = 1 / zmu
        if inplace:
            out = m6
        else:
            out = None
        m6r = scm.ScaledMatrixStack(Z).matmul(
            M.matmul(scm.ScaledMatrixStack(iZ).matmul(m6, out=out), out=out), out=out
        )
    else:
        m6r = None
    return m2r, m4r, m6r


def propagate_deriv(
    omega, k, dz, sigma, mu, rho, m2=None, m4=None, m6=None, inplace=True
):
    if np.any(k == 0):
        raise ValueError("propagate_deriv does not handle k==0.")
    if omega == 0:  # Special handler for zero-frequency system
        return propagate_zerofreq_deriv(k, dz, sigma, mu, rho, m2, m4, m6, inplace)
    nk = k.shape[0]
    if m4 is not None or m6 is not None:
        zsig = np.lib.scimath.sqrt(k**2 - rho * omega**2 / sigma)
        csig, ssig, scalesig = exphyp(dz * zsig)
    zmu = np.lib.scimath.sqrt(k**2 - rho * omega**2 / mu)
    cmu, smu, scalemu = exphyp(dz * zmu)
    if m2 is not None:
        M = np.zeros((nk, 2, 2), dtype="complex128")
        M[:, 0, 0] = zmu * smu
        M[:, 0, 1] = cmu / mu
        M[:, 1, 0] = mu * cmu * zmu**2
        M[:, 1, 1] = zmu * smu
        if inplace:
            out = m2
        else:
            out = None
        m2r = scm.ScaledMatrixStack(M, scalemu.copy()).matmul(m2, out=out)
        del M
    else:
        m2r = None
    if m4 is not None:
        exphap_s = np.zeros([nk, 4, 4], dtype="complex128")
        exphap_s[:, 0, 0] = ssig * zsig
        exphap_s[:, 0, 1] = k * ssig * zsig / omega**2
        exphap_s[:, 0, 2] = -csig * (zsig / omega) ** 2
        exphap_s[:, 2, 0] = -(omega**2) * csig
        exphap_s[:, 2, 1] = -k * csig
        exphap_s[:, 2, 2] = ssig * zsig
        exphap_s[:, 3, 0] = -k * csig
        exphap_s[:, 3, 1] = -((k / omega) ** 2) * csig
        exphap_s[:, 3, 2] = k * ssig * zsig / omega**2
        M = scm.ScaledMatrixStack(exphap_s, scalesig.copy())
        del exphap_s

        exphap_m = np.zeros([nk, 4, 4], dtype="complex128")
        exphap_m[:, 0, 1] = -k * smu * zmu / omega**2
        exphap_m[:, 0, 2] = (k / omega) ** 2 * cmu
        exphap_m[:, 0, 3] = -k * cmu
        exphap_m[:, 1, 1] = smu * zmu
        exphap_m[:, 1, 2] = -k * cmu
        exphap_m[:, 1, 3] = omega**2 * cmu
        exphap_m[:, 3, 1] = cmu * (zmu / omega) ** 2
        exphap_m[:, 3, 2] = -k * zmu * smu / omega**2
        exphap_m[:, 3, 3] = smu * zmu
        M += scm.ScaledMatrixStack(exphap_m, scalemu.copy())
        del exphap_m
        rtrho = np.sqrt(rho)
        Z = np.zeros([nk, 4, 4], dtype="complex128")
        Z[:, 0, 0] = 1 / rtrho
        Z[:, 1, 3] = -1 / rtrho
        Z[:, 2, 2] = rtrho
        Z[:, 2, 3] = -2 * mu * k / rtrho
        Z[:, 3, 0] = 2 * mu * k / rtrho
        Z[:, 3, 1] = rtrho

        iZ = np.zeros([nk, 4, 4], dtype="complex128")
        iZ[:, 0, 0] = rtrho
        iZ[:, 1, 0] = -2 * mu * k / rtrho
        iZ[:, 1, 3] = 1 / rtrho
        iZ[:, 2, 1] = -2 * mu * k / rtrho
        iZ[:, 2, 2] = 1 / rtrho
        iZ[:, 3, 1] = -rtrho
        if inplace:
            out = m4
        else:
            out = None
        m4r = scm.ScaledMatrixStack(Z).matmul(
            M.matmul(scm.ScaledMatrixStack(iZ).matmul(m4, out=out), out=out), out=out
        )
        del Z, iZ, M
    else:
        m4r = None
    if m6 is not None:
        Pc = cmu * csig
        Ps = smu * ssig
        X1 = cmu * ssig
        X2 = csig * smu
        M = np.zeros([nk, 6, 6], dtype="complex128")
        M[:, 0, 0] = X2 * zmu + X1 * zsig
        M[:, 0, 1] = -Pc - Ps * zsig / zmu
        M[:, 0, 2] = Pc + Ps * zsig / zmu
        M[:, 0, 3] = -Pc / sigma - Ps * zsig / (zmu * mu)
        M[:, 0, 4] = Pc + Ps * zsig / zmu
        M[:, 0, 5] = -zsig * (X1 + X2 * zsig / zmu)

        M[:, 1, 0] = -Pc - Ps * zmu / zsig
        M[:, 1, 1] = X2 / zmu + X1 / zsig
        M[:, 1, 2] = -X2 / zmu - X1 / zsig
        M[:, 1, 3] = X2 / (mu * zmu) + X1 / (sigma * zsig)
        M[:, 1, 4] = -X2 / zmu - X1 / zsig
        M[:, 1, 5] = Pc + Ps * zsig / zmu

        M[:, 2, 0] = -Pc / mu - Ps * zmu / (sigma * zsig)
        M[:, 2, 1] = X2 / (mu * zmu) + X1 / (sigma * zsig)
        M[:, 2, 2] = -X2 / (mu * zmu) - X1 / (sigma * zsig)
        M[:, 2, 3] = (
            (k**2 * (mu - sigma) + rho * omega**2) * X1 * zmu
            + (k**2 * (sigma - mu) + rho * omega**2) * X2 * zsig
        ) / (mu * rho * sigma * omega**2 * zmu * zsig)
        M[:, 2, 4] = -X2 / (mu * zmu) - X1 / (sigma * zsig)
        M[:, 2, 5] = Pc / sigma + Ps * zsig / (mu * zmu)

        M[:, 3, 0] = Pc + Ps * zmu / zsig
        M[:, 3, 1] = -X2 / zmu - X1 / zsig
        M[:, 3, 2] = X2 / zmu + X1 / zsig
        M[:, 3, 3] = -X2 / (mu * zmu) - X1 / (sigma * zsig)
        M[:, 3, 4] = X2 / zmu + X1 / zsig
        M[:, 3, 5] = -Pc - Ps * zsig / zmu

        M[:, 4, 0] = Pc + Ps * zmu / zsig
        M[:, 4, 1] = -X2 / zmu - X1 / zsig
        M[:, 4, 2] = X2 / zmu + X1 / zsig
        M[:, 4, 3] = -X2 / (mu * zmu) - X1 / (sigma * zsig)
        M[:, 4, 4] = X2 / zmu + X1 / zsig
        M[:, 4, 5] = -Pc - Ps * zsig / zmu

        M[:, 5, 0] = zmu * (-X2 - X1 * zmu / zsig)
        M[:, 5, 1] = Pc + Ps * zmu / zsig
        M[:, 5, 2] = -Pc - Ps * zmu / zsig
        M[:, 5, 3] = Pc / mu + Ps * zmu / (sigma * zsig)
        M[:, 5, 4] = -Pc - Ps * zmu / zsig
        M[:, 5, 5] = X2 * zmu + X1 * zsig
        M = scm.ScaledMatrixStack(M, scalemu + scalesig)
        Z = np.zeros([nk, 6, 6], dtype="complex128")
        Z[:, 0, 2] = -1
        Z[:, 1, 1] = k
        Z[:, 1, 2] = -2 * k * mu
        Z[:, 2, 0] = 1
        Z[:, 3, 5] = 1
        Z[:, 4, 2] = 2 * k * mu
        Z[:, 4, 4] = k
        Z[:, 5, 1] = -2 * k**2 * mu
        Z[:, 5, 2] = 4 * k**2 * mu**2
        Z[:, 5, 3] = -rho * omega**2
        Z[:, 5, 4] = 2 * k**2 * mu
        iZ = np.zeros([nk, 6, 6], dtype="complex128")
        iZ[:, 0, 2] = 1
        iZ[:, 1, 0] = -2 * mu * k**2
        iZ[:, 1, 1] = k
        iZ[:, 2, 0] = -rho * omega**2
        iZ[:, 3, 0] = 4 * k**2 * mu**2
        iZ[:, 3, 1] = -2 * k * mu
        iZ[:, 3, 4] = 2 * k * mu
        iZ[:, 3, 5] = -1
        iZ[:, 4, 0] = 2 * mu * k**2
        iZ[:, 4, 4] = k
        iZ[:, 5, 3] = 1
        if inplace:
            out = m6
        else:
            out = None
        m6r = scm.ScaledMatrixStack(Z).matmul(
            M.matmul(scm.ScaledMatrixStack(iZ).matmul(m6, out=out), out=out), out=out
        )
    else:
        m6r = None
    return m2r, m4r, m6r


def makeN(s):
    # (2011) Eq. 52
    #
    m = s.M
    N = np.zeros([s.nStack, 4, 4], dtype="complex128")
    #
    # R = np.zeros([4,4,6])
    # R[0,0,1] = -1
    # R[0,1,2] = -1
    # R[0,3,0] = 1
    # R[1,0,3] = -1
    # R[1,1,4] = -1
    # R[1,2,0] = -1
    # R[2,1,5] = -1
    # R[2,2,1] = -1
    # R[2,3,3] = -1
    # R[3,0,5] = 1
    # R[3,2,2] = -1
    # R[3,3,4] = -1
    N[:, 0, 0] = -m[:, 1, 0]
    N[:, 0, 1] = -m[:, 2, 0]
    N[:, 0, 3] = m[:, 0, 0]
    N[:, 1, 0] = -m[:, 3, 0]
    N[:, 1, 1] = -m[:, 4, 0]
    N[:, 1, 2] = -m[:, 0, 0]
    N[:, 2, 1] = -m[:, 5, 0]
    N[:, 2, 2] = -m[:, 1, 0]
    N[:, 2, 3] = -m[:, 3, 0]
    N[:, 3, 0] = m[:, 5, 0]
    N[:, 3, 2] = -m[:, 2, 0]
    N[:, 3, 3] = -m[:, 4, 0]
    return scm.ScaledMatrixStack(N, s.scale.copy())


def makeDelta(scm1, scm2, sh=False):
    # 2011 Eqs. 47 & 53
    m1 = scm1.M
    m2 = scm2.M
    if not scm1.nStack == scm2.nStack:
        raise ValueError("Dimension mismatch")
    m = np.zeros([scm1.nStack, 1, 1], dtype="complex128")
    if sh:
        m[:, 0, 0] = m1[:, 0, 0] * m2[:, 1, 0] - m1[:, 1, 0] * m2[:, 0, 0]
    else:
        m[:, 0, 0] = (
            m1[:, 0, 0] * m2[:, 5, 0]
            - m1[:, 1, 0] * m2[:, 4, 0]
            + m1[:, 2, 0] * m2[:, 3, 0]
            + m1[:, 3, 0] * m2[:, 2, 0]
            - m1[:, 4, 0] * m2[:, 1, 0]
            + m1[:, 5, 0] * m2[:, 0, 0]
        )
    return scm.ScaledMatrixStack(m, scm1.scale + scm2.scale)


def compute_H_matrices(k, omega, dz, sigma, mu, rho, isrc, irec, do_derivatives=False):
    """
    Compute the "H" matrices, as defined in O'Toole & Woodhouse 2011.
    """
    # (2011) Eqs. 46 &  54 (= eq. 51)
    #
    # Note on derivatives: the P-SV calculations are done with everything
    # propagated to the receiver depth, whereas SH calculations mix source
    # and receiver quantities. Thus the algorithmic structure of the derivative
    # calculation differs.
    nlayers = dz.shape[0]
    # Propagate surface b/c to receiver
    if do_derivatives:
        surface_bc_sh_drv = []
        surface_bc_psv_drv = []
    if mu[0] == 0:
        surface_bc_sh = oceanFloorBoundary(dz[0], omega, k, sigma[0], rho[0], True)
        surface_bc_psv = oceanFloorBoundary(dz[0], omega, k, sigma[0], rho[0], False)
        ibc = 1
        if do_derivatives:
            surface_bc_sh_drv += [
                oceanFloorBoundary_deriv(dz[0], omega, k, sigma[0], rho[0], True)
            ]
            surface_bc_psv_drv += [
                oceanFloorBoundary_deriv(dz[0], omega, k, sigma[0], rho[0], False)
            ]
    else:
        surface_bc_sh = freeSurfaceBoundary(k.shape[0], True)
        surface_bc_psv = freeSurfaceBoundary(k.shape[0], False)
        ibc = 0

    for i in range(ibc, irec):
        # Deal with derivatives first - before we do in-place propagation of the vectors
        if do_derivatives:
            for j in range(len(surface_bc_sh_drv)):
                # Normal propagation on every derivative that already exists
                surface_bc_sh_drv[j], _, surface_bc_psv_drv[j] = propagate(
                    omega,
                    k,
                    -dz[i],
                    sigma[i],
                    mu[i],
                    rho[i],
                    m2=surface_bc_sh_drv[j],
                    m6=surface_bc_psv_drv[j],
                )
            sh_drv, _, psv_drv = propagate_deriv(
                omega,
                k,
                -dz[i],
                sigma[i],
                mu[i],
                rho[i],
                m2=surface_bc_sh,
                m6=surface_bc_psv,
                inplace=False,
            )
            surface_bc_sh_drv += [sh_drv]
            surface_bc_psv_drv += [psv_drv]
        surface_bc_sh, _, surface_bc_psv = propagate(
            omega,
            k,
            -dz[i],
            sigma[i],
            mu[i],
            rho[i],
            m2=surface_bc_sh,
            m6=surface_bc_psv,
        )
    if do_derivatives:
        for i in range(irec, nlayers - 1):
            surface_bc_sh_drv += [None]

    # Propagate basal b/c to source depth
    basal_bc_sh = underlyingHalfspaceBoundary(
        omega, k, sigma[-1], mu[-1], rho[-1], True
    )
    basal_bc_psv = underlyingHalfspaceBoundary(
        omega, k, sigma[-1], mu[-1], rho[-1], False
    )
    if do_derivatives:
        basal_bc_sh_drv = []
        basal_bc_psv_drv = []
    for i in range(nlayers - 2, isrc - 1, -1):
        # print(i,dz[i],sigma[i],mu[i],rho[i])
        if do_derivatives:
            for j in range(len(basal_bc_sh_drv)):
                basal_bc_sh_drv[j], _, basal_bc_psv_drv[j] = propagate(
                    omega,
                    k,
                    dz[i],
                    sigma[i],
                    mu[i],
                    rho[i],
                    m2=basal_bc_sh_drv[j],
                    m6=basal_bc_psv_drv[j],
                )
            sh_drv, _, psv_drv = propagate_deriv(
                omega,
                k,
                dz[i],
                sigma[i],
                mu[i],
                rho[i],
                m2=basal_bc_sh,
                m6=basal_bc_psv,
                inplace=False,
            )
            basal_bc_sh_drv += [sh_drv]
            basal_bc_psv_drv += [psv_drv]
        basal_bc_sh, _, basal_bc_psv = propagate(
            omega, k, dz[i], sigma[i], mu[i], rho[i], m2=basal_bc_sh, m6=basal_bc_psv
        )
    basal_bc_sh_at_src = basal_bc_sh.copy()
    if do_derivatives:
        basal_bc_sh_drv_at_src = [b.copy() for b in basal_bc_sh_drv]
    # basal_bc_psv_at_src = basal_bc_psv.copy()
    # Create N and continue to propagate everything up to receiver
    N = makeN(basal_bc_psv)
    if do_derivatives:
        N_drv = [makeN(b) for b in basal_bc_psv_drv]
    for i in range(isrc - 1, irec - 1, -1):
        if do_derivatives:
            for j in range(len(basal_bc_sh_drv)):
                basal_bc_sh_drv[j], N_drv[j], basal_bc_psv_drv[j] = propagate(
                    omega,
                    k,
                    dz[i],
                    sigma[i],
                    mu[i],
                    rho[i],
                    m2=basal_bc_sh_drv[j],
                    m4=N_drv[j],
                    m6=basal_bc_psv_drv[j],
                )
            sh_drv, N_drv_entry, psv_drv = propagate_deriv(
                omega,
                k,
                dz[i],
                sigma[i],
                mu[i],
                rho[i],
                m2=basal_bc_sh,
                m4=N,
                m6=basal_bc_psv,
                inplace=False,
            )
            basal_bc_sh_drv += [sh_drv]
            basal_bc_sh_drv_at_src += [None]
            N_drv += [N_drv_entry]
            basal_bc_psv_drv += [psv_drv]
        basal_bc_sh, N, basal_bc_psv = propagate(
            omega,
            k,
            dz[i],
            sigma[i],
            mu[i],
            rho[i],
            m2=basal_bc_sh,
            m4=N,
            m6=basal_bc_psv,
        )
    if do_derivatives:
        for i in range(irec - 1, -1, -1):
            basal_bc_sh_drv += [None]
            basal_bc_sh_drv_at_src += [None]
    # Now assemble H
    H_psv = (makeN(surface_bc_psv) @ N) / makeDelta(surface_bc_psv, basal_bc_psv)
    H_sh = np.zeros([k.shape[0], 2, 2], dtype="complex128")
    H_sh[:, 0, 0] = surface_bc_sh.M[:, 0, 0] * basal_bc_sh_at_src.M[:, 1, 0]
    H_sh[:, 0, 1] = -surface_bc_sh.M[:, 0, 0] * basal_bc_sh_at_src.M[:, 0, 0]
    H_sh[:, 1, 0] = surface_bc_sh.M[:, 1, 0] * basal_bc_sh_at_src.M[:, 1, 0]
    H_sh[:, 1, 1] = -surface_bc_sh.M[:, 1, 0] * basal_bc_sh_at_src.M[:, 0, 0]
    H_sh = scm.ScaledMatrixStack(
        H_sh, surface_bc_sh.scale + basal_bc_sh_at_src.scale
    ) / makeDelta(surface_bc_sh, basal_bc_sh, sh=True)
    if do_derivatives:
        # We want to work top -> bottom so reverse the lists built bottom -> top
        basal_bc_sh_drv.reverse()
        basal_bc_sh_drv_at_src.reverse()
        basal_bc_psv_drv.reverse()
        N_drv.reverse()

        H_sh_drv = []
        H_psv_drv = []

        delta = makeDelta(surface_bc_psv, basal_bc_psv)
        for s in surface_bc_psv_drv:
            H_psv_drv += [((makeN(s) @ N) - H_psv * makeDelta(s, basal_bc_psv)) / delta]
        for Nd, b in zip(N_drv, basal_bc_psv_drv):
            H_psv_drv += [
                ((makeN(surface_bc_psv) @ Nd) - H_psv * makeDelta(surface_bc_psv, b))
                / delta
            ]
        for s, bsrc, b in zip(
            surface_bc_sh_drv, basal_bc_sh_drv_at_src, basal_bc_sh_drv
        ):
            D1 = np.zeros([k.shape[0], 2, 2], dtype="complex128")
            ddelta = scm.ScaledMatrixStack(
                np.zeros([k.shape[0], 1, 1], dtype="complex128")
            )
            if s is not None:
                D1[:, 0, 0] = s.M[:, 0, 0] * basal_bc_sh_at_src.M[:, 1, 0]
                D1[:, 0, 1] = -s.M[:, 0, 0] * basal_bc_sh_at_src.M[:, 0, 0]
                D1[:, 1, 0] = s.M[:, 1, 0] * basal_bc_sh_at_src.M[:, 1, 0]
                D1[:, 1, 1] = -s.M[:, 1, 0] * basal_bc_sh_at_src.M[:, 0, 0]
                D1 = scm.ScaledMatrixStack(D1, s.scale + basal_bc_sh_at_src.scale)
                ddelta += makeDelta(s, basal_bc_sh, sh=True)
            else:
                D1 = scm.ScaledMatrixStack(D1)
            D2 = np.zeros([k.shape[0], 2, 2], dtype="complex128")
            if bsrc is not None:
                D2[:, 0, 0] = surface_bc_sh.M[:, 0, 0] * bsrc.M[:, 1, 0]
                D2[:, 0, 1] = -surface_bc_sh.M[:, 0, 0] * bsrc.M[:, 0, 0]
                D2[:, 1, 0] = surface_bc_sh.M[:, 1, 0] * bsrc.M[:, 1, 0]
                D2[:, 1, 1] = -surface_bc_sh.M[:, 1, 0] * bsrc.M[:, 0, 0]
                D2 = scm.ScaledMatrixStack(D2, surface_bc_sh.scale + bsrc.scale)
            else:
                D2 = scm.ScaledMatrixStack(D2)
            if b is not None:
                ddelta += makeDelta(surface_bc_sh, b, sh=True)
            D = (D1 + D2 - H_sh * ddelta) / makeDelta(
                surface_bc_sh, basal_bc_sh, sh=True
            )
            H_sh_drv += [D]
    if do_derivatives:
        return H_psv, H_sh, H_psv_drv, H_sh_drv
    else:
        return H_psv, H_sh
