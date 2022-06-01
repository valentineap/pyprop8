import pyprop8 as pp
from pyprop8.utils import stf_trapezoidal,make_moment_tensor,rtf2xyz
import numpy as np

def tests():
    print("Running tests.")
    print("")
    print(" 1. Creating objects")
    model = pp.LayeredStructureModel(
        [
            (3.0, 1.8, 0.0, 1.02),
            (2.0, 4.5, 2.4, 2.57),
            (5.0, 5.8, 3.3, 2.63),
            (20.0, 6.5, 3.65, 2.85),
            (np.inf, 8.0, 4.56, 3.34),
        ]
    )

    source = pp.PointSource(
        0,
        0,
        20,
        rtf2xyz(make_moment_tensor(340, 70, 20, 2.4e8, 0, 0)),
        np.zeros([3, 1]),
        0,
    )
    stations = pp.RegularlyDistributedReceivers(30, 50, 3, 0, 360, 3, depth=3).asListOfReceivers()

    derivs = pp.DerivativeSwitches(x=True, y=True, z=True)

    source_time_function = lambda w: stf_trapezoidal(w, 3, 6)

    print(" 2. Computing seismograms...")

    nt=33 #257
    dt=0.5
    alpha=0.023
    pad_frac=1

    tt, seis0, drv = pp.compute_seismograms(
        model,
        source,
        stations,
        nt,
        dt,
        alpha,
        pad_frac=pad_frac,
        derivatives=derivs,
        source_time_function=source_time_function,
        xyz=True,
    )

    epsilon = 1e-4
    print(" 3. Comparing with finite-difference derivatives.")
    print("    Using finite-difference perturbation eps = %.2f metres"%(epsilon*1000))
    print("    a. Perturbing source in x.")


    source_x = source.copy()
    source_x.x += epsilon
    
    tt, seis_x = pp.compute_seismograms(
        model,
        source_x,
        stations,
        nt,
        dt,
        alpha,
        pad_frac=pad_frac,
        derivatives=None,
        source_time_function=source_time_function,
        xyz=True,
    )

    fd = (seis_x - seis0)/epsilon # finite difference estimate
    max_x = abs(drv[:,derivs.i_x,:,:]).max(-1).reshape(stations.nstations,3,1) # Maximum absolute value of trace
    perc_err_x = 100*(abs(drv[:,derivs.i_x,:,:]-fd)/max_x) #For each trace, calculate finite difference 'error' as percentage of trace amplitude
    print("       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"%perc_err_x.max())

    print("    b. Perturbing source in y.")
    source_y = source.copy()
    source_y.y += epsilon
    
    tt, seis_y = pp.compute_seismograms(
        model,
        source_y,
        stations,
        nt,
        dt,
        alpha,
        pad_frac=pad_frac,
        derivatives=None,
        source_time_function=source_time_function,
        xyz=True,
    )

    fd = (seis_y - seis0)/epsilon # finite difference estimate
    max_y = abs(drv[:,derivs.i_y,:,:]).max(-1).reshape(stations.nstations,3,1) # Maximum absolute value of trace
    perc_err_y = 100*(abs(drv[:,derivs.i_y,:,:]-fd)/max_y) #For each trace, calculate finite difference 'error' as percentage of trace amplitude
    print("       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"%perc_err_y.max())
    

    print("    c. Perturbing source in z.")
    source_z = source.copy()
    source_z.dep -= epsilon # coordinate system is z-up so a positive epsilon in z is a *reduction* in source depth
    
    tt, seis_z = pp.compute_seismograms(
        model,
        source_z,
        stations,
        nt,
        dt,
        alpha,
        pad_frac=pad_frac,
        derivatives=None,
        source_time_function=source_time_function,
        xyz=True,
    )

    fd = (seis_z - seis0)/epsilon # finite difference estimate
    max_z = abs(drv[:,derivs.i_z,:,:]).max(-1).reshape(stations.nstations,3,1) # Maximum absolute value of trace
    perc_err_z = 100*(abs(drv[:,derivs.i_z,:,:]-fd)/max_z) #For each trace, calculate finite difference 'error' as percentage of trace amplitude
    print("       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"%perc_err_z.max())

    print(" 4. Computing static displacement field")
    stat0, drv = pp.compute_static(
        model,
        source,
        stations,
        derivatives=derivs,
    )
    
    print(" 5. Comparing with finite-difference derivatives")
    print("    a. Perturbing source in x.")
   
    stat_x = pp.compute_static(
        model,
        source_x,
        stations,
    )
    fd = (stat_x - stat0)/epsilon
    max_x  = abs(drv[:,derivs.i_x,:]).max(0).reshape(1,3)
    perc_err_x = 100*abs(drv[:,derivs.i_x,:]-fd)/max_x
    print("       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"%(perc_err_x.max()))
    print("    b. Perturbing source in y.")
   
    stat_y = pp.compute_static(
        model,
        source_y,
        stations,
    )
    fd = (stat_y - stat0)/epsilon
    max_y  = abs(drv[:,derivs.i_y,:]).max(0).reshape(1,3)
    perc_err_y = 100*abs(drv[:,derivs.i_y,:]-fd)/max_y
    print("       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"%(perc_err_y.max()))
    print("    c. Perturbing source in z.")
   
    stat_z = pp.compute_static(
        model,
        source_z,
        stations,
    )
    fd = (stat_z - stat0)/epsilon
    max_z  = abs(drv[:,derivs.i_z,:]).max(0).reshape(1,3)
    perc_err_z = 100*abs(drv[:,derivs.i_z,:]-fd)/max_z
    print("       Worst-case difference between 'true' and finite-difference derivatives: %.3f%%"%(perc_err_z.max()))

if __name__ == "__main__":
    tests()