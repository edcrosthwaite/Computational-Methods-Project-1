from src.ODEmain import ode_bump
from src.ks_fine_tune import ks_rootfinding
from src.ODEmain_iso import ode_iso
from src.passive_pareto import passive_suspension_pareto
from src.skyhook_pareto import skyhook_suspension_pareto
from passive_interpol import  piecewise_interpolation
from src.skyhook_interpol import skyhook_interpolation
from src.interpolation_comparison import interpolation_comparison
from src.rk45_verification import rk45_convergence_test

if __name__ == "__main__":
    ode_bump()
    ode_iso()
    ks_rootfinding()
    passive_suspension_pareto()
    skyhook_suspension_pareto()
    piecewise_interpolation()
    skyhook_interpolation()
    interpolation_comparison()
    rk45_convergence_test()
    

