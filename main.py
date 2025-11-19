from src.ODEmain import ode_bump
from src.ks_fine_tune import ks_rootfinding
from src.ODEmain_iso import ode_iso
from src.passive_pareto import passive_suspension_pareto
from src.skyhook_pareto import skyhook_suspension_pareto
from src.piecewise_interpol import  piecewise_interpolation
from src.skyhook_interpol import skyhook_interpolation

if __name__ == "__main__":
    ode_bump()
    ode_iso()
    ks_rootfinding()
    passive_suspension_pareto()
    skyhook_suspension_pareto()
    piecewise_interpolation()
    skyhook_interpolation()

