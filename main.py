from src.ode_solver import ode
from src.root_finding import ks_rootfinding
from src.pareto_passive import passive_suspension_pareto
from src.pareto_skyhook import skyhook_suspension_pareto
from src.interpolation_velocity_acceleration import inter
from src.interpolation_passive_damper import piecewise_interpolation
from src.interpolation_skyhook_damper import skyhook_interpolation  

if __name__ == "__main__":
    ks_rootfinding()
    ode(test_type="bump")
    ode(test_type="iso")
    inter()
    piecewise_interpolation()
    skyhook_interpolation()
    passive_suspension_pareto()
    skyhook_suspension_pareto()