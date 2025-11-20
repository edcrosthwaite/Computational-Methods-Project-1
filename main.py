from src.ode_solver import ode
from src.root_finding import ks_rootfinding
from src.pareto_passive import passive_suspension_pareto
from src.pareto_skyhook import skyhook_suspension_pareto
from src.interpolation_velocity_acceleration import inter

if __name__ == "__main__":
    inter()
    ode(test_type="bump")
    ode(test_type="iso")
    ks_rootfinding()
    passive_suspension_pareto()
    skyhook_suspension_pareto()
