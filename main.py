from src.ODEnew import ode
from src.ks_fine_tune import ks_rootfinding
from src.passive_pareto import passive_suspension_pareto
from src.skyhook_pareto import skyhook_suspension_pareto
from src.Peak_Acc_Interol import inter

if __name__ == "__main__":
    inter()
    ode(test_type="bump")
    ode(test_type="iso")
    ks_rootfinding()
    passive_suspension_pareto()
    skyhook_suspension_pareto()
