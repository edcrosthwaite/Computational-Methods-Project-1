from src.Root_Finding import ks_rootfinding

def rob_test(per:float = 0.03, initial:float = 1.45):
    upper = initial + (initial*0.05)
    lower = initial - (initial*0.05)
    values = [upper, lower]

    k0 = ks_rootfinding(initial)[initial][0]

    for i in range(2):
        k = ks_rootfinding(values[i])[values[i]][0]
        print(f"Frequency after {per*100} percent change is {k}")
        print(f"Percentage change is {100*(k0-k)/k}")

if __name__ == "__main__":
    rob_test()