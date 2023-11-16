import numpy as np
from numpy import random

def generate_values_with_probability_distribution(v):
    k = len(v)
    probabilities = [(k + 1 - i) / sum(k + 1 - j for j in range(1, k + 1)) for i in range(1, k + 1)]

    print("probabilities ", probabilities)
    # n - # of experiments
    selected_indices = random.multinomial(n=k, pvals=probabilities)

    #print("selected_indices ", selected_indices)

    # Pick values from v using the selected indices
    selected_values = [v[i] for i in selected_indices]

    return selected_values

def main():
    # assume the lowest value from the list is the minimum required. The values generated should be >= that this value and close to it.
    # possible CPU values
    CPULst = [1, 2, 4, 6, 8, 16, 17, 32, 64]
    # In other words, each entry ``out[i, j, ..., :]`` is an N - dimensional value drawn from the distribution.
    selected_values_CPU = generate_values_with_probability_distribution(CPULst)
    print(f"Generated CPU values with probability distribution: {selected_values_CPU}")
    # possible Mem values
    MemLst = sorted([3750, 15250, 30500, 7500, 7000, 15000, 68400, 61000, 117000, 30000, 128000, 60500, 60000, 244000,1952, 256000,488000, 976000])
    print("MemLst ", MemLst)
    selected_values_Mem = generate_values_with_probability_distribution(MemLst)
    print(f"Generated Mem values with probability distribution: {selected_values_Mem}")
    # possible Sto values
    StoLst = sorted([1000, 8000, 2000, 4000, 12000, 24000, 6000])
    print("StoLst ", StoLst)
    selected_values_Sto = generate_values_with_probability_distribution(StoLst)
    print(f"Generated Sto values with probability distribution: {selected_values_Sto}")

if __name__ == "__main__":
    main()
