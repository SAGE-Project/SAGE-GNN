from z3 import *
import time
import numpy as np

set_option(verbose=10)
set_param("opt.elim_01", False)            # perhaps not necessary, but keeps it simpler wrt initialization
set_param("opt.dump_models", True)  # dump best current solution so far
set_param("smt.elim_term_ite", False)  # avoids creating new variables that can obscure initial value setting.
opt = Optimize()
# Constants
NUM_COMPONENTS = 16
NUM_VMS = 19

TIMEOUT = 1800  # 30 minutes

# Fixed component requirements
comp_cpu =     [4, 2, 6, 3, 6, 3, 3, 4, 5, 4, 6, 1, 1, 5, 5, 5]
comp_ram =     [9, 31, 20, 11, 13, 27, 6, 22, 19, 15, 26, 19, 19, 15, 16, 20]
comp_storage = [93, 144, 98, 149, 41, 188, 79, 154, 78, 119, 198, 111, 75, 112, 149, 31]

# Fixed hardware specifications for each VM
vm_cpu =     [10, 8, 12, 12, 19, 12, 15, 13, 13, 13, 11, 11, 5, 18, 11, 10, 5, 9, 10]
vm_ram =     [93, 33, 75, 52, 73, 123, 97, 102, 93, 64, 46, 79, 62, 47, 22, 111, 55, 23, 119]
vm_storage = [811, 824, 687, 470, 281, 588, 499, 680, 556, 262, 937, 356, 421, 541, 287, 650, 316, 385, 265]
vm_cost =    [30, 16, 25, 26, 13, 19, 26, 20, 28, 30, 28, 25, 27, 15, 10, 10, 21, 22, 24]

initial_assign = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

## Build Z3 model
x = [[Int(f"x_{i}_{j}") for j in range(NUM_VMS)] for i in range(NUM_COMPONENTS)]
y = [Int(f"y_{j}") for j in range(NUM_VMS)]

# Constraints
for i in range(NUM_COMPONENTS):
    for j in range(NUM_VMS):
        opt.add(x[i][j] >= 0, x[i][j] <= 1)
for j in range(NUM_VMS):
    opt.add(y[j] >= 0, y[j] <= 1)
for i in range(NUM_COMPONENTS):
    opt.add(Sum(x[i]) == 1)
for j in range(NUM_VMS):
    for i in range(NUM_COMPONENTS):
        opt.add(y[j] >= x[i][j])
    opt.add(Sum([x[i][j] * comp_cpu[i] for i in range(NUM_COMPONENTS)]) <= vm_cpu[j] * y[j])
    opt.add(Sum([x[i][j] * comp_ram[i] for i in range(NUM_COMPONENTS)]) <= vm_ram[j] * y[j])
    opt.add(Sum([x[i][j] * comp_storage[i] for i in range(NUM_COMPONENTS)]) <= vm_storage[j] * y[j])

total_cost = Sum([y[j] * vm_cost[j] for j in range(NUM_VMS)])
opt.minimize(total_cost)

start_time = time.time()
opt.set("timeout", TIMEOUT * 1000)
result = opt.check()
elapsed = time.time() - start_time

# Set initial values as hints to guide the solver
for i in range(NUM_COMPONENTS):
    for j in range(NUM_VMS):
        print(x[i][j], "=", initial_assign[i][j])
        opt.set_initial_value(x[i][j], initial_assign[i][j])

vm_cpu     = [Int(f"vm_cpu_{j}") for j in range(NUM_VMS)]
vm_ram     = [Int(f"vm_ram_{j}") for j in range(NUM_VMS)]
vm_storage = [Int(f"vm_storage_{j}") for j in range(NUM_VMS)]
vm_cost    = [Int(f"vm_cost_{j}") for j in range(NUM_VMS)]

# # VM 5; 19, 73, 281, 13
opt.set_initial_value(vm_cpu[4], 19)
opt.set_initial_value(vm_ram[4], 73)
opt.set_initial_value(vm_storage[4], 281)
opt.set_initial_value(vm_cost[4], 13)
opt.set_initial_value(y[4], 1)

# # VM 6; 12, 123, 588, 19
opt.set_initial_value(vm_cpu[5], 12)
opt.set_initial_value(vm_ram[5], 123)
opt.set_initial_value(vm_storage[5], 588)
opt.set_initial_value(vm_cost[5], 19)
opt.set_initial_value(y[5], 1)
#
# # VM 14; 18, 47, 541, 15
opt.set_initial_value(vm_cpu[13], 18)
opt.set_initial_value(vm_ram[13], 47)
opt.set_initial_value(vm_storage[13], 541)
opt.set_initial_value(vm_cost[13], 15)
opt.set_initial_value(y[13], 1)
#
# #VM 15; 11, 22, 287, 10
opt.set_initial_value(vm_cpu[14], 11)
opt.set_initial_value(vm_ram[14], 22)
opt.set_initial_value(vm_storage[14], 287)
opt.set_initial_value(vm_cost[14], 10)
opt.set_initial_value(y[14], 1)
#
# #VM16; 10, 111, 650, 10
opt.set_initial_value(vm_cpu[15], 10)
opt.set_initial_value(vm_ram[15], 111)
opt.set_initial_value(vm_storage[15], 650)
opt.set_initial_value(vm_cost[15], 10)
opt.set_initial_value(y[15], 1)

print("set-init-values")
# Solve and time
start_time = time.time()
result = opt.check()
elapsed = time.time() - start_time

print(f"\nZ3 check() result: {result}")
print(f"⏱️ Time taken: {elapsed:.2f} seconds")

# Output
if result == sat:
    model = opt.model()
    assignment_matrix = np.zeros((NUM_COMPONENTS, NUM_VMS), dtype=int)

    for i in range(NUM_COMPONENTS):
        for j in range(NUM_VMS):
            if model.eval(x[i][j]).as_long() == 1:
                assignment_matrix[i][j] = 1

    # Print assignment matrix
    print("\n📋 Component-to-VM assignment matrix:")
    for row in assignment_matrix:
        print(" ".join(map(str, row)))

    print(f"\n💰 Total cost of active VMs: {model.eval(total_cost)}")

    # Print specs only for used VMs
    print("\n🖥️ Used VM Hardware Specifications:")
    print(" VM | CPU | RAM | Storage | Cost")
    print("-" * 34)
    for j in range(NUM_VMS):
        if model.eval(y[j]).as_long() == 1:
            print(f"{j:>3} | {model.eval(vm_cpu[j], model_completion=True).as_long():>3} | "
                  f"{model.eval(vm_ram[j], model_completion=True).as_long():>4} | "
                  f"{model.eval(vm_storage[j], model_completion=True).as_long():>7} | "
                  f"{model.eval(vm_cost[j], model_completion=True).as_long():>4}")
else:
    print("UNSAT")

