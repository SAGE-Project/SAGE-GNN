from z3 import *
import time
import numpy as np

# Constants
NUM_COMPONENTS = 17
NUM_VMS = 17

TIMEOUT = 1800  # 30 minutes

# Fixed component requirements
comp_cpu =     [2, 2, 4, 4, 2, 6, 2, 3, 4, 4, 3, 4, 1, 3, 4, 3, 3]
comp_ram =     [23, 22, 28, 19, 12, 12, 6, 6, 12, 6, 17, 23, 7, 9, 22, 4, 18]
comp_storage = [125, 111, 43, 140, 28, 103, 141, 89, 133, 21, 146, 77, 176, 40, 178, 175, 167]

# Fixed hardware specifications for each VM
vm_cpu =     [13, 18, 8, 20, 17, 12, 16, 7, 15, 19, 9, 10, 19, 20, 18, 10, 18]
vm_ram =     [55, 112, 117, 82, 32, 48, 80, 72, 81, 99, 114, 51, 26, 125, 119, 103, 62]
vm_storage = [910, 402, 576, 719, 443, 227, 730, 242, 454, 680, 596, 485, 722, 339, 389, 554, 991]
vm_cost =    [25, 19, 20, 20, 10, 22, 25, 15, 29, 16, 19, 13, 14, 18, 25, 26, 18]

initial_assign = [
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
]

## Build Z3 model
x = [[Int(f"x_{i}_{j}") for j in range(NUM_VMS)] for i in range(NUM_COMPONENTS)]
y = [Int(f"y_{j}") for j in range(NUM_VMS)]
set_option(verbose=10)
set_param("opt.elim_01", False)            # perhaps not necessary, but keeps it simpler wrt initialization
set_param("opt.dump_models", True)  # dump best current solution so far
set_param("smt.elim_term_ite", False)  # avoids creating new variables that can obscure initial value setting.
opt = Optimize()

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
        opt.set_initial_value(x[i][j], initial_assign[i][j])

vm_cpu     = [Int(f"vm_cpu_{j}") for j in range(NUM_VMS)]
vm_ram     = [Int(f"vm_ram_{j}") for j in range(NUM_VMS)]
vm_storage = [Int(f"vm_storage_{j}") for j in range(NUM_VMS)]
vm_cost    = [Int(f"vm_cost_{j}") for j in range(NUM_VMS)]

# # VM 10; 19, 99, 680, 16
opt.set_initial_value(vm_cpu[9], 19)
opt.set_initial_value(vm_ram[9], 99)
opt.set_initial_value(vm_storage[9], 680)
opt.set_initial_value(vm_cost[9], 16)
opt.set_initial_value(y[9], 1)
# # VM 14; 20, 125, 339, 18
opt.set_initial_value(vm_cpu[13], 20)
opt.set_initial_value(vm_ram[13], 125)
opt.set_initial_value(vm_storage[13], 339)
opt.set_initial_value(vm_cost[13], 18)
opt.set_initial_value(y[13], 1)
# VM 17; 18, 62, 991, 18
opt.set_initial_value(vm_cpu[16], 18)
opt.set_initial_value(vm_ram[16], 62)
opt.set_initial_value(vm_storage[16], 991)
opt.set_initial_value(vm_cost[16], 18)
opt.set_initial_value(y[16], 1)

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

