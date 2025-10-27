from z3 import *
import time
import numpy as np

# Constants
NUM_COMPONENTS = 15
NUM_VMS = 15

TIMEOUT = 1800  # 30 minutes

# Fixed component requirements
comp_cpu =     [4, 4, 4, 5, 2, 6, 4, 4, 5, 3, 4, 3, 5, 5, 3]
comp_ram =     [28, 26, 15, 4, 26, 20, 18, 32, 19, 32, 6, 11, 8, 32, 22]
comp_storage = [80, 27, 176, 81, 44, 29, 33, 108, 46, 157, 113, 105, 85, 143, 78]


# Fixed hardware specifications for each VM
vm_cpu =     [17, 5, 14, 10, 6, 17, 10, 15, 19, 12, 9, 11, 20, 10, 6]
vm_ram =     [87, 105, 23, 118, 34, 117, 43, 98, 25, 71, 48, 86, 26, 27, 111]
vm_storage = [408, 394, 557, 918, 825, 946, 625, 339, 453, 987, 911, 470, 327, 765, 576]
vm_cost =    [20, 10, 25, 17, 26, 10, 24, 29, 12, 20, 30, 26, 18, 27, 26]


initial_assign = [
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
]

## Build Z3 model
x = [[Int(f"x_{i}_{j}") for j in range(NUM_VMS)] for i in range(NUM_COMPONENTS)]
y = [Int(f"y_{j}") for j in range(NUM_VMS)]
set_option(verbose=10)
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

# VM 1; 17, 87, 408, 20
opt.set_initial_value(vm_cpu[0], 17)
opt.set_initial_value(vm_ram[0], 87)
opt.set_initial_value(vm_storage[0], 408)
opt.set_initial_value(vm_cost[0], 20)
opt.set_initial_value(y[0], 1)
# VM 2; 5, 105, 394, 10
opt.set_initial_value(vm_cpu[1], 5)
opt.set_initial_value(vm_ram[1], 105)
opt.set_initial_value(vm_storage[1], 394)
opt.set_initial_value(vm_cost[1], 10)
opt.set_initial_value(y[1], 1)
# VM 4; 10, 118, 918, 17
opt.set_initial_value(vm_cpu[3], 10)
opt.set_initial_value(vm_ram[3], 118)
opt.set_initial_value(vm_storage[3], 918)
opt.set_initial_value(vm_cost[3], 17)
opt.set_initial_value(y[3], 1)
# VM 6; 17, 117, 946, 10
opt.set_initial_value(vm_cpu[5], 17)
opt.set_initial_value(vm_ram[5], 117)
opt.set_initial_value(vm_storage[5], 946)
opt.set_initial_value(vm_cost[5], 10)
opt.set_initial_value(y[5], 1)
# VM 9; 19, 25, 453, 12
opt.set_initial_value(vm_cpu[8], 19)
opt.set_initial_value(vm_ram[8], 25)
opt.set_initial_value(vm_storage[8], 453)
opt.set_initial_value(vm_cost[8], 12)
opt.set_initial_value(y[8], 1)

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

