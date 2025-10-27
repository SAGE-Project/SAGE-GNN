from z3 import *
import time
import numpy as np

# Constants
NUM_COMPONENTS = 15
NUM_VMS = 18

TIMEOUT = 1800  # 30 minutes

# Fixed component requirements
comp_cpu =     [6, 3, 2, 2, 3, 2, 1, 6, 6, 5, 4, 1, 2, 5, 6]
comp_ram =     [13, 26, 24, 11, 21, 7, 24, 29, 17, 5, 16, 14, 11, 31, 32]
comp_storage = [198, 176, 163, 42, 151, 45, 107, 22, 20, 160, 30, 66, 104, 27, 102]

# Fixed hardware specifications for each VM
vm_cpu =     [20, 5, 8, 19, 8, 15, 11, 15, 17, 14, 17, 11, 18, 18, 11, 15, 17, 9]
vm_ram =     [57, 121, 69, 75, 26, 66, 100, 127, 93, 104, 109, 48, 50, 111, 55, 122, 83, 35]
vm_storage = [800, 865, 620, 705, 644, 654, 835, 659, 533, 644, 405, 591, 499, 637, 667, 858, 971, 793]
vm_cost =    [20, 25, 19, 24, 24, 28, 24, 13, 17, 25, 16, 14, 23, 17, 11, 23, 29, 29]

initial_assign = [
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
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

# VM 8; 15, 127, 659, 13
opt.set_initial_value(vm_cpu[7], 15)
opt.set_initial_value(vm_ram[7], 127)
opt.set_initial_value(vm_storage[7], 659)
opt.set_initial_value(vm_cost[7], 13)
opt.set_initial_value(y[7], 1)
# VM 11; 17, 109, 405, 16
opt.set_initial_value(vm_cpu[10], 17)
opt.set_initial_value(vm_ram[10], 109)
opt.set_initial_value(vm_storage[10], 405)
opt.set_initial_value(vm_cost[10], 16)
opt.set_initial_value(y[10], 1)
# VM 12; 11, 48, 591, 14
opt.set_initial_value(vm_cpu[11], 11)
opt.set_initial_value(vm_ram[11], 48)
opt.set_initial_value(vm_storage[11], 591)
opt.set_initial_value(vm_cost[11], 14)
opt.set_initial_value(y[11], 1)
# VM 15; 11, 55, 667, 11
opt.set_initial_value(vm_cpu[14], 11)
opt.set_initial_value(vm_ram[14], 55)
opt.set_initial_value(vm_storage[14], 667)
opt.set_initial_value(vm_cost[14], 11)
opt.set_initial_value(y[14], 1)
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

