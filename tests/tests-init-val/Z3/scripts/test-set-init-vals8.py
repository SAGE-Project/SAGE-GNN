from z3 import *
import time
import numpy as np

# Constants
NUM_COMPONENTS = 16
NUM_VMS = 19

TIMEOUT = 1800  # 30 minutes

# Fixed component requirements
comp_cpu =     [3, 5, 1, 4, 5, 6, 3, 6, 1, 4, 2, 4, 4, 4, 3, 4]
comp_ram =     [26, 31, 18, 13, 16, 22, 5, 29, 19, 14, 23, 11, 24, 22, 19, 16]
comp_storage = [137, 61, 171, 88, 80, 163, 122, 198, 43, 96, 178, 21, 186, 68, 26, 109]

# Fixed hardware specifications for each VM
vm_cpu =     [6, 20, 8, 6, 11, 19, 11, 9, 15, 20, 6, 5, 5, 14, 5, 17, 7, 9, 11]
vm_ram =     [55, 74, 122, 78, 58, 44, 33, 43, 66, 50, 32, 25, 57, 92, 27, 99, 51, 87, 70]
vm_storage = [834, 288, 545, 621, 858, 670, 838, 431, 780, 837, 892, 909, 831, 858, 531, 590, 829, 262, 557]
vm_cost =    [28, 21, 11, 28, 20, 27, 13, 30, 15, 17, 18, 15, 23, 27, 12, 17, 30, 10, 22]

initial_assign = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
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

# VM 3; 8, 122, 545, 11
opt.set_initial_value(vm_cpu[2], 8)
opt.set_initial_value(vm_ram[2], 122)
opt.set_initial_value(vm_storage[2], 545)
opt.set_initial_value(vm_cost[2], 11)
opt.set_initial_value(y[2], 1)
# VM 7; 11, 33, 838, 13
opt.set_initial_value(vm_cpu[6], 11)
opt.set_initial_value(vm_ram[6], 33)
opt.set_initial_value(vm_storage[6], 838)
opt.set_initial_value(vm_cost[6], 13)
opt.set_initial_value(y[6], 1)
# VM 9; 15, 66, 780, 15
opt.set_initial_value(vm_cpu[8], 15)
opt.set_initial_value(vm_ram[8], 66)
opt.set_initial_value(vm_storage[8], 780)
opt.set_initial_value(vm_cost[8], 15)
opt.set_initial_value(y[8], 1)
# VM 16; 17, 99, 590, 17
opt.set_initial_value(vm_cpu[15], 17)
opt.set_initial_value(vm_ram[15], 99)
opt.set_initial_value(vm_storage[15], 590)
opt.set_initial_value(vm_cost[15], 17)
opt.set_initial_value(y[15], 1)
# VM 18; 9, 87, 262, 10
opt.set_initial_value(vm_cpu[17], 9)
opt.set_initial_value(vm_ram[17], 87)
opt.set_initial_value(vm_storage[17], 262)
opt.set_initial_value(vm_cost[17], 10)
opt.set_initial_value(y[17], 1)


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

