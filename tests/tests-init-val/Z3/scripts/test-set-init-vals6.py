from z3 import *
import time
import numpy as np

# Constants
NUM_COMPONENTS = 17
NUM_VMS = 15

TIMEOUT = 1800  # 30 minutes

# Fixed component requirements
comp_cpu =     [6, 2, 2, 3, 5, 3, 6, 4, 3, 1, 6, 6, 2, 4, 2, 4, 2]
comp_ram =     [20, 26, 32, 32, 13, 25, 15, 21, 7, 23, 11, 28, 6, 32, 15, 22, 17]
comp_storage = [98, 38, 182, 113, 70, 172, 177, 151, 182, 45, 91, 64, 98, 148, 66, 132, 78]

# Fixed hardware specifications for each VM
vm_cpu =     [5, 6, 12, 14, 14, 19, 19, 8, 12, 12, 11, 19, 16, 15, 14]
vm_ram =     [102, 22, 88, 82, 37, 80, 69, 76, 30, 122, 96, 38, 84, 75, 60]
vm_storage = [658, 716, 961, 586, 755, 661, 549, 502, 251, 782, 664, 390, 990, 731, 573]
vm_cost =    [28, 15, 16, 26, 12, 25, 18, 10, 14, 19, 18, 13, 17, 20, 24]

initial_assign = [
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
]
## Build Z3 model
x = [[Int(f"x_{i}_{j}") for j in range(NUM_VMS)] for i in range(NUM_COMPONENTS)]
y = [Int(f"y_{j}") for j in range(NUM_VMS)]
set_option(verbose=10)
opt = Optimize()


# Constraints
for i in range(NUM_COMPONENTS):
    print("i=", i)
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

# VM 5; 14, 37, 755, 12
opt.set_initial_value(vm_cpu[4], 14)
opt.set_initial_value(vm_ram[4], 37)
opt.set_initial_value(vm_storage[4], 755)
opt.set_initial_value(vm_cost[4], 12)
opt.set_initial_value(y[4], 1)
# VM 8; 8, 76, 502, 10
opt.set_initial_value(vm_cpu[7], 8)
opt.set_initial_value(vm_ram[7], 76)
opt.set_initial_value(vm_storage[7], 502)
opt.set_initial_value(vm_cost[7], 10)
opt.set_initial_value(y[7], 1)
# VM 10; 12, 122, 782, 19
opt.set_initial_value(vm_cpu[9], 12)
opt.set_initial_value(vm_ram[9], 122)
opt.set_initial_value(vm_storage[9], 782)
opt.set_initial_value(vm_cost[9], 19)
opt.set_initial_value(y[9], 1)
# VM 12; 19, 38, 390, 13
opt.set_initial_value(vm_cpu[11], 19)
opt.set_initial_value(vm_ram[11], 38)
opt.set_initial_value(vm_storage[11], 390)
opt.set_initial_value(vm_cost[11], 13)
opt.set_initial_value(y[11], 1)
# VM 13; 16, 84, 990, 17
opt.set_initial_value(vm_cpu[12], 16)
opt.set_initial_value(vm_ram[12], 84)
opt.set_initial_value(vm_storage[12], 990)
opt.set_initial_value(vm_cost[12], 17)
opt.set_initial_value(y[12], 1)

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

