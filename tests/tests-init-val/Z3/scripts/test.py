import random
import time
import numpy as np
import multiprocessing
from z3 import *

NUM_TESTS = 15
TIMEOUT = 1800  # 30 minutes

def run_test(test_id):
    """Run a single test case and write the output."""
    NUM_COMPONENTS = random.randint(15, 20)
    NUM_VMS = random.randint(15, 20)

    # VM specifications
    vm_cpu = [random.randint(5, 20) for _ in range(NUM_VMS)]
    vm_ram = [random.randint(20, 128) for _ in range(NUM_VMS)]
    vm_storage = [random.randint(200, 1000) for _ in range(NUM_VMS)]
    vm_cost = [random.randint(10, 30) for _ in range(NUM_VMS)]

    # Component requirements
    comp_cpu = [random.randint(1, 6) for _ in range(NUM_COMPONENTS)]
    comp_ram = [random.randint(4, 32) for _ in range(NUM_COMPONENTS)]
    comp_storage = [random.randint(20, 200) for _ in range(NUM_COMPONENTS)]

    # Save input
    input_file = f"inputs/input_test_{test_id}.txt"
    with open(input_file, "w") as f:
        f.write(f"{NUM_COMPONENTS}\n")
        f.write(f"{NUM_VMS}\n")
        f.write(f"{comp_cpu}\n")
        f.write(f"{comp_ram}\n")
        f.write(f"{comp_storage}\n")
        f.write(f"{vm_cpu}\n")
        f.write(f"{vm_ram}\n")
        f.write(f"{vm_storage}\n")
        f.write(f"{vm_cost}\n")

    # Build Z3 model
    x = [[Int(f"x_{i}_{j}") for j in range(NUM_VMS)] for i in range(NUM_COMPONENTS)]
    y = [Int(f"y_{j}") for j in range(NUM_VMS)]
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

    # Write output
    output_file = f"outputs/output_test_{test_id}.txt"
    with open(output_file, "w") as f:
        f.write(f"{NUM_COMPONENTS}\n")
        f.write(f"{NUM_VMS}\n")
        f.write(f"{vm_cpu}\n")
        f.write(f"{vm_ram}\n")
        f.write(f"{vm_storage}\n")
        f.write(f"{vm_cost}\n")
        f.write(f"{comp_cpu}\n")
        f.write(f"{comp_ram}\n")
        f.write(f"{comp_storage}\n\n")

        if result == unknown:
            f.write("Timeout\n")
            f.write(f"{elapsed:.2f}\n")
            return

        f.write(f"{result}\n")
        f.write(f"{elapsed:.2f}\n")

        if result == sat:
            model = opt.model()
            assignment_matrix = np.zeros((NUM_COMPONENTS, NUM_VMS), dtype=int)
            for i in range(NUM_COMPONENTS):
                for j in range(NUM_VMS):
                    if model.eval(x[i][j]).as_long() == 1:
                        assignment_matrix[i][j] = 1

            total_cost_value = model.eval(total_cost)
            f.write(f"{total_cost_value}\n")

            for row in assignment_matrix:
                f.write(" ".join(map(str, row)) + "\n")

            for j in range(NUM_VMS):
                if model.eval(y[j]).as_long() == 1:
                    f.write(
                        f"{vm_cpu[j]}, {vm_ram[j]}, {vm_storage[j]}, {vm_cost[j]}\n"
                    )
        else:
            f.write("No solution found\n")


if __name__ == "__main__":
    os.makedirs("../inputs", exist_ok=True)
    os.makedirs("../outputs", exist_ok=True)

    for test_id in range(1, NUM_TESTS + 1):
        process = multiprocessing.Process(target=run_test, args=(test_id,))
        process.start()
        process.join(TIMEOUT)

        if process.is_alive():
            print(f"Test {test_id} exceeded {TIMEOUT}s, force killing...")
            process.terminate()
            process.join()
            with open(f"outputs/output_test_{test_id}.txt", "w") as f:
                f.write("Timeout - Process hard killed due to exceeding 30 minutes\n")



