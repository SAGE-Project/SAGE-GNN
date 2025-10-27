#!/usr/bin/env python3
"""
Structured Z3 Optimize script wrapping the SMT-LIB model in
'WordPress3_b703e050-a7ae-483d-96ab-7a3b7241f02e'.
"""

from z3 import *
from typing import List, Sequence, Dict
import time

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SMT_FILE = "WordPress3_b703e050-a7ae-483d-96ab-7a3b7241f02e"
VERBOSE  = True


# -----------------------------------------------------------------------------
# Initialize Optimize and verbosity
# -----------------------------------------------------------------------------
if VERBOSE:
    set_option(verbose=10)

opt = Optimize()

# -----------------------------------------------------------------------------
# Parse SMT-LIB file (returns list of asserted formulas).
# -----------------------------------------------------------------------------
formulas = parse_smt2_file(SMT_FILE)

# Add all parsed assertions to the optimizer
opt.add(formulas)

# -----------------------------------------------------------------------------
# Helper to fetch an Int variable by *exact* name (declared in the file)
# -----------------------------------------------------------------------------
def V(name: str) -> IntNumRef:
    return Int(name)

NUM_COMPONENTS = 5
NUM_VMS     = 8

# variables C{c+1}_VM{v+1}
ComponentsVMs: List[List[IntNumRef]] = [
    [V(f"C{c}_VM{v}") for v in range(1, NUM_VMS + 1)]
    for c in range(1, NUM_COMPONENTS + 1)
]

VMType      : List[IntNumRef] = [V(f"VM{i}Type")      for i in range(1, NUM_VMS + 1)]
PriceProv   : List[IntNumRef] = [V(f"PriceProv{i}")   for i in range(1, NUM_VMS + 1)]
StorageProv : List[IntNumRef] = [V(f"StorageProv{i}") for i in range(1, NUM_VMS + 1)]
MemProv     : List[IntNumRef] = [V(f"MemProv{i}")     for i in range(1, NUM_VMS + 1)]
ProcProv    : List[IntNumRef] = [V(f"ProcProv{i}")    for i in range(1, NUM_VMS + 1)]

objective = opt.minimize(Sum(PriceProv))


# -----------------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------------
t0 = time.perf_counter()
result = opt.check()
t1 = time.perf_counter()

solve_wall = t1 - t0
print("Result:", result)
print(f"Optimize.check() wall time: {solve_wall:.6f} s")

if result not in (sat, unknown):
    print("No model available (unsat).")
    exit(0)

model = opt.model()
print("\nObjective (sum PriceProv):", model.eval(Sum(PriceProv), model_completion=True))


# -----------------------------------------------------------------------------
# Helper utilities for extracting model data
# -----------------------------------------------------------------------------
def mval(x):
    """Return Python int (or None) for model value x."""
    if x is None:
        return None
    xv = model.eval(x, model_completion=True)
    if isinstance(xv, IntNumRef):
        try:
            return int(str(xv))
        except ValueError:
            return None
    return xv

def values(seq: Sequence) -> List[int]:
    return [mval(s) for s in seq]

def summarize_clients():
    print("\nComponent ↦ VM assignment matrix (C{c}_VM{v}):")
    for c_idx, row in enumerate(ComponentsVMs, start=1):
        row_vals = values(row)
        print(f"  C{c_idx}: {row_vals}")

def summarize_resources():
    print("\nResource Provisioning:")
    print("  PriceProv   :", values(PriceProv))
    print("  StorageProv :", values(StorageProv))
    print("  MemProv     :", values(MemProv))
    print("  ProcProv    :", values(ProcProv))
    print("  VMType      :", values(VMType))


# -----------------------------------------------------------------------------
# Pretty-print selected model information
# -----------------------------------------------------------------------------
if result == sat:
    summarize_clients()
    summarize_resources()

    # (Optional) build a dictionary for programmatic downstream use
    model_dict = {d.name(): mval(d()) for d in model.decls()}

    # Example: print first few entries
    sample_items = list(model_dict.items())[:10]
    print("\nSample of model_dict entries:", sample_items)

elif result == unknown:
    print("Status unknown; reason:", opt.reason_unknown())

