# pip install z3-solver
from z3 import *

# ------------------------------
# Problem parameters
# ------------------------------
VMS  = range(1, 7)   # VM indices: 1..6
COMP = range(1, 6)   # components/classes: 1..5

# Per-component resource requirements (per unit)
REQ_PROC    = {1: 4,    2: 2,    3: 4,     4: 8,     5: 1}
REQ_MEM     = {1: 2048, 2: 512,  3: 2048,  4: 16000, 5: 256}
REQ_STORAGE = {1: 500,  2: 1000, 3: 1000,  4: 2000,  5: 250}

# VM type table (t=0 is "unused", only non-zero types bind capacities/prices)
TYPE_TABLE = {
    1:(8403,64,976000,1000),  2:(9152,64,488000,8000),   3:(10638,64,976000,1000),
    4:(16000,64,1952,1000),   5:(13005,64,976000,1000),  6:(4105,32,244000,2000),
    7:(2752,32,244000,4000),  8:(4576,32,244000,4000),   9:(6672,32,976000,1000),
    10:(5570,17,117000,24000),11:(1373,16,122000,2000),  12:(1430,16,30000,2000),
    13:(5400,17,117000,24000),14:(1654,16,122000,2000),  15:(5520,17,117000,24000),
    16:(3079,16,122000,1000), 17:(1637,16,122000,2000),  18:(1470,8,61000,6000),
    19:(1301,8,68400,2000),   20:(665,8,61000,1000),     21:(632,8,7000,4000),
    22:(1288,8,68400,2000),   23:(402,4,15000,2000),     24:(827,4,30500,3000),
    25:(266,4,15000,2000),    26:(252,4,7500,2000),      27:(252,4,7500,2000),
    28:(809,4,30500,3000),    29:(379,4,30500,1000),     30:(146,2,7500,1000),
    31:(207,2,15250,1000),    32:(293,2,17100,1000),     33:(220,2,15250,1000),
    34:(197,2,1700,1000),     35:(180,2,1700,1000),      36:(275,2,17100,1000),
    37:(128,2,3750,2000),     38:(58,1,1700,1000),       39:(93,1,3750,1000),
    40:(98,1,3750,1000)
}

# ------------------------------
# Build model
# ------------------------------
def build_optimizer():
    opt = Optimize()

    # Variables
    C      = {c: {v: Int(f"C{c}_VM{v}") for v in VMS} for c in COMP}  # binary 0/1
    VMType = {v: Int(f"VM{v}Type") for v in VMS}

    Price  = {v: Int(f"PriceProv{v}")   for v in VMS}
    Proc   = {v: Int(f"ProcProv{v}")    for v in VMS}
    Mem    = {v: Int(f"MemProv{v}")     for v in VMS}
    Store  = {v: Int(f"StorageProv{v}") for v in VMS}

    # 1) Binary domains + linkage to VMType != 0 when selected
    for c in COMP:
        for v in VMS:
            opt.add(Or(C[c][v] == 0, C[c][v] == 1))
            opt.add(Implies(C[c][v] == 1, VMType[v] != 0))

    # Additionally mirror the exact single clause you specified:
    # (assert (=> (= C3_VM5 1) (not (= VM5Type 0))))
    opt.add(Implies(C[3][5] == 1, VMType[5] != 0))

    # 2) Pairwise <= 1 constraints (exactly as in SMT2)
    # Block A: (C1 + Ck) <= 1 for k = 2,3,4,5 (per-VM)
    for v in VMS:
        for k in [2, 3, 4, 5]:
            opt.add(C[1][v] + C[k][v] <= 1)

    # Block B: (C2 + C3) <= 1 (per-VM)
    for v in VMS:
        opt.add(C[2][v] + C[3][v] <= 1)

    # Block C: (C4 + Cx) <= 1 for x = 1,2,3,5 (per-VM)
    for v in VMS:
        for x in [1, 2, 3, 5]:
            opt.add(C[4][v] + C[x][v] <= 1)

    # 3) Cardinalities (global)
    opt.add(Sum(*[C[1][v] for v in VMS]) == 1)
    opt.add(Sum(*[(C[2][v] + C[3][v]) for v in VMS]) >= 3)
    for c in COMP:
        opt.add(Sum(*[C[c][v] for v in VMS]) >= 1)

    # 4) Presence ITE (per VM): C5 + C4 + C1 == ite( sum C1..C5 >= 1, 1, 0 )
    for v in VMS:
        total = Sum(*[C[c][v] for c in COMP])
        opt.add(C[5][v] + C[4][v] + C[1][v] == If(total >= 1, 1, 0))

    # 5) Global balance: 0 < 10*ΣC4 - ΣC5 <= 10
    sumC4 = Sum(*[C[4][v] for v in VMS])
    sumC5 = Sum(*[C[5][v] for v in VMS])
    expr = 10 * sumC4 - sumC5
    opt.add(expr > 0)
    opt.add(expr <= 10)

    # 6) Price zero if VM unused
    for v in VMS:
        total = Sum(*[C[c][v] for c in COMP])
        opt.add(Implies(total == 0, Price[v] == 0))

    # 7) VMType domain 0..40
    for v in VMS:
        opt.add(Or(*[VMType[v] == t for t in range(0, 41)]))

    # 8) Type mapping (t = 1..40)
    for v in VMS:
        for t in range(1, 41):
            p, pr, me, st = TYPE_TABLE[t]
            opt.add(Implies(VMType[v] == t, And(
                Price[v] == p,
                Proc[v]  == pr,
                Mem[v]   == me,
                Store[v] == st
            )))

    # 9) Capacity constraints (per VM)
    for v in VMS:
        opt.add(Sum(*[REQ_PROC[c]    * C[c][v] for c in COMP])    <= Proc[v])
        opt.add(Sum(*[REQ_MEM[c]     * C[c][v] for c in COMP])    <= Mem[v])
        opt.add(Sum(*[REQ_STORAGE[c] * C[c][v] for c in COMP])    <= Store[v])

    # 10) Objective
    total_price = Sum(*[Price[v] for v in VMS])
    opt.minimize(total_price)

    
    # ---- Initial values (parsed from SecureWebContainer_off_20.out) ----
    INITIAL_ASSIGNMENTS = [
        ('StorageProv1', 2000),
        ('C2_VM5', 0),
        ('C1_VM4', 0),
        ('PriceProv2', 402),
        ('PriceProv6', 402),
        ('C5_VM6', 1),
        ('C5_VM4', 0),
        ('ProcProv1', 8),
        ('PriceProv1', 1288),
        ('VM4Type', 0),
        ('ProcProv4', 7),
        ('MemProv1', 68400),
        ('C3_VM4', 0),
        ('StorageProv6', 2000),
        ('C5_VM3', 0),
        ('C3_VM2', 0),
        ('VM6Type', 13),
        ('VM1Type', 12),
        ('C1_VM2', 0),
        ('MemProv3', 68400),
        ('MemProv4', 121999),
        ('ProcProv5', 4),
        ('MemProv2', 15000),
        ('ProcProv6', 4),
        ('C4_VM6', 0),
        ('C4_VM3', 1),
        ('StorageProv3', 2000),
        ('StorageProv4', 1999),
        ('C3_VM3', 0),
        ('C5_VM2', 1),
        ('MemProv5', 30500),
        ('MemProv6', 15000),
        ('C4_VM4', 0),
        ('C1_VM6', 0),
        ('ProcProv3', 8),
        ('C3_VM1', 1),
        ('C4_VM2', 0),
        ('StorageProv2', 2000),
        ('C2_VM3', 0),
        ('PriceProv5', 379),
        ('C1_VM5', 1),
        ('VM5Type', 15),
        ('C3_VM6', 0),
        ('C4_VM5', 0),
        ('StorageProv5', 1000),
        ('VM2Type', 13),
        ('PriceProv4', 0),
        ('VM3Type', 12),
        ('ProcProv2', 4),
        ('C5_VM1', 1),
        ('C4_VM1', 0),
        ('C2_VM6', 1),
        ('C1_VM3', 0),
        ('C2_VM2', 1),
        ('C3_VM5', 0),
        ('C5_VM5', 0),
        ('C2_VM4', 0),
        ('PriceProv3', 1288),
        ('C2_VM1', 0),
        ('C1_VM1', 0)
    ]

    import re as _re  # for resolving symbolic names to Z3 vars
    def _resolve_symbol(name: str):
        # Maps strings like "StorageProv1", "PriceProv6", "ProcProv2", "MemProv3", "VM5Type", "C3_VM2"
        # to the corresponding Z3 variables created above.
        if name.startswith("StorageProv"):
            idx = int(name[len("StorageProv"):])
            return Store[idx]
        if name.startswith("PriceProv"):
            idx = int(name[len("PriceProv"):])
            return Price[idx]
        if name.startswith("ProcProv"):
            idx = int(name[len("ProcProv"):])
            return Proc[idx]
        if name.startswith("MemProv"):
            idx = int(name[len("MemProv"):])
            return Mem[idx]
        if name.startswith("VM") and name.endswith("Type"):
            idx = int(name[2:-4])
            return VMType[idx]
        m = _re.match(r"^C(\d+)_VM(\d+)$", name)
        if m:
            c = int(m.group(1)); v = int(m.group(2))
            return C[c][v]
        raise KeyError(f"Unknown symbol in initial assignments: {name}")

    for _name, _val in INITIAL_ASSIGNMENTS:
        opt.set_initial_value(_resolve_symbol(_name), _val)
    # ---- End initial values ----
    return opt, C, VMType, Price, Proc, Mem, Store, total_price

# ------------------------------
# SMT-LIB2 export (robust to older Z3)
# ------------------------------
def write_smt2(opt: Optimize, path: str):
    """
    Export a standalone SMT-LIB2 file.
    Uses Optimize.to_smt2() when available; falls back to Optimize.sexpr() otherwise.
    Appends (check-sat), (get-model), (get-objectives), (exit).
    """
    if hasattr(opt, "to_smt2"):
        smt = opt.to_smt2()
    else:
        smt = opt.sexpr()  # includes objectives and all constraints

    trailer = "\n(check-sat)\n(get-model)\n(get-objectives)\n(exit)\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(smt)
        f.write(trailer)

# ------------------------------
# Run
# ------------------------------
def main():
    # (Optional) print Z3 version for debugging
    try:
        import z3
        print("Z3 version:", z3.get_version())
    except Exception:
        pass

    opt, C, VMType, Price, Proc, Mem, Store, total_price = build_optimizer()

    # Write SMT-LIB2 file before solving (so it matches the pre-solve model)
    write_smt2(opt, "model.smt2")
    print("SMT-LIB2 written to model.smt2")

    # Solve
    res = opt.check()
    print("SAT result:", res)

    if res == sat:
        m = opt.model()
        print("\n=== Objective value (total price) ===")
        print(m.eval(total_price))

        print("\n=== Model ===")
        for v in VMS:
            print(f"\nVM{v}:")
            print("  VMType     :", m.eval(VMType[v]))
            print("  PriceProv  :", m.eval(Price[v]))
            print("  ProcProv   :", m.eval(Proc[v]))
            print("  MemProv    :", m.eval(Mem[v]))
            print("  StorageProv:", m.eval(Store[v]))
            print("  C[1..5]    :", [m.eval(C[c][v]) for c in COMP])

        print("\n=== Statistics ===")
        print(opt.statistics())
    else:
        print("No solution.")

if __name__ == "__main__":
    main()
