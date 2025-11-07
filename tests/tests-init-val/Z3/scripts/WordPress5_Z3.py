# pip install z3-solver
from z3 import *

WORDPRESS_SMT_PATH = r"/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Output/SMT-LIB/Wordpress5/WordPress5_off_250"

# ------------------------------
# Parsing helpers
# ------------------------------
import re

def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _parse_num_vms(s):
    idxs = [int(m.group(1)) for m in re.finditer(r"\(declare-fun C1_VM(\d+) \(\) Int\)", s)]
    return max(idxs) if idxs else 0

def _parse_type_mappings(s):
    # (assert (=> (= VMxType t) (and (= PriceProvx p) (= ProcProvx pr) (= MemProvx me) (= StorageProvx st))))
    tbl = {}
    pat = re.compile(
        r"\(assert \(=> \(= VM\d+Type (\d+)\)\s*\(and \(= PriceProv\d+ (\d+)\)\s*\(= ProcProv\d+ (\d+)\)\s*\(= MemProv\d+ (\d+)\)\s*\(= StorageProv\d+ (\d+)\)\)\)\)"
    )
    for m in pat.finditer(s):
        t, price, proc, mem, store = map(int, m.groups())
        tbl[t] = (price, proc, mem, store)
    return tbl

def _extract_last_block_before(s, token):
    end = s.find(token)
    if end == -1:
        return ""
    start = s.rfind("(assert (<= (+ 0", 0, end)
    return s[start:end]

def _parse_component_requirements(s, vm_index=1):
    # Proc
    proc_block = _extract_last_block_before(s, f"ProcProv{vm_index}))")
    req_proc = {int(c): int(val) for c, val in re.findall(r"\(\* C(\d)_VM{vm} (\d+)\)".format(vm=vm_index), proc_block)}
    # Mem
    mem_block = _extract_last_block_before(s, f"MemProv{vm_index}))")
    req_mem = {int(c): int(val) for c, val in re.findall(r"\(\* C(\d)_VM{vm} (\d+)\)".format(vm=vm_index), mem_block)}
    # Storage
    sto_block = _extract_last_block_before(s, f"StorageProv{vm_index}))")
    req_store = {int(c): int(val) for c, val in re.findall(r"\(\* C(\d)_VM{vm} (\d+)\)".format(vm=vm_index), sto_block)}
    return req_proc, req_mem, req_store

def _parse_globals_and_pairs(s, N):
    # Σ Cc >= k
    glob_mins = {}
    for c in range(1, 6):
        pat = r"\(assert \(>= \(\+ 0\s*" + "".join([rf"C{c}_VM{i}\s*" for i in range(1, N+1)]) + r"\)\s*(\d+)\)\)"
        m = re.search(pat, s)
        if m:
            glob_mins[c] = int(m.group(1))
    # Combined Σ C4 + Σ C3 >= k
    comb_k = None
    pat = r"\(assert \(>= \(\+ 0\s*" + "".join([rf"C4_VM{i}\s*" for i in range(1, N+1)]) + r"0\s*" + "".join([rf"C3_VM{i}\s*" for i in range(1, N+1)]) + r"\)\s*(\d+)\)\)"
    m = re.search(pat, s)
    if m:
        comb_k = int(m.group(1))
    # Pairwise (<= 1) — detect for VM1 and extend to all VMs
    pairs = set()
    for a in range(1, 6):
        for b in range(1, 6):
            if a == b: 
                continue
            if re.search(rf"\(assert \(<= \(\+ 0 C{a}_VM1 C{b}_VM1\) 1\)\)", s):
                pairs.add((a, b))
    return glob_mins, comb_k, sorted(pairs)

# ------------------------------
# Problem parameters (parsed)
# ------------------------------
_WP = _read_text(WORDPRESS_SMT_PATH)
NUM_VMS = _parse_num_vms(_WP)
TYPE_TABLE = _parse_type_mappings(_WP)         # dict: t -> (price, proc, mem, storage)
REQ_PROC, REQ_MEM, REQ_STORAGE = _parse_component_requirements(_WP, 1)
GLOBAL_MINS, COMBINED_MIN, PAIRS = _parse_globals_and_pairs(_WP, NUM_VMS)

VMS  = range(1, NUM_VMS+1)   # 1..12 for WordPress5
COMP = range(1, 6)           # 1..5 components

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

    # 2) Pairwise <= 1 constraints derived from SMT
    for (a, b) in PAIRS:
        for v in VMS:
            opt.add(C[a][v] + C[b][v] <= 1)

    # 3) Global cardinalities
    for c, k in GLOBAL_MINS.items():
        opt.add(Sum(*[C[c][v] for v in VMS]) >= k)

    # 4) Combined cardinality (ΣC4 + ΣC3 >= k), if present
    if COMBINED_MIN is not None:
        sumC3 = Sum(*[C[3][v] for v in VMS])
        sumC4 = Sum(*[C[4][v] for v in VMS])
        opt.add(sumC3 + sumC4 >= COMBINED_MIN)

    # 5) Price zero if VM unused
    for v in VMS:
        total = Sum(*[C[c][v] for c in COMP])
        opt.add(Implies(total == 0, Price[v] == 0))

    # 6) VMType domain: 0 plus all parsed types
    allowed_types = [0] + sorted(TYPE_TABLE.keys())
    for v in VMS:
        opt.add(Or(*[VMType[v] == t for t in allowed_types]))

    # 7) Type mapping (only for non-zero types)
    for v in VMS:
        for t, (p, pr, me, st) in TYPE_TABLE.items():
            opt.add(Implies(VMType[v] == t, And(
                Price[v] == p,
                Proc[v]  == pr,
                Mem[v]   == me,
                Store[v] == st
            )))

    # 8) Capacity constraints (per VM) — use parsed per-component requirements
    for v in VMS:
        opt.add(Sum(*[REQ_PROC[c]    * C[c][v] for c in COMP])    <= Proc[v])
        opt.add(Sum(*[REQ_MEM[c]     * C[c][v] for c in COMP])    <= Mem[v])
        opt.add(Sum(*[REQ_STORAGE[c] * C[c][v] for c in COMP])    <= Store[v])

    # 9) Objective
    total_price = Sum(*[Price[v] for v in VMS])
    opt.minimize(total_price)
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
    smt = None
    try:
        smt = opt.to_smt2()
    except Exception:
        smt = opt.sexpr()
    trailer = "\n(check-sat)\n(get-model)\n(get-objectives)\n(exit)\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(smt)
        f.write(trailer)

# ------------------------------
# Run
# ------------------------------
def main():
    try:
        import z3
        print(f"Z3 version: {z3.get_version_string()}")
    except Exception:
        pass

    opt, C, VMType, Price, Proc, Mem, Store, total_price = build_optimizer()

    # You can solve here or only emit SMT-LIB
    if opt.check() == sat:
        m = opt.model()
        print("SAT. Objective (total price) =", m.eval(total_price))
        for v in VMS:
            print(f"VM {v}: Type =", m.eval(VMType[v]), " Price=", m.eval(Price[v]))
            print("  Provided Proc/Mem/Store:", m.eval(Proc[v]), m.eval(Mem[v]), m.eval(Store[v]))
            print("  C[1..5] :", [m.eval(C[c][v]) for c in COMP])
        try:
            out_path = "WordPress5_reconstructed.smt2"
            write_smt2(opt, out_path)
            print("SMT-LIB written to", out_path)
        except Exception as e:
            print("SMT-LIB export failed:", e)
    else:
        print("UNSAT or unknown.")

if __name__ == "__main__":
    main()
