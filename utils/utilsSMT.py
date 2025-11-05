import re
from typing import Dict, List, Tuple

def parse_smt_file(path: str) -> Tuple[List[str], List[str], List[List[int]], Dict[str, Dict[str, int]]]:
    """
    Parse an SMT-LIB style 'sat (...)' file to extract:
      - components:  ['C1', 'C2', ..., 'Ck']
      - vms:         ['V1', 'V2', ..., 'Vm']
      - matrix:      k×m list of ints; matrix[i][j] is value for Ci, V(j+1)
      - specs:       {'V1': {'CPU':..., 'Mem':..., 'Sto':..., 'Price':...}, ...}

    Accepted bindings (any order, any spacing/newlines):
      (define-fun C<ci>_VM<vj> () Int <0|1>)
      (define-fun ProcProv<j>   () Int <int>)
      (define-fun MemProv<j>    () Int <int>)
      (define-fun StorageProv<j>() Int <int>)
      (define-fun PriceProv<j>  () Int <int>)

    Notes:
      - Automatically sizes the matrix to the max component/VM indices seen.
      - Later occurrences overwrite earlier ones (typical SMT model semantics).
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Regexes (dotall to allow newlines between tokens)
    cv_pat     = re.compile(r"\(define-fun\s+C(\d+)_VM(\d+)\s*\(\)\s*Int\s*([\-\d]+)\s*\)", re.S)
    proc_pat   = re.compile(r"\(define-fun\s+ProcProv(\d+)\s*\(\)\s*Int\s*([\-\d]+)\s*\)", re.S)
    mem_pat    = re.compile(r"\(define-fun\s+MemProv(\d+)\s*\(\)\s*Int\s*([\-\d]+)\s*\)", re.S)
    sto_pat    = re.compile(r"\(define-fun\s+StorageProv(\d+)\s*\(\)\s*Int\s*([\-\d]+)\s*\)", re.S)
    price_pat  = re.compile(r"\(define-fun\s+PriceProv(\d+)\s*\(\)\s*Int\s*([\-\d]+)\s*\)", re.S)

    # Collect all (Ci, Vj) => value
    cv_values: Dict[Tuple[int,int], int] = {}
    max_c = max_v = 0
    for c_str, v_str, val_str in cv_pat.findall(text):
        c = int(c_str); v = int(v_str); val = int(val_str)
        cv_values[(c, v)] = val
        if c > max_c: max_c = c
        if v > max_v: max_v = v

    # Build labels
    components = [f"C{i}" for i in range(1, max_c + 1)] if max_c > 0 else []
    vms        = [f"V{j}" for j in range(1, max_v + 1)] if max_v > 0 else []

    # Build matrix (default 0)
    matrix: List[List[int]] = [[0 for _ in range(max_v)] for _ in range(max_c)]
    for (c, v), val in cv_values.items():
        matrix[c - 1][v - 1] = val

    # Extract per-VM specs; default to None if missing, but kept as absent keys
    def _collect(pat) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for idx_str, val_str in pat.findall(text):
            out[int(idx_str)] = int(val_str)
        return out

    procs = _collect(proc_pat)
    mems  = _collect(mem_pat)
    stos  = _collect(sto_pat)
    prices= _collect(price_pat)

    specs: Dict[str, Dict[str, int]] = {}
    for j in range(1, max_v + 1):
        vm_name = f"V{j}"
        # Only include keys that exist; omit if not present in the file
        entry: Dict[str, int] = {}
        if j in procs:  entry["CPU"]   = procs[j]
        if j in mems:   entry["Mem"]   = mems[j]
        if j in stos:   entry["Sto"]   = stos[j]
        if j in prices: entry["Price"] = prices[j]
        specs[vm_name] = entry

    #return components, vms, matrix, specs
    return matrix, specs
