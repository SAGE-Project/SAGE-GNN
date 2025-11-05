import json
from Solvers.Core.ProblemDefinition import ManeuverProblem
from src.init import log
import src.smt
import numpy as np
import uuid
from typing import Dict, List

from z3 import *

def add_pred_soft_constraints(solver, prediction):
    # prediction is a matrix of size (nr comp) * (nr vms * nr offers)
    constraints = []
    nrOffers = solver.nrOffers
    nrVms = solver.nrVM
    nrComponents = solver.nrComp
    print("prediction ", len(prediction), " ", len(prediction[0]), " ", prediction)
    for comp_idx in range(nrComponents):
        pred_comp = prediction[comp_idx]
        matrix = np.reshape(pred_comp, (nrOffers, nrVms))
        print("matrix ", matrix)
        for vm_idx in range(solver.nrVM):
            pred_comp_vm = matrix[:, vm_idx]
            placements = [i for i, x
                          in enumerate(pred_comp_vm)
                          if x == 1]
            print("placements ", placements)
            a_matrix_index = comp_idx * solver.nrVM + vm_idx
            if len(placements) != 0:
                constraints.append(solver.a[a_matrix_index] == 1)
            else:
                constraints.append(solver.a[a_matrix_index] == 0)
            for placement in placements:
                vmType = placement #+ 1

                #constraints.append(solver.vmType[vm_idx] == vmType)
                #       solver.MemProv[vm_idx] == solver.offers_list[vmType][2],
                #       solver.StorageProv[vm_idx] == solver.offers_list[vmType][3],
                #       solver.PriceProv[vm_idx] == solver.offers_list[vmType][4])

                constraints.append(solver.ProcProv[vm_idx] == solver.offers_list[vmType][1])
                constraints.append(solver.MemProv[vm_idx] == solver.offers_list[vmType][2])
                constraints.append(solver.StorageProv[vm_idx] == solver.offers_list[vmType][3])
                constraints.append(solver.PriceProv[vm_idx] == solver.offers_list[vmType][4])

    constraints = list(set(constraints))
    # I'm already in the gnn model
    if solver.sb_option != "None":
        print("at most")
        constraints.append(len(constraints))
        solver.solver.add(AtMost(constraints))
    else:
        print("add soft")
        solver.solver.add_soft(constraints)

def add_init_values(solver, matrix_init: List[List[int]], VMSpecs_init: Dict[str, Dict[str, int]]) -> None:
    """
    matrix_init: list[list[int]] with shape (solver.nrComp x solver.nrVM), entries in {0,1}
    VMSpecs_init: {'V1': {'CPU':..,'Mem':..,'Sto':..,'Price':..}, ..., 'Vn': {...}}
    """
    nrVms = solver.nrVM
    nrComponents = solver.nrComp

    # --- Sanity checks ---
    if len(matrix_init) != nrComponents:
        raise ValueError(f"matrix_init rows {len(matrix_init)} != nrComponents {nrComponents}")
    for r, row in enumerate(matrix_init):
        if len(row) != nrVms:
            raise ValueError(f"matrix_init row {r} length {len(row)} != nrVms {nrVms}")

    # Ensure we have exactly V1..VnrVms
    expected_keys = {f"V{i}" for i in range(1, nrVms + 1)}
    got_keys = set(VMSpecs_init.keys())
    if got_keys != expected_keys:
        missing = expected_keys - got_keys
        extra = got_keys - expected_keys
        msg = []
        if missing: msg.append(f"missing {sorted(missing)}")
        if extra:   msg.append(f"extra {sorted(extra)}")
        raise KeyError("VMSpecs_init keys mismatch: " + ", ".join(msg))

    # Validate each VM spec has CPU/Mem/Sto/Price
    for j in range(1, nrVms + 1):
        vmk = f"V{j}"
        spec = VMSpecs_init[vmk]
        for key in ("CPU", "Mem", "Sto", "Price"):
            if key not in spec:
                raise KeyError(f"{vmk} missing '{key}' in VMSpecs_init")

    # --- Initialize VM provision variables ---
    # solver.* arrays are 0-based; VM labels are 1-based
    for j in range(1, nrVms + 1):
        vm0 = j - 1
        spec = VMSpecs_init[f"V{j}"]
        print("--->", int(spec["CPU"]), int(spec["Mem"]), int(spec["Sto"]), int(spec["Price"]))
        solver.solver.set_initial_value(solver.ProcProv[vm0],   int(spec["CPU"]))
        solver.solver.set_initial_value(solver.MemProv[vm0],    int(spec["Mem"]))
        solver.solver.set_initial_value(solver.StorageProv[vm0],int(spec["Sto"]))
        solver.solver.set_initial_value(solver.PriceProv[vm0],  int(spec["Price"]))

    # --- Initialize placement variables a[(component, vm)] from matrix_init ---
    # Flattened indexing convention: comp_idx * nrVms + vm_idx
    for ci in range(nrComponents):          # component index 0..nrComponents-1  (C1..)
        row = matrix_init[ci]
        for vj in range(nrVms):             # vm index 0..nrVms-1 (..Vn)
            a_idx = ci * nrVms + vj
            print("--->", solver.a[a_idx], int(row[vj]))
            solver.solver.set_initial_value(solver.a[a_idx], int(row[vj]))


class Wrapper_Z3:
    def __init__(self, symmetry_breaker="None", solver_id="z3"):
        self.symmetry_breaker = symmetry_breaker
        self.solver_id = solver_id

    def solve(
            self,
            application_model_json,
            offers_json,
            prediction=None,
            prediction_init=None,
            inst=5,
            out=True,
            #out=False,
            mode=None,
            matrix_init=None,
            VMSpecs_init=None
    ):
        SMTsolver = src.smt.getSolver(self.solver_id)
        availableConfigurations = []
        for key, value in offers_json.items():
            specs_list = [
                key,
                value["cpu"],
                value["memory"],
                value["storage"],
                value["price"],
            ]
            availableConfigurations.append(specs_list)

        problem = ManeuverProblem()
        problem.readConfigurationJSON(
            application_model_json, availableConfigurations, inst
        )
        if out:
            SMTsolver.init_problem(problem, "optimize", sb_option=self.symmetry_breaker,
                                   smt2lib=f"../Output/SMT-LIB/" + application_model_json["application"] + "_" + str(uuid.uuid4()),
                                   smt2libsol=f"../Output/SMT-LIB/" + application_model_json["application"] + "_" + str(uuid.uuid4())+".out")
        else:
            SMTsolver.init_problem(problem, "optimize", sb_option=self.symmetry_breaker)
        print("mode ", mode)
        print("matrix_init ", matrix_init)
        print("VMSpecs_init ", VMSpecs_init)
        if mode == "init" and matrix_init is not None and VMSpecs_init is not None:
            print("init Z3 from prev SMT-LIB")
            add_init_values(SMTsolver, matrix_init, VMSpecs_init)
        elif mode == "init":
            print("in z3-init add_init_values")
            add_init_values(SMTsolver)
        if prediction is not None:
            print("in add_pred_soft_constraints")
            add_pred_soft_constraints(SMTsolver, prediction)
        elif prediction_init is not None:
            print("in add_init_values")
            add_init_values(SMTsolver, prediction_init)
        price, distr, runtime, a_mat, vms_type = SMTsolver.run()

        if not runtime or runtime > 10000:
            log("TESTING", "WARN", "Test aborted. Timeout")
        else:
            # print(runtime)
            vm_specs = []
            for index, (key, value) in enumerate(offers_json.items()):
                if (index + 1) in vms_type:
                    vm_spec = {key: value}
                    vm_spec[key]["id"] = index + 1
                    vm_specs.append(vm_spec)

            output = {
                "output": {
                    "min_price": sum(distr),
                    "type_of_sym_br": self.symmetry_breaker or "None",
                    "time (secs)": runtime,
                    "types_of_VMs": [vm_type.as_long() for vm_type in vms_type],
                    "prices_of_VMs": distr,
                    "VMs specs": vm_specs,
                    "assign_matr": [[el.as_long() for el in row] for row in a_mat],
                    "offers": offers_json
                }
            }
            application_model_json.update(output)
            return application_model_json