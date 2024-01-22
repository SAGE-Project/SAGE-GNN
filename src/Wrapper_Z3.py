import json
from Solvers.Core.ProblemDefinition import ManeuverProblem
from src.init import log
import src.smt
import numpy as np
import uuid
from z3 import *

def add_pred_soft_constraints(solver, prediction):
    # prediction is a matrix of size (nr comp) * (nr vms * nr offers)
    constraints = []
    nrOffers = solver.nrOffers
    nrVms = solver.nrVM
    nrComponents = solver.nrComp
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


# def add_pred_soft_constraints_sim(solver, prediction):
#     for idx, vm_type in enumerate(prediction["output"]["types_of_VMs"]):
#         solver.solver.add_soft(solver.vmType[idx] == vm_type)
#     a_matrix_flatten = [item for sublist in prediction["output"]["assign_matr"] for item in sublist]
#     for idx, val in enumerate(a_matrix_flatten):
#         solver.solver.add_soft(solver.a[idx] == val)


class Wrapper_Z3:
    def __init__(self, symmetry_breaker="None", solver_id="z3"):
        self.symmetry_breaker = symmetry_breaker
        self.solver_id = solver_id

    def solve(
            self,
            application_model_json,
            offers_json,
            prediction=None,
            prediction_sim=None,
            inst=0,
            out=True
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
                                   smt2lib=f"../Output/SMT-LIB-assert-soft/" + application_model_json["application"] + "_" + str(uuid.uuid4()))
        else:
            SMTsolver.init_problem(problem, "optimize", sb_option=self.symmetry_breaker)
        if prediction is not None:
            add_pred_soft_constraints(SMTsolver, prediction)
        # elif prediction_sim is not None:
        #     add_pred_soft_constraints_sim(SMTsolver, prediction_sim)
        price, distr, runtime, a_mat, vms_type = SMTsolver.run()

        if not runtime or runtime > 2400:
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