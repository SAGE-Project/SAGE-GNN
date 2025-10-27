import json
import pprint

from Solvers.Core.ProblemDefinition import ManeuverProblem
from src.init import log
import src.smt
import numpy as np
import uuid
from z3 import *

constraints = []

def add_pred_soft_constraints(solver, prediction):
    # prediction is a matrix of size (nr comp) * (nr vms * nr offers)
    constraints = []
    nrOffers = solver.nrOffers
    nrVms = solver.nrVM
    nrComponents = solver.nrComp
    for comp_idx in range(nrComponents):
        pred_comp = prediction[comp_idx]
        print(len(pred_comp))
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
        print("constraints", len(constraints), constraints)
        solver.solver.add_soft(constraints)

def add_pred_init_vals(solver, prediction):
    # prediction is a matrix of size (nr comp) * (nr vms * nr offers)
    print("in add_pred_init_vals")
    constraints = []
    nrOffers = solver.nrOffers
    nrVms = solver.nrVM
    nrComponents = solver.nrComp
    # Component 0 assignments
    solver.solver.set_initial_value(solver.a[0 * 8 + 0], 1)
    solver.solver.set_initial_value(solver.a[0 * 8 + 1], 0)
    solver.solver.set_initial_value(solver.a[0 * 8 + 2], 0)
    solver.solver.set_initial_value(solver.a[0 * 8 + 3], 1)
    solver.solver.set_initial_value(solver.a[0 * 8 + 4], 0)
    solver.solver.set_initial_value(solver.a[0 * 8 + 5], 1)
    solver.solver.set_initial_value(solver.a[0 * 8 + 6], 0)
    solver.solver.set_initial_value(solver.a[0 * 8 + 7], 0)
    # Component 1 assignments
    solver.solver.set_initial_value(solver.a[1 * 8 + 0], 0)
    solver.solver.set_initial_value(solver.a[1 * 8 + 1], 0)
    solver.solver.set_initial_value(solver.a[1 * 8 + 2], 1)
    solver.solver.set_initial_value(solver.a[1 * 8 + 3], 0)
    solver.solver.set_initial_value(solver.a[1 * 8 + 4], 0)
    solver.solver.set_initial_value(solver.a[1 * 8 + 5], 0)
    solver.solver.set_initial_value(solver.a[1 * 8 + 6], 1)
    solver.solver.set_initial_value(solver.a[1 * 8 + 7], 0)
    # # Component 2 assignments
    solver.solver.set_initial_value(solver.a[2 * 8 + 0], 0)
    solver.solver.set_initial_value(solver.a[2 * 8 + 1], 0)
    solver.solver.set_initial_value(solver.a[2 * 8 + 2], 0)
    solver.solver.set_initial_value(solver.a[2 * 8 + 3], 0)
    solver.solver.set_initial_value(solver.a[2 * 8 + 4], 0)
    solver.solver.set_initial_value(solver.a[2 * 8 + 5], 0)
    solver.solver.set_initial_value(solver.a[2 * 8 + 6], 0)
    solver.solver.set_initial_value(solver.a[2 * 8 + 7], 0)
    # # Component 3 assignments
    solver.solver.set_initial_value(solver.a[3 * 8 + 0], 0)
    solver.solver.set_initial_value(solver.a[3 * 8 + 1], 0)
    solver.solver.set_initial_value(solver.a[3 * 8 + 2], 0)
    solver.solver.set_initial_value(solver.a[3 * 8 + 3], 0)
    solver.solver.set_initial_value(solver.a[3 * 8 + 4], 0)
    solver.solver.set_initial_value(solver.a[3 * 8 + 5], 0)
    solver.solver.set_initial_value(solver.a[3 * 8 + 6], 1)
    solver.solver.set_initial_value(solver.a[3 * 8 + 7], 0)
    # # Component 4 assignments
    solver.solver.set_initial_value(solver.a[4 * 8 + 0], 0)
    solver.solver.set_initial_value(solver.a[4 * 8 + 1], 1)
    solver.solver.set_initial_value(solver.a[4 * 8 + 2], 0)
    solver.solver.set_initial_value(solver.a[4 * 8 + 3], 0)
    solver.solver.set_initial_value(solver.a[4 * 8 + 4], 0)
    solver.solver.set_initial_value(solver.a[4 * 8 + 5], 0)
    solver.solver.set_initial_value(solver.a[4 * 8 + 6], 0)
    solver.solver.set_initial_value(solver.a[3 * 8 + 7], 1)
    # ##### VM 0
    solver.solver.set_initial_value(solver.ProcProv[0], solver.offers_list[432][1])
    solver.solver.set_initial_value(solver.MemProv[0] , solver.offers_list[432][2])
    solver.solver.set_initial_value(solver.StorageProv[0], solver.offers_list[432][3])
    solver.solver.set_initial_value(solver.PriceProv[0], solver.offers_list[432][4])
    # # VM 1
    solver.solver.set_initial_value(solver.ProcProv[1], solver.offers_list[340][1])
    solver.solver.set_initial_value(solver.MemProv[1], solver.offers_list[340][2])
    solver.solver.set_initial_value(solver.StorageProv[1], solver.offers_list[340][3])
    solver.solver.set_initial_value(solver.PriceProv[1], solver.offers_list[340][4])
    # # VM 2
    solver.solver.set_initial_value(solver.ProcProv[2], solver.offers_list[340][1])
    solver.solver.set_initial_value(solver.MemProv[2], solver.offers_list[340][2])
    solver.solver.set_initial_value(solver.StorageProv[2], solver.offers_list[340][3])
    solver.solver.set_initial_value(solver.PriceProv[2], solver.offers_list[340][4])
    # # VM 3
    solver.solver.set_initial_value(solver.ProcProv[3], solver.offers_list[432][1])
    solver.solver.set_initial_value(solver.MemProv[3], solver.offers_list[432][2])
    solver.solver.set_initial_value(solver.StorageProv[3], solver.offers_list[432][3])
    solver.solver.set_initial_value(solver.PriceProv[3], solver.offers_list[432][4])
    # # VM 4
    solver.solver.set_initial_value(solver.ProcProv[4], solver.offers_list[340][1])
    solver.solver.set_initial_value(solver.MemProv[4], solver.offers_list[340][2])
    solver.solver.set_initial_value(solver.StorageProv[4], solver.offers_list[340][3])
    solver.solver.set_initial_value(solver.PriceProv[4], solver.offers_list[340][4])
    # # VM 5
    solver.solver.set_initial_value(solver.ProcProv[5], solver.offers_list[432][1])
    solver.solver.set_initial_value(solver.MemProv[5], solver.offers_list[432][2])
    solver.solver.set_initial_value(solver.StorageProv[5], solver.offers_list[432][3])
    solver.solver.set_initial_value(solver.PriceProv[5], solver.offers_list[432][4])
    # # VM 6
    solver.solver.set_initial_value(solver.ProcProv[6], solver.offers_list[432][1])
    solver.solver.set_initial_value(solver.MemProv[6], solver.offers_list[432][2])
    solver.solver.set_initial_value(solver.StorageProv[6], solver.offers_list[432][3])
    solver.solver.set_initial_value(solver.PriceProv[6], solver.offers_list[432][4])
    # # VM 7
    solver.solver.set_initial_value(solver.ProcProv[7], solver.offers_list[432][1])
    solver.solver.set_initial_value(solver.MemProv[7], solver.offers_list[432][2])
    solver.solver.set_initial_value(solver.StorageProv[7], solver.offers_list[432][3])
    solver.solver.set_initial_value(solver.PriceProv[7], solver.offers_list[432][4])

    # with open('out.txt', 'w+') as f:
    #     for comp_idx in range(nrComponents):
    #         pred_comp = prediction[comp_idx]
    #         matrix = np.reshape(pred_comp, (nrOffers, nrVms))
    #         for vm_idx in range(solver.nrVM):
    #             pred_comp_vm = matrix[:, vm_idx]
    #             print("pred_comp_vm ", pred_comp_vm, file=f)
    #             placements = [i for i, x
    #                       in enumerate(pred_comp_vm)
    #                       if x == 1]
    #             print("placements ", placements)
    #             a_matrix_index = comp_idx * solver.nrVM + vm_idx
    #             print("a_matrix_index ",a_matrix_index)
    #             if len(placements) != 0:
    #                 print(solver.a[a_matrix_index], '=1', file=f)
    #                 #print(" = 1", file=f)
    #                 solver.solver.set_initial_value(solver.a[a_matrix_index], 1)
    #             else:
    #                 print(solver.a[a_matrix_index], '=0', file=f)
    #                 #print(" = 0", file=f)
    #                 solver.solver.set_initial_value(solver.a[a_matrix_index], 0)
    #         for placement in placements:
    #             vmType = placement + 1
    #             print("vmType ", vmType)
    #             solver.solver.set_initial_value(solver.ProcProv[vm_idx], solver.offers_list[vmType][1])
    #             print(solver.ProcProv[vm_idx], solver.offers_list[vmType][1], file=f)
    #             solver.solver.set_initial_value(solver.MemProv[vm_idx] , solver.offers_list[vmType][2])
    #             print(solver.MemProv[vm_idx] , solver.offers_list[vmType][2], file=f)
    #             solver.solver.set_initial_value(solver.StorageProv[vm_idx], solver.offers_list[vmType][3])
    #             print(solver.StorageProv[vm_idx], solver.offers_list[vmType][3], file=f)
    #             solver.solver.set_initial_value(solver.PriceProv[vm_idx], solver.offers_list[vmType][4])
    #             print(solver.PriceProv[vm_idx], solver.offers_list[vmType][4], file=f)

class Wrapper_Z3:
    def __init__(self, symmetry_breaker="FVPR", solver_id="z3"):
        self.symmetry_breaker = symmetry_breaker
        self.solver_id = solver_id

    def solve(
            self,
            application_model_json,
            offers_json,
            mode=None,
            prediction=None,
            #prediction_sim=None,
            inst = 3,
            out=True
            #out=False
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
                                   smt2lib=f"/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Output/SMT-LIB/SecureWebContainer/" + application_model_json["application"] + "_" + str(uuid.uuid4()))
        else:
            SMTsolver.init_problem(problem, "optimize", sb_option=self.symmetry_breaker)
        if (prediction is not None) and (mode=="init"):
            add_pred_init_vals(SMTsolver, prediction)
        elif (prediction is not None) and (mode=="gnn"):
            add_pred_soft_constraints(SMTsolver, prediction)
        print("prediction ", prediction)
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