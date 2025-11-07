from torch.ao.quantization import per_channel_weight_observer_range_neg_127_to_127
from z3 import *
from Solvers.Core.ManuverSolver import ManuverSolver
import time

class Z3_Solver_Int_Parent(ManuverSolver):#ManeuverProblem):

    def _initSolver(self):
        """
        Initializes the solver
        :return: None
        """
        if self.solverTypeOptimize:
            #TODO: set_param does not take into account the mode=init or not and
            # has to be commented manually for wrapper_z3.solve(application, offers_do) respectivelly for print(wrapper_z3.solve(application, offers_do, matrix_init=matrix, VMSpecs_init=VMSpecs, mode="init"))
            # set_option(verbose=10)
            # set_param("opt.elim_01", False)  # perhaps not necessary, but keeps it simpler wrt initialization
            # set_param("opt.dump_models", True)  # dump best current solution so far
            # set_param("smt.elim_term_ite", False)  # avoids creating new variables that can obscure initial value setting.
            self.solver = Optimize()
        else:
            self.solver = Solver()
            self.solver.set(unsat_core=True)
            #self.solver.set("sat.cardinality.solver", True)
            self.labelIdx = 0
            self.labelIdx_oneToOne = 0
            self.labelIdx_offer = 0
            self.labelIdx_conflict = 0

        self.vmIds_for_fixedComponents = set()
        self._defineVariablesAndConstraints()

    def _defineVariablesAndConstraints(self):
        # VM usage vector vm in {0, 1}, k = 1..M; vm_k = 1 if at least one component is assigned to vm_k.
        #self.vm = {}
        # Assignment matrix a_{alpha,k}: 1 if component alpha is on machine k, 0 otherwise
        self.a = {}
        # VMType  - type of a leased VM
        self.VMType = {}

    def RestrictionLexBinaryNumber(self):
        """
        Transform column into a binary number - not tested preprocessing reorder components by number
        :return:
        """

        for vm_id in range(self.nrVM - 1):
            list_comps = []
            for comp_id in range(self.nrComp):
                if not self.problem.componentsList[comp_id].fullDeployedComponent:
                    list_comps.append(comp_id)
        n = len(list_comps)
        n = n - 1
        for vm_id in range(self.nrVM - 1):
            self.solver.add(
                sum([self.a[list_comps[i] * self.nrVM + vm_id] * (2 ** (n - i)) for i in range(len(list_comps))]) >=
                sum([self.a[list_comps[i] * self.nrVM + vm_id + 1] * (2 ** (n - i)) for i in range(len(list_comps))]))


    def RestrictionLex(self, vm_id, additional_constraints=[]):
        """
        Lexicografic order between two consecutive columns
        :param vm_id: the vm id
        :param additonal_constraints: other constraints
        :return:
        """
        l = additional_constraints.copy()
        #print("self.nrComp ", self.nrComp)
        for i in range(0, self.nrComp):
            if i > 0:
                u = i-1
                l.append(self.a[u * self.nrVM + vm_id] == self.a[u * self.nrVM + vm_id + 1])
            #print(l)
            self.solver.add(Implies(And(l), self.a[i * self.nrVM + vm_id] >= self.a[i * self.nrVM + vm_id + 1]))


    def RestrictionPrice(self, vm_id, additional_constraints=[]):
        """
        Lexicografic order between two consecutive columns
        :param vm_id: the vm id
        :param additonal_constraints: other constraints
        :return:
        """
        #print("Price vm ", vm_id, "combined=", additional_constraints)
        if len(additional_constraints) == 0:
            self.solver.add(self.PriceProv[vm_id] >= self.PriceProv[vm_id + 1])
        else:
            if len(additional_constraints) == 1:
                self.solver.add(
                    Implies(additional_constraints[0], self.PriceProv[vm_id] >= self.PriceProv[vm_id + 1]))
            else:
                self.solver.add\
                    (Implies(And(additional_constraints), self.PriceProv[vm_id] >= self.PriceProv[vm_id + 1]))

        return self.PriceProv[vm_id] == self.PriceProv[vm_id + 1]

    def RestrictionFixComponentOnVM(self, comp_id, vm_id, value):
        """
        Force placing component on a specific VM
        :param comp_id: the ID of the component
        :param vm_id: the ID of the VM
        :param value: 1 assigned; 0 unassigned
        :return: None
        """
        self.solver.add(self.a[comp_id * self.nrVM + vm_id] == value)

    def RestrictionLoad(self, vm_id, additional_constraints=[]):
        """
        Lexicografic order between two consecutive columns
        :param vm_id: the vm id
        :param additonal_constraints: other constraints
        :return:
        """
        if len(additional_constraints) == 0:
           self.solver.add(sum([self.a[i + vm_id] for i in range(0, len(self.a), self.nrVM)])
                                 >= sum([self.a[i + vm_id + 1] for i in range(0, len(self.a), self.nrVM)]))
        else:
            if len(additional_constraints) == 1:
                self.solver.add(Implies(additional_constraints[0],
                                        sum([self.a[i + vm_id] for i in range(0, len(self.a), self.nrVM)])
                                        >= sum([self.a[i + vm_id + 1] for i in range(0, len(self.a), self.nrVM)])))
            else:
                self.solver.add(Implies(And(additional_constraints),
                                        sum([self.a[i + vm_id] for i in range(0, len(self.a), self.nrVM)])
                                        >= sum([self.a[i + vm_id + 1] for i in range(0, len(self.a), self.nrVM)])))


        return sum([self.a[i + vm_id] for i in range(0, len(self.a), self.nrVM)]) == \
               sum([self.a[i + vm_id + 1] for i in range(0, len(self.a), self.nrVM)])

    def RestrictionConflict(self, alphaCompId, conflictCompsIdList):
        """
        Constraint describing the conflict between components. The 2 params. should not be placed on the same VM
        :param alphaCompId: id of the first conflict component
        :param conflictCompsIdList: id of the second conflict component
        :return: None
        """
        self.problem.logger.debug(
            "RestrictionConflict: alphaCompId = {} conflictComponentsList = {}".format(alphaCompId,
                                                                                       conflictCompsIdList))
        for j in range(self.nrVM):
            for conflictCompId in conflictCompsIdList:
                # self.problem.logger.debug("...{} <= 1".format([self.a[alphaCompId * self.nrVM + j], self.a[conflictCompId * self.nrVM + j]]))
                if self.solverTypeOptimize:
                    self.solver.add(sum([self.a[alphaCompId * self.nrVM + j],
                                         self.a[conflictCompId * self.nrVM + j]]) <= 1)
                else:
                    self.solver.assert_and_track(
                        sum([self.a[alphaCompId * self.nrVM + j], self.a[conflictCompId * self.nrVM + j]]) <= 1,
                        "LabelConflict: " + str(self.labelIdx_conflict))
                    self.labelIdx_conflict += 1

    def RestrictionOneToOneDependency(self, alphaCompId, betaCompId):
        """
        Contraint describing that alphaCompId and betaCompId should be deployed on the same VM
        :param alphaCompId: id of the first component
        :param betaCompId: id of the second component
        :return: None
        """
        for j in range(self.nrVM):
            if self.solverTypeOptimize:
                self.solver.add(
                    self.a[alphaCompId * self.nrVM + j] == self.a[betaCompId * self.nrVM + j])
            else:
                self.solver.assert_and_track(
                    self.a[alphaCompId * self.nrVM + j] == self.a[betaCompId * self.nrVM + j],
                    "LabelOneToOne" + str(self.labelIdx))
                self.labelIdx_oneToOne += 1

    def RestrictionManyToManyDependency(self, alphaCompId, betaCompId, relation):
        """
        The number of instances of component alphaCompId depends on the number of instances of component betaCompId
        :param alphaCompId: id of the first component
        :param betaCompId: id of the second component
        :param relation: one of the strings in the set {"=", "<=", ">="}
            "=": sum(instances of alpha component) == sum(instances of beta component)
            "<=": sum(instances of alpha component) <= sum(instances of beta component)
            ">=": sum(instances of alpha component) >= sum(instances of beta component)
        :return: None
        """
        if relation == "<=":
            if self.solverTypeOptimize:
                self.solver.add(
                    sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) <=
                    sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]))
            else:
                self.solver.assert_and_track(
                    sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) <=
                    sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]), "LabelManyToMany1: " + str(self.labelIdx))
                self.labelIdx += 1
        elif relation == ">=":
            if self.solverTypeOptimize:
                self.solver.add(
                    sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) >=
                    sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]))
            else:
                self.solver.assert_and_track(
                    sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) >=
                    sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]), "LabelManyToMany2: " + str(self.labelIdx))
                self.labelIdx += 1
        elif relation == "=":
            if self.solverTypeOptimize:
                self.solver.add(
                    sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) ==
                    sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]))
            else:
                self.solver.assert_and_track(
                    sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) ==
                    sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]), "LabelManyToMany3: " + str(self.labelIdx))
                self.labelIdx += 1

    def RestrictionOneToManyDependency(self, alphaCompId, betaCompId, noInstances):
        """
        At each alphaCompId component should be deployed noInstances betaCompId components
        :param alphaCompId: id of the first component
        :param betaCompId: id of the second component
        :param noInstances: depending instances number
        :return: None
        """
        if self.solverTypeOptimize:
            self.solver.add(
                noInstances * sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) -
                              sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]) > 0)
        else:
            self.solver.assert_and_track(
                noInstances * sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) -
                              sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]) > 0, "LabelOneToMany: " + str(self.labelIdx))
            self.labelIdx += 1

        if self.solverTypeOptimize:
            self.solver.add(
                noInstances * sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) -
                              sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]) <= noInstances)
        else:
            self.solver.assert_and_track(
                noInstances * sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) -
                              sum([self.a[betaCompId  * self.nrVM + j] for j in range(self.nrVM)]) <= noInstances, "LabelOneToMany: " + str(self.labelIdx))
            self.labelIdx += 1

    def RestrictionUpperLowerEqualBound(self, compsIdList, bound, operator):
        """
        Defines an upper/lower/equal bound on the number of instances that a component must have
        :param compsIdList: list of components
        :param bound: a positive number
        :param operator: <=, >=, =
            "<=": sum(compsIdList) <= bound
            ">=": sum(compsIdList) >= bound
            "==":  sum(compsIdList) == bound
        :return: None
        """

        self.problem.logger.debug("RestrictionUpperLowerEqualBound: {} {} {} ".format(compsIdList, operator, bound))

        if operator == "<=":
            if self.solverTypeOptimize:
                self.solver.add(
                    sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)])
                    <= bound)
            else:
                #self.__constMap[str("LabelUpperLowerEqualBound" + str(self.labelIdx))] = sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)]) <= bound
                self.solver.assert_and_track(
                    sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)])
                    <= bound, "LabelUpperLowerEqualBound" + str(self.labelIdx))
                self.labelIdx += 1
        elif operator == ">=":
            if self.solverTypeOptimize:
                self.solver.add(
                    sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)]) >= bound)
            else:
                #self.__constMap[str("LabelUpperLowerEqualBound" + str(self.labelIdx))] = sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)]) >= bound
                self.solver.assert_and_track(
                    sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)]) >= bound, "LabelUpperLowerEqualBound" + str(self.labelIdx))
                self.labelIdx += 1
        elif operator == "=":
            if self.solverTypeOptimize:
                self.solver.add(
                    sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)]) == bound)
            else:
                #self.__constMap[str("LabelUpperLowerEqualBound" + str(self.labelIdx))] = sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)]) == bound

                self.solver.assert_and_track(
                    sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)]) == bound, "LabelUpperLowerEqualBound" + str(self.labelIdx))
                self.labelIdx += 1
        else:
            self.problem.logger.info("Unknown operator")

    def RestrictionRangeBound(self, compsIdList, lowerBound, upperBound):
        """
        Defines a lower and upper bound of instances that a component must have
        :param compsIdList: list of components
        :param lowerBound: a positive number
        :param upperBound: a positive number
        :return:
        """
        for i in range(len(compsIdList)): compsIdList[i] -= 1
        if self.solverTypeOptimize:
            self.solver.add(sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)]) >= lowerBound)
        else:
            self.solver.assert_and_track(
                sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)]) >= lowerBound, "LabelRangeBound: " + str(self.labelIdx))
            self.labelIdx += 1
        if self.solverTypeOptimize:
            self.solver.add(sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)]) <= upperBound)
        else:
            self.solver.assert_and_track(
                sum([self.a[compId * self.nrVM + j] for compId in compsIdList for j in range(self.nrVM)]) <= upperBound, "LabelRangeBound: " + str(self.labelIdx))
            self.labelIdx += 1

    def RestrictionFullDeployment(self, alphaCompId, notInConflictCompsIdList):
        """
        Adds the fact that the component alphaCompId must be deployed on all machines except the ones that contain
         components that alphaCompId alpha is in conflict with
        :param alphaCompId: the component which must be fully deployed
        :param notInConflictCompsIdList: the list of components that alphaCompId is not in conflict in
        :return: None
        """
        for j in range(self.nrVM):
            if self.solverTypeOptimize:
                self.solver.add(
                    (sum([self.a[alphaCompId * self.nrVM + j]] + [self.a[_compId * self.nrVM + j] for _compId in
                                                                  notInConflictCompsIdList]))
                    ==
                    (If(sum([self.a[i + j] for i in range(0, len(self.a), self.nrVM)]) >= 1, 1, 0)))
            else:
                self.solver.assert_and_track(
                    (sum([self.a[alphaCompId * self.nrVM + j]] + [self.a[_compId * self.nrVM + j] for _compId in
                                                                  notInConflictCompsIdList]))
                    ==
                    (If(sum([self.a[i + j] for i in range(0, len(self.a), self.nrVM)]) >= 1, 1, 0)),
                    "LabelFullDeployment: " + str(self.labelIdx)
                )
                self.labelIdx += 1

    def RestrictionRequireProvideDependency(self, alphaCompId, betaCompId, alphaCompIdInstances, betaCompIdInstances):
        """
        The number of instances of component alpha depends on the number of instances of component beta
        :param alphaCompId: id of the first component
        :param betaCompId: id of the second component
        :param alphaCompIdInstances: number of instances of component alphaCompId
        :param betaCompIdInstances: number of instances of component betaCompId
        :return: None
        """
        # self.problem.logger.debug("RestrictionRequireProvideDependency: alphaCompId={}, betaCompId={}, alphaCompIdInstances={}, "
        #                          "betaCompIdInstances={}".format(alphaCompId, betaCompId, alphaCompIdInstances, betaCompIdInstances))

        if self.solverTypeOptimize:
            self.solver.add(Or(sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]) == 0,
                alphaCompIdInstances * sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) <=
                betaCompIdInstances * sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)])))
        else:
            self.__constMap["LabelRequireProvide: " + str(self.labelIdx)] = \
                alphaCompIdInstances * sum([If(self.a[alphaCompId * self.nrVM + j], 1, 0) for j in range(self.nrVM)]) \
                <= \
                betaCompIdInstances * sum([If(self.a[betaCompId * self.nrVM + j], 1, 0) for j in range(self.nrVM)])
            self.solver.assert_and_track(
                alphaCompIdInstances * sum([If(self.a[alphaCompId * self.nrVM + j], 1, 0) for j in range(self.nrVM)]) <=
                betaCompIdInstances * sum([If(self.a[betaCompId * self.nrVM + j], 1, 0) for j in range(self.nrVM)]),
                "LabelRequireProvide: " + str(self.labelIdx))
            self.labelIdx += 1

    def RestrictionAlphaOrBeta(self, alphaCompId, betaCompId):
        """
        Describes the fact that alphaCompId or betaCompId not both
        :param alphaCompId: id of the first component
        :param betaCompId: id of the second component
        :return:
        """
        self.problem.logger.debug("RestrictionAlphaOrBeta: alphaCompId={}, betaCompId={}".format(alphaCompId, betaCompId))
        if self.solverTypeOptimize:
            self.solver.add(Or(sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]) == 0,
                               sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]) >= 1))

            self.solver.add(Or(sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) == 0,
                               sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) >= 1))

            self.solver.add(sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]) +
                            sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) >= 1)

            # self.solver.add(
            #     Xor(sum([self.a[betaCompId * self.nrVM + j] for j in range(self.nrVM)]) == 0,
            #         sum([self.a[alphaCompId * self.nrVM + j] for j in range(self.nrVM)]) == 0, True))
        else:
            self.solver.assert_and_track(
                Or(sum([If(self.a[betaCompId * self.nrVM + j], 1, 0) for j in range(self.nrVM)]) == 0,
                   sum([If(self.a[betaCompId * self.nrVM + j], 1, 0) for j in range(self.nrVM)]) >= 1),
                "LabelAlphaOrBeta: " + str(self.labelIdx))
            self.labelIdx += 1

            self.solver.assert_and_track(
                Or(sum([If(self.a[alphaCompId * self.nrVM + j], 1, 0) for j in range(self.nrVM)]) == 0,
                   sum([If(self.a[alphaCompId * self.nrVM + j], 1, 0) for j in range(self.nrVM)]) >= 1),
                "LabelAlphaOrBeta: " + str(self.labelIdx))
            self.labelIdx += 1

            self.solver.assert_and_track(sum([If(self.a[betaCompId * self.nrVM + j], 1, 0) for j in range(self.nrVM)]) +
                                         sum([If(self.a[alphaCompId * self.nrVM + j], 1, 0) for j in range(self.nrVM)])
                                         >= 1, "LabelAlphaOrBeta: " + str(self.labelIdx))
            self.labelIdx += 1

    def createSMT2LIBFile(self, fileName):
        """
        File creation
        :param fileName: string representing the file name storing the SMT2LIB formulation of the problem
        :return:
        """
        #with open(fileName, 'w+') as fo:
        #   fo.write("(set-logic QF_LIA)\n") # quantifier free linear integer-real arithmetic
        #fo.close()
        if fileName is None: return
        with open(fileName, 'w+') as fo:
            fo.write(self.solver.sexpr())
        fo.close()

    def createSMT2LIBFileSolution(self, fileName, status, model):
        """
        File creation
        :param fileName: string representing the file name storing the SMT2LIB formulation of the problem
        :param status: SAT/UNSAT
        :param model: string representing key-values pairs for the variables in the model
        :return:
        """
        if fileName is None: return
        with open(fileName, 'w+') as foo:
            foo.write(repr(status)+ '[\n')
            for k in model:
                foo.write('%s = %s, ' % (k, model[k]))
                foo.write('\n')
            foo.write(']')
        foo.close()

    def convert_price(self, price):
        return price

    def run(self):
        """
        Invokes the solving of the problem, generates the SMT-LIB problem file,
        and writes the result (SAT/UNSAT/UNKNOWN), model, and statistics
        into the solution file.
        """

        # If optimization mode is active
        if self.solverTypeOptimize:
            opt = sum(self.PriceProv)
            min_obj = self.solver.minimize(opt)

        # 1️⃣ Save the SMT problem encoding before solving
        self.createSMT2LIBFile(self.smt2lib)

        # 2️⃣ Run the solver and measure runtime
        self.get_current_time()
        start_time = time.time()
        status = self.solver.check()

        stop_time = time.time()

        # 3️⃣ Handle UNSAT cores (for non-optimize mode)
        if not self.solverTypeOptimize:
            unsat_core = self.solver.unsat_core()
            self.problem.logger.debug(f"unsat_constraints= {unsat_core}")
            if len(unsat_core) > 0:
                print(f"unsat_constraints= {unsat_core}")

        # 4️⃣ Log and print status
        self.problem.logger.info(f"Z3 status: {status}")
        print(f"Z3 status: {status}")

        # 5️⃣ Collect model data if SAT
        model_dict = {}
        if status == sat:
            model = self.solver.model()
            for d in model.decls():
                model_dict[str(d)] = str(model[d])

        # 6️⃣ Write result + model + statistics into one SMT2LIB solution file
        self.createSMT2LIBFileSolution(
            self.smt2libsol,
            str(status).upper(),
            model_dict,
            solver=self.solver
        )

        # 7️⃣ Post-process results depending on optimization mode
        if self.solverTypeOptimize:
            if status == sat:
                # Build assignment matrix
                a_mat = []
                for i in range(self.nrComp):
                    l = []
                    for k in range(self.nrVM):
                        l.append(model[self.a[i * self.nrVM + k]])
                    a_mat.append(l)

                # VM prices
                vms_price = [
                    int(self.convert_price(model[self.PriceProv[k]]) * 1000)
                    for k in range(self.nrVM)
                ]

                # VM types
                vms_type = [model[self.vmType[k]] for k in range(self.nrVM)]

                # Return optimization data
                return (
                    self.convert_price(min_obj.value()),
                    vms_price,
                    stop_time - start_time,
                    a_mat,
                    vms_type,
                )
            else:
                # UNSAT in optimization case
                return -1, None, None, None, None
        else:
            # Non-optimization case: just return timing
            return None, None, stop_time - start_time

    def createSMT2LIBFileSolution(self, fileName, status, model, solver=None):
        """
        Writes the solver result, model, objective values, and stats into one SMT-LIB-like file.
        """
        if fileName is None:
            return

        with open(fileName, 'w+', encoding='utf-8') as f:
            f.write("; =========================================\n")
            f.write("; Z3 Solver Result\n")
            f.write("; =========================================\n\n")

            f.write(f"; STATUS: {status}\n\n")
            f.write("(set-logic QF_LIA)\n\n")

            if status == "SAT":
                # 1) Model
                f.write("(model\n")
                for k, v in model.items():
                    f.write(f"  (define-fun {k} () Int {v})\n")
                f.write(")\n\n")

                # 2) Objectives (only if Optimize was used)
                if solver is not None and isinstance(solver, Optimize):
                    # Ensure a model exists for evaluation (should, since status == SAT)
                    try:
                        m = solver.model()
                    except Exception:
                        m = None

                    objs = list(solver.objectives())
                    if objs:
                        f.write("; Optimized objective values\n")
                        for i, obj in enumerate(objs, 1):
                            val = None
                            # Prefer OptimizeObjective.value() where available
                            try:
                                val = obj.value()
                            except Exception:
                                pass
                            # Fallback: try evaluating as an expression on the model
                            if val is None and m is not None:
                                try:
                                    val = m.eval(obj, model_completion=True)
                                except Exception:
                                    val = None
                            if val is not None:
                                try:
                                    sort_name = val.sort().name()
                                except Exception:
                                    sort_name = "Int"  # safe default
                                f.write(f"(define-fun objective{i} () {sort_name} {val})\n")
                            else:
                                f.write(f"; objective{i} value unavailable\n")
                        f.write("\n")
            else:
                f.write(f"; No model available ({status})\n\n")

            # Statistics (similar to -st)
            if solver is not None:
                try:
                    f.write("; =========================================\n")
                    f.write("; Z3 Statistics (-st equivalent)\n")
                    f.write("; =========================================\n\n")
                    stats = solver.statistics()
                    for k in stats.keys():
                        f.write(f"{k}: {stats.get_key_value(k)}\n")
                except Exception:
                    pass

            f.write("\n; End of SMT2LIB result file\n")



