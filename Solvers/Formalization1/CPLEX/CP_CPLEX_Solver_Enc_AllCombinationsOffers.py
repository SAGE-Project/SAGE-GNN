from Solvers.Formalization1.CPLEX.CP_CPLEX_Solver import CPlex_Solver_Parent
from Solvers.Core.ManuverSolver_SB import ManuverSolver_SB

class CPlex_Solver_SB_Enc_AllCombinationsOffers(CPlex_Solver_Parent, ManuverSolver_SB):

    def _define_variables(self):
        """
        Creates the variables used in the solver and the constraints on them as well as others (offers encoding,
        usage vector, etc.)
        :return: None
        """

        # VM usage vector vm in {0, 1}, k = 1..M; vm_k = 1 if at least one component is assigned to vm_k.
        self.vm = {j: self.model.binary_var(name="vm{0}".format(j+1)) for j in range(self.nr_vms)}

        # definition of assignment matrix a \in {0, 1}. Even in the output one has 0<=x<=1, the below ensures integrality
        self.a = {}
        for comp_idx in range(self.nr_comps):
            for vm_idx in range(self.nr_vms):
                var_name = f"C{comp_idx + 1}_VM{vm_idx + 1}"
                self.a[(comp_idx, vm_idx)] = self.model.binary_var(name=var_name)

        # occupancy
        # for j in range(self.nr_vms):
        #     self.model.add_equivalence( self.vm[j], self.model.sum(self.a[i, j] for i in range(self.nr_comps)) >= 1, name="c{0}_vm_allocated".format(j))
        # It generates the following but this is how implications are handled. In fact <-> is equiv by definition
        # occupancy_vm_0: _bool  # 1 = 1 -> vm1 = 1
        #lc3: _bool  # 2 = 1 <-> C1_VM2 + C2_VM2 + C3_VM2 + C4_VM2 + C5_VM2 >= 1
        for j in range(self.nr_vms):
            sum_a = self.model.sum(self.a[i, j] for i in range(self.nr_comps))
            # create the implication constraint
            occ_constraint = self.model.if_then(sum_a >= 1, self.vm[j] == 1)
            # assign a name (after creation)
            occ_constraint.name = f"occupancy_vm_{j}"
            # add it to the model
            self.model.add(occ_constraint)

        #Variables for offers description
        maxType = len(self.offers_list)
        self.vmType = {(j): self.model.integer_var(lb=0, ub=maxType, name="vmType{0}".format(j + 1))
                       for j in range(self.nr_vms)}

        minProc = min(self.offers_list[t][1] for t in range(len(self.offers_list)))
        maxProc = max(self.offers_list[t][1] for t in range(len(self.offers_list)))
        self.ProcProv = {(j): self.model.integer_var(lb=minProc, ub=maxProc, name="ProcProv{0}".format(j + 1)) for j in range(self.nr_vms)}

        minMem = min(self.offers_list[t][2] for t in range(len(self.offers_list)))
        maxMem = max(self.offers_list[t][2] for t in range(len(self.offers_list)))
        self.MemProv = {(j): self.model.integer_var(lb=minMem, ub=maxMem, name="MemProv{0}".format(j + 1)) for j in range(self.nr_vms)}

        minSto = min(self.offers_list[t][3] for t in range(len(self.offers_list)))
        maxSto = max(self.offers_list[t][3] for t in range(len(self.offers_list)))
        self.StorageProv = {(j): self.model.integer_var(lb=minSto, ub=maxSto, name="StorageProv{0}".format(j + 1)) for j in range(self.nr_vms)}

        maxPrice = max(self.offers_list[t][len(self.offers_list[0]) - 1] for t in range(len(self.offers_list)))
        self.PriceProv = {(j): self.model.integer_var(lb=0, ub=maxPrice, name="PriceProv{0}".format(j + 1)) for j in range(self.nr_vms)}

        # If a machine is not leased then its price is 0
        for j in range(self.nr_vms):
            self.model.add_indicator(self.vm[j], self.PriceProv[j] == 0, active_value=0, name="c{0}_vm_free_price_0".format(j))

    def _hardware_and_offers_restrictionns(self, scaleFactor):
        """
        Describes the hardware requirements for each component
        :param componentsRequirements: list of components requirements as given by the user
        :return: None
        """
        for k in range(self.nr_vms):
            self.model.add_constraint(ct=self.model.sum(self.a[i, k] * (self.problem.componentsList[i].HC)
                                    for i in range(self.nr_comps)) <= self.ProcProv[k], ctname="c_hard_cpu")
            self.model.add_constraint(ct=self.model.sum(self.a[i, k] * (self.problem.componentsList[i].HM)
                                    for i in range(self.nr_comps)) <= self.MemProv[k], ctname="c_hard_mem")
            self.model.add_constraint(ct=self.model.sum(self.a[i, k] * (self.problem.componentsList[i].HS)
                                        for i in range(self.nr_comps)) <= self.StorageProv[k], ctname="c_hard_storage")

        # index_constraint = 0
        # for vm_id in range(self.nr_vms):
        #     cnt = 0
        #     for offer in self.offers_list:
        #         cnt += 1
        #         index_constraint += 1
        #
        #         var = self.model.binary_var(name="aux_hard{0}".format(index_constraint))
        #         ct = self.model.add_equivalence(var, self.vmType[vm_id] == cnt)
        #
        #         self.model.add_indicator(var,
        #                                  self.PriceProv[vm_id] == int(offer[len(self.offers_list[0]) - 1]),
        #                                  active_value=1, name="c_order_vm_price".format(vm_id))
        #         self.model.add_indicator(var, (self.ProcProv[vm_id] == int(offer[1])),
        #                                  name="c_order_vm_cpu".format(vm_id))
        #         self.model.add_indicator(var, (self.MemProv[vm_id] == int(offer[2])),
        #                                  name="c_order_vm_memory".format(vm_id))
        #         self.model.add_indicator(var, (self.StorageProv[vm_id] == int(offer[3])),
        #                                  name="c_order_vm_storage".format(vm_id))
        #
        #     lst = [(self.vmType[vm_id] == offer) for offer in range(1, len(self.offers_list)+1)]
        #     ct = self.model.add_indicator(self.vm[vm_id], self.vmType[vm_id] >= 1)
        # --- replace the whole block with this ---

        O = len(self.offers_list)
        offer_cpu = [int(self.offers_list[o][1]) for o in range(O)]
        offer_mem = [int(self.offers_list[o][2]) for o in range(O)]
        offer_sto = [int(self.offers_list[o][3]) for o in range(O)]
        offer_price = [int(self.offers_list[o][4]) for o in range(O)]

        # One-hot type selection: y[j,o] = 1 iff VM j uses offer (o+1)
        self.y = {(j, o): self.model.binary_var(name=f"y_vm{j + 1}_t{o + 1}")
                  for j in range(self.nr_vms) for o in range(O)}

        for j in range(self.nr_vms):
            # exactly one type iff VM is used (no equivalence/indicators needed)
            self.model.add_constraint(
                self.model.sum(self.y[j, o] for o in range(O)) == self.vm[j],
                ctname=f"link_use_select_vm{j + 1}"
            )

            # provision variables equal the specs of the chosen offer
            self.model.add_constraint(
                self.PriceProv[j] == self.model.sum(offer_price[o] * self.y[j, o] for o in range(O)),
                ctname=f"link_price_vm{j + 1}"
            )
            self.model.add_constraint(
                self.ProcProv[j] == self.model.sum(offer_cpu[o] * self.y[j, o] for o in range(O)),
                ctname=f"link_cpu_vm{j + 1}"
            )
            self.model.add_constraint(
                self.MemProv[j] == self.model.sum(offer_mem[o] * self.y[j, o] for o in range(O)),
                ctname=f"link_mem_vm{j + 1}"
            )
            self.model.add_constraint(
                self.StorageProv[j] == self.model.sum(offer_sto[o] * self.y[j, o] for o in range(O)),
                ctname=f"link_sto_vm{j + 1}"
            )

            # (Optional) keep vmType: index of chosen offer, or 0 if vm is unused
            if hasattr(self, "vmType"):
                self.model.add_constraint(
                    self.vmType[j] == self.model.sum((o + 1) * self.y[j, o] for o in range(O)),
                    ctname=f"link_vmType_vm{j + 1}"
                )

        # NOTE: with the one-hot relation, this is now redundant and can be removed:
        # ct = self.model.add_indicator(self.vm[vm_id], self.vmType[vm_id] >= 1)
        # Because sum_o y[j,o] == vm[j] already enforces vmType>0 when vm[j]=1 (if you keep vmType).

    def _same_type(self, var, vm_id):
        self.model.add_equivalence(var, self.vmType[vm_id] == self.vmType[vm_id + 1])

    def _get_solution_vm_type(self):
        vm_types = []
        for index, var in self.vmType.items():
            vm_types.append(var.solution_value)
        return vm_types