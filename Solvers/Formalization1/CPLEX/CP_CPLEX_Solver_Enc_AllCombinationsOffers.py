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

        # Assignment matrix a_{alpha,k}: 1 if component alpha is on machine k, 0 otherwise

        self.a = {(i, j): self.model.binary_var(name="C{0}_VM{1}".format(i+1, j+1))
                  for i in range(self.nr_comps) for j in range(self.nr_vms)}

        #for j in range(self.nr_vms):
         #   self.model.add_equivalence( self.vm[j], self.model.sum(self.a[i, j] for i in range(self.nr_comps)) >= 1, name="c{0}_vm_allocated".format(j))
        for j in range(self.nr_vms):
            for i in range(self.nr_comps):
                self.model.add_constraint(self.a[i, j] <= self.vm[j], ctname=f"link_a_vm_{i}_{j}")

        #Variables for offers description
        #maxType = len(self.offers_list)
        #self.vmType = {(j): self.model.integer_var(lb=0, ub=maxType, name="vmType{0}".format(j + 1))
        #               for j in range(self.nr_vms)}

        self.nr_offers = len(self.offers_list)
        self.z = {(j, t): self.model.binary_var(name="select_vm{0}_offer{1}".format(j + 1, t + 1))
                  for j in range(self.nr_vms) for t in range(self.nr_offers)}

        # Constrangere: Un VM 'j' poate avea cel mult un tip de oferta.
        # Suma(z[j,t]) == vm[j]. Daca vm[j] e 0, nu se alege nicio oferta. Daca e 1, se alege exact una.
        for j in range(self.nr_vms):
            self.model.add_constraint(
                self.model.sum(self.z[j, t] for t in range(self.nr_offers)) == self.vm[j],
                ctname=f"select_one_type_vm_{j}"
            )

        minProc = min(self.offers_list[t][1] for t in range(self.nr_offers))
        maxProc = max(self.offers_list[t][1] for t in range(self.nr_offers))
        self.ProcProv = {j: self.model.integer_var(lb=0, ub=maxProc, name="ProcProv{0}".format(j + 1))
                         for j in range(self.nr_vms)}

        minMem = min(self.offers_list[t][2] for t in range(self.nr_offers))
        maxMem = max(self.offers_list[t][2] for t in range(self.nr_offers))
        self.MemProv = {j: self.model.integer_var(lb=0, ub=maxMem, name="MemProv{0}".format(j + 1))
                        for j in range(self.nr_vms)}

        minSto = min(self.offers_list[t][3] for t in range(self.nr_offers))
        maxSto = max(self.offers_list[t][3] for t in range(self.nr_offers))
        self.StorageProv = {j: self.model.integer_var(lb=0, ub=maxSto, name="StorageProv{0}".format(j + 1))
                            for j in range(self.nr_vms)}

        maxPrice = max(self.offers_list[t][len(self.offers_list[0]) - 1] for t in range(self.nr_offers))
        self.PriceProv = {j: self.model.integer_var(lb=0, ub=maxPrice, name="PriceProv{0}".format(j + 1))
                          for j in range(self.nr_vms)}

        # Variabila vmType pentru compatibilitate (optional, calculata din z)
        # vmType[j] = sum( (t+1) * z[j,t] )
        self.vmType = {j: self.model.integer_var(lb=0, ub=self.nr_offers, name="vmType{0}".format(j + 1))
                       for j in range(self.nr_vms)}

        for j in range(self.nr_vms):
            self.model.add_constraint(
                self.vmType[j] == self.model.sum((t + 1) * self.z[j, t] for t in range(self.nr_offers)),
                ctname=f"calc_vmType_{j}"
            )
    def _hardware_and_offers_restrictionns(self, scaleFactor):
        """
        Describes the hardware requirements for each component
        :param componentsRequirements: list of components requirements as given by the user
        :return: None
        """

        price_index = len(self.offers_list[0]) - 1

        for j in range(self.nr_vms):
            # Link Price
            self.model.add_constraint(
                self.PriceProv[j] == self.model.sum(self.z[j, t] * int(self.offers_list[t][price_index])
                                                    for t in range(self.nr_offers)),
                ctname=f"set_price_{j}"
            )

            # Link CPU
            self.model.add_constraint(
                self.ProcProv[j] == self.model.sum(self.z[j, t] * int(self.offers_list[t][1])
                                                   for t in range(self.nr_offers)),
                ctname=f"set_cpu_{j}"
            )

            # Link Memory
            self.model.add_constraint(
                self.MemProv[j] == self.model.sum(self.z[j, t] * int(self.offers_list[t][2])
                                                  for t in range(self.nr_offers)),
                ctname=f"set_mem_{j}"
            )

            # Link Storage
            self.model.add_constraint(
                self.StorageProv[j] == self.model.sum(self.z[j, t] * int(self.offers_list[t][3])
                                                      for t in range(self.nr_offers)),
                ctname=f"set_storage_{j}"
            )

        for k in range(self.nr_vms):
            self.model.add_constraint(ct=self.model.sum(self.a[i, k] * (self.problem.componentsList[i].HC)
                                                        for i in range(self.nr_comps)) <= self.ProcProv[k],ctname="c_hard_cpu")
            self.model.add_constraint(ct=self.model.sum(self.a[i, k] * (self.problem.componentsList[i].HM)
                                                        for i in range(self.nr_comps)) <= self.MemProv[k], ctname="c_hard_mem")
            self.model.add_constraint(ct=self.model.sum(self.a[i, k] * (self.problem.componentsList[i].HS)
                                                        for i in range(self.nr_comps)) <= self.StorageProv[k],ctname="c_hard_storage")

    def _same_type(self, var, vm_id):
        # Lista pentru a stoca variabilele auxiliare "match" pe fiecare tip de oferta
        matches = []

        for t in range(self.nr_offers):
            # Definim o variabila auxiliara binara: both_t
            # both_t = 1 DOAR DACA vm_id are tipul t SI vm_id+1 are tipul t
            both_t = self.model.binary_var(name=f"match_{vm_id}_{t}")

            z1 = self.z[vm_id, t]
            z2 = self.z[vm_id + 1, t]

            # Liniarizarea standard a operatorului AND (produs de binare):
            # 1. both_t nu poate fi 1 daca z1 e 0
            self.model.add_constraint(both_t <= z1)
            # 2. both_t nu poate fi 1 daca z2 e 0
            self.model.add_constraint(both_t <= z2)
            # 3. both_t trebuie sa fie 1 daca ambele sunt 1
            self.model.add_constraint(both_t >= z1 + z2 - 1)

            matches.append(both_t)

        self.model.add_constraint(var == self.model.sum(matches))

    def _get_solution_vm_type(self):
        vm_types = []
        for index, var in self.vmType.items():
            vm_types.append(var.solution_value)
        return vm_types
