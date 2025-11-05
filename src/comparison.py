import json
from trainRGCN import Model, RGCN, HeteroMLPPredictor
from Wrapper_GNN import Wrapper_GNN
from Wrapper_GNN_Z3 import Wrapper_GNN_Z3
from Wrapper_Z3 import Wrapper_Z3
from Wrapper_CPLEX import Wrapper_CPLEX
from utils import utilsSMT


with open("../Models/json/SecureWebContainer.json", "r") as file:
    application = json.load(file)

with open("../Data/json/offers_40.json", "r") as file:
    offers_do = json.load(file)

#1.
# wrapper_z3 = Wrapper_Z3(symmetry_breaker="None")
# print(wrapper_z3.solve(application, offers_do))
#1.
wrapper_CPLEX = Wrapper_CPLEX(symmetry_breaker="None")
print(wrapper_CPLEX.solve(application, offers_do))
#
# #2.
# wrapper_z3 = Wrapper_Z3(symmetry_breaker="FVPR")
# print(wrapper_z3.solve(application, offers_do))

# # 3.
# wrapper_gnn_z3 = Wrapper_GNN_Z3(symmetry_breaker="None")
# print(wrapper_gnn_z3.solve(application, offers_do, mode="gnn"))

# 4.
# wrapper_gnn_z3 = Wrapper_GNN_Z3(symmetry_breaker="FVPR")
# print(wrapper_gnn_z3.solve(application, offers_do, mode="gnn"))

#5.
# wrapper_z3 = Wrapper_Z3(symmetry_breaker="None")
# matrix, VMSpecs = utilsSMT.parse_smt_file("/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Output/SMT-LIB/Wordpress4/Wordpress4_off_20.out")
# print(wrapper_z3.solve(application, offers_do, matrix_init=matrix, VMSpecs_init=VMSpecs, mode="init"))


# # 5.
# wrapper_gnn_z3 = Wrapper_GNN_Z3(symmetry_breaker="None")
# print(wrapper_gnn_z3.solve(application, offers_do, mode="init"))

# obsolete
# wrapper_gnn_z3 = Wrapper_Z3_Unsat(symmetry_breaker="FVPR")
# print(wrapper_gnn_z3.solve(application, offers_do))

# wrapper_gnn = Wrapper_GNN()
# print(wrapper_gnn.solve(application, offers_do))

#wrapper_gnn.solve(application, dict(random.sample(list(offers_do.items()), 19)))

# wrapper_gnn_z3 = Wrapper_GNN_Z3()


# wrapper_gnn_z3 = Wrapper_GNN_Z3(symmetry_breaker="FVPR")
# print(wrapper_gnn_z3.solve(application, offers_do, mode="none"))
#
# wrapper_gnn_z3 = Wrapper_GNN_Z3(symmetry_breaker=None)
# print("after wrapper_gnn_z3")
# print(wrapper_gnn_z3.solve(application, offers_do, mode="gnn"))
#