import json
import random
from gnn import Model, RGCN, HeteroMLPPredictor
from Wrapper_GNN import Wrapper_GNN
from Wrapper_GNN_Z3 import Wrapper_GNN_Z3
from Wrapper_Z3 import Wrapper_Z3

with open("../Models/json/SecureWebContainer.json", "r") as file:
    application = json.load(file)

with open("../Data/json/DigitalOcean.json", "r") as file:
    offers_do = json.load(file)

#1.
# wrapper_z3 = Wrapper_Z3(symmetry_breaker="None")
# print(wrapper_z3.solve(application, offers_do))
#
# #2.
# wrapper_z3 = Wrapper_Z3(symmetry_breaker="FVPR")
# print(wrapper_z3.solve(application, offers_do))

# # 3.
# wrapper_gnn_z3 = Wrapper_GNN_Z3(symmetry_breaker="None")
# print(wrapper_gnn_z3.solve(application, offers_do, mode="gnn"))

# 4.
wrapper_gnn_z3 = Wrapper_GNN_Z3(symmetry_breaker="FVPR")
print(wrapper_gnn_z3.solve(application, offers_do, mode="gnn"))

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