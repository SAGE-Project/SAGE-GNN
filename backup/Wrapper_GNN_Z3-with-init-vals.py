from Wrapper_GNN import Wrapper_GNN
from Wrapper_Z3 import Wrapper_Z3


class Wrapper_GNN_Z3:
    def __init__(self, model_path="/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Models/GNNs/ModelsBetterDiversityDatasets/SecureWebContainer/model_RGCN_1000_samples_100_epochs_64_batchsize.pth", symmetry_breaker="None"):
        self.gnn_predictor = Wrapper_GNN(model_path=model_path)
        self.symmetry_breaker = symmetry_breaker

    def solve(self, application_model_json, offers_json, mode="gnn+initv"):
        z3_solver = Wrapper_Z3(symmetry_breaker=self.symmetry_breaker)
        if mode == "gnn+initv":
            print("mode = gnn+initv")
            prediction = self.gnn_predictor.solve(application_model_json, offers_json)
            solution = z3_solver.solve(application_model_json, offers_json, prediction=prediction, out=True, mode="gnn+initv")
        elif mode == "gnn+pseudob":
            print("gnn+pseudob")
            prediction = self.gnn_predictor.solve(application_model_json, offers_json)
            solution = z3_solver.solve(application_model_json, offers_json, prediction=prediction, out=True, mode="gnn+pseudob")
        # elif mode == "sim":
        #     sim_perfect_prediction = z3_solver.solve(application_model_json, offers_json, out=True)
        #     solution = z3_solver.solve(application_model_json, offers_json, prediction_sim=sim_perfect_prediction,
        #                                out=False)
        else:
            print("wrong option!")
        print(solution["output"]["time (secs)"])
        print(solution["output"]["min_price"])

        return solution