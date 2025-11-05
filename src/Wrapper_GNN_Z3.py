import sys

from Wrapper_GNN import Wrapper_GNN
from Wrapper_Z3 import Wrapper_Z3


class Wrapper_GNN_Z3:
    def __init__(self, model_path="../Models/GNNs/oryx2_model_RGCN_1000_samples_100_epochs_128_batchsize.pth", symmetry_breaker="None"):
        print("in Wrapper_GNN_Z3")
        self.gnn_predictor = Wrapper_GNN(model_path=model_path)
        self.symmetry_breaker = symmetry_breaker

    def solve(self, application_model_json, offers_json, mode="gnn"):
        z3_solver = Wrapper_Z3(symmetry_breaker=self.symmetry_breaker)
        if mode == "gnn":
            print("mode = gnn")
            prediction = self.gnn_predictor.solve(application_model_json, offers_json)
            solution = z3_solver.solve(application_model_json, offers_json, prediction=prediction, out=True)
        elif mode == "sim":
            print("mode = sim")
            sim_perfect_prediction = z3_solver.solve(application_model_json, offers_json, out=False)
            solution = z3_solver.solve(application_model_json, offers_json, prediction_sim=sim_perfect_prediction,
                                       out=False)
        elif mode == "init":
            print("mode = init")
            prediction_init = self.gnn_predictor.solve(application_model_json, offers_json)
            print("prediction_init", prediction_init)
            solution = z3_solver.solve(application_model_json, offers_json, prediction_init=prediction_init, out=True)
        else:
            print("No mode selected. Exiting")
            sys.exit()
        print(solution["output"]["time (secs)"])
        print(solution["output"]["min_price"])

        return solution