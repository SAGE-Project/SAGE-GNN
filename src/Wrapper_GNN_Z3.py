from Wrapper_GNN import Wrapper_GNN
from Wrapper_Z3 import Wrapper_Z3


class Wrapper_GNN_Z3:
    def __init__(self, model_path="/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Models/GNNs/Models_20_7_Datasets-improved-Gini/Models_20_7_Wordpress-improved-Gini/model_RGCN_5636_samples_100_epochs_1024_batchsize.pth", symmetry_breaker="None"):
        print("in Wrapper_GNN_Z3")
        self.gnn_predictor = Wrapper_GNN(model_path=model_path)
        self.symmetry_breaker = symmetry_breaker

    def solve(self, application_model_json, offers_json, mode="gnn"):
        z3_solver = Wrapper_Z3(symmetry_breaker=self.symmetry_breaker)
        print("in wrapper ", mode)
        if mode == "gnn":
            prediction = self.gnn_predictor.solve(application_model_json, offers_json)
            print("prediction: ", prediction)
            solution = z3_solver.solve(application_model_json, offers_json, prediction=prediction, out=True)
        elif mode == "sim":
            sim_perfect_prediction = z3_solver.solve(application_model_json, offers_json, out=False)
            solution = z3_solver.solve(application_model_json, offers_json, prediction_sim=sim_perfect_prediction,
                                       out=False)
        else:
            solution = z3_solver.solve(application_model_json, offers_json, out=False)
        print(solution["output"]["time (secs)"])
        print(solution["output"]["min_price"])

        return solution