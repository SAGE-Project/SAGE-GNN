import json
import dgl
import torch
from Wrapper_Z3 import Wrapper_Z3
from dgl_graph import DGLGraph
from trainRGCN import get_graph_data, to_assignment_matrix, count_matches_and_diffs


class Wrapper_GNN:
    def __init__(self, model_path="/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Models/GNNs/Models_20_7_Datasets-improved-Gini/Models_20_7_Wordpress-improved-Gini/model_RGCN_5636_samples_100_epochs_1024_batchsize.pth"):
        print("in Wrapper_GNN")
        # load pre-existing trained model
        self.model = torch.load(model_path)
        # set the model to evaluation mode
        self.model.eval()

    def solve(self, application_model_json, offers_json):
        # Obtain data in required form (ignore solution)
        # TODO: do not use solve here as it is not really needed
        app_json = Wrapper_Z3().solve(application_model_json, offers_json, out=False)
        # Transform into graph data structure
        print("app_json ", app_json)
        graph = get_graph_data(app_json, app_json["application"])
        # Transform into required DGL graph structure
        dataset = DGLGraph(graph)
        #dgl_graph = dataset[0].to('cuda')
        dgl_graph = dataset[0]

        # create empty lists to store the predictions and true labels
        y_pred = []
        y_true = []

        dec_graph = dgl_graph['component', :, 'vm']
        print("dec_graph", dec_graph)

        edge_label = dec_graph.edata[dgl.ETYPE]
        print("edge_label", edge_label)
        comp_feats = dgl_graph.nodes['component'].data['feat']
        print("comp_feats", comp_feats)
        vm_feats = dgl_graph.nodes['vm'].data['feat']
        print("vm_feats", vm_feats)
        node_features = {'component': comp_feats, 'vm': vm_feats}
        print("node_features", node_features)
        with torch.no_grad():
            logits = self.model(dgl_graph, node_features, dec_graph)
        pred = logits.argmax(dim=-1)
        y_pred.append(pred)
        # Oryx = 10
        # Wordpress = 5
        # SecureWeb, Secure Billing = 5
        print("in Wrapper_GNN assignment_pred")
        assignment_pred = to_assignment_matrix(dgl_graph, dec_graph, pred, 5)
        print("in Wrapper_GNN assignment_actual")
        #assignment_actual = to_assignment_matrix(dgl_graph, dec_graph, edge_label, 5)
        #matches, diffs = count_matches_and_diffs([element for row in assignment_pred for element in row],
        #                                         [element for row in assignment_actual for element in row])
        #print(f"{matches} values match; {diffs} don't")
        print(f"Prediction {assignment_pred}")
        #print(f"Actual     {assignment_actual}")
        return assignment_pred