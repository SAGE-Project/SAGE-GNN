import os
import json
import dgl
import torch
import time

import numpy as np
from sympy.assumptions.handlers import test_closed_group

from graph import Node, Graph
from dgl_graph import DGLGraph, print_dataset
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import matplotlib.pyplot as plt
from focal_loss.focal_loss import FocalLoss

def read_jsons(path_to_dir):
    all_json_data = []
    index = 0
    for file_name in os.listdir(path_to_dir):
        index = index + 1
        #print(f"DURING DIR READ {index}")
        file_path = f'{path_to_dir}/{file_name}'
        with open(file_path, 'r', encoding='utf-8-sig') as json_file:
            json_data = json.load(json_file)
            json_data['filename'] = file_path
            all_json_data.append(json_data)
    return all_json_data

def without_keys(d, keys):
    return {k: v for k, v in d.items() if k not in keys}

def get_node_features(component, restrictions, max_cpu, max_mem, max_storage, id):
    # normalize values
    cpu = component["Compute"]["CPU"] / max_cpu
    memory = component["Compute"]["Memory"] / max_mem
    storage = component["Storage"]["StorageSize"] / max_storage
    full_deploy = 0
    upper_b = 0
    lower_b = 0
    eq_b = 0
    #print("====")
    node_id = component['id']
    # print(node_id)
    # print(restrictions)
    for res in restrictions:
        if res["type"] == "FullDeployment" and res["alphaCompId"] == node_id:
            full_deploy = 1
        if res["type"] == "UpperBound" and len(res["compsIdList"]) == 1 and node_id in res["compsIdList"]:
            upper_b = 1
        if res["type"] == "LowerBound" and len(res["compsIdList"]) == 1 and node_id in res["compsIdList"]:
            lower_b = 1
        if res["type"] == "EqualBound" and len(res["compsIdList"]) == 1 and node_id in res["compsIdList"]:
            eq_b = 1
    #print(full_deploy, upper_b, lower_b, eq_b)

    return [id/4, cpu, memory, storage, full_deploy, upper_b, lower_b, eq_b]

def get_component_nodes(json_data):
    restrictions = json_data["restrictions"]
    assign = json_data["output"]["assign_matr"]
    # Determine max of each cpu/memory/storage/price for normalization in [0, 1]
    max_cpu, max_mem, max_storage, max_price = 0, 0, 0, 0
    for component in json_data['components']:
        cpu = component["Compute"]["CPU"]
        memory = component["Compute"]["Memory"]
        storage = component["Storage"]["StorageSize"]
        if cpu > max_cpu: max_cpu = cpu
        if memory > max_mem: max_mem = memory
        if storage > max_storage: max_storage = storage

    component_nodes = []
    for id,component in enumerate(json_data['components']):
        features = get_node_features(component, restrictions, max_cpu, max_mem, max_storage,id)
        component_node = Node(component['id'], features, "component")
        component_nodes.append(component_node)
    return component_nodes

def get_vm_nodes(json_data, starting_index, surrogate_result):
    max_cpu, max_mem, max_storage, max_price = 0, 0, 0, 0

    offers = json_data['output']['offers'].values()  # Extract all offers

    # Extract max values correctly
    for vm_specs in offers:
        max_cpu = max(max_cpu, vm_specs["cpu"])
        max_mem = max(max_mem, vm_specs["memory"])
        max_storage = max(max_storage, vm_specs["storage"])
        max_price = max(max_price, vm_specs["price"])

    vm_nodes = []
    idx = 0

    # Normalize and create nodes
    for vm_specs in json_data['output']['offers'].values():
        vm_features = [
            vm_specs["cpu"] / max_cpu if max_cpu else 0,
            vm_specs["memory"] / max_mem if max_mem else 0,
            vm_specs["storage"] / max_storage if max_storage else 0,
            vm_specs["price"] / max_price if max_price else 0,
        ]
        for i in range(surrogate_result):
            #vm_nodes.append(Node(starting_index + idx, vm_features + [(i + 1) / surrogate_result], "vm"))
            vm_nodes.append(Node(starting_index + idx, vm_features, "vm"))
            idx += 1

    return vm_nodes


def get_graph_data(json_data, file_name):
    restrictions = json_data["restrictions"]
    assign = json_data["output"]["assign_matr"]
    # # Determine max of each cpu/memory/storage/price for normalization in [0, 1]
    # max_cpu, max_mem, max_storage, max_price = 0, 0, 0, 0
    # for component in json_data['components']:
    #     cpu = component["Compute"]["CPU"]
    #     memory = component["Compute"]["Memory"]
    #     storage = component["Storage"]["StorageSize"]
    #     if cpu > max_cpu: max_cpu = cpu
    #     if memory > max_mem: max_mem = memory
    #     if storage > max_storage: max_storage = storage
    # for vm_type in json_data['output']['types_of_VMs']:
    #     print("---> ", [vm for vm in json_data['output']['VMs specs'] if list(vm.values())[0]['id'] == vm_type])
    #     if ([vm for vm in json_data['output']['VMs specs'] if list(vm.values())[0]['id'] == vm_type] != []):
    #         vm_specs = [vm for vm in json_data['output']['VMs specs'] if list(vm.values())[0]['id'] == vm_type][0]
    #         cpu = list(vm_specs.values())[0]["cpu"]
    #         memory = list(vm_specs.values())[0]["memory"]
    #         storage = list(vm_specs.values())[0]["storage"]
    #         price = list(vm_specs.values())[0]["price"]
    #         #print("cpu ",cpu)
    #         if cpu > max_cpu: max_cpu = cpu
    #         if memory > max_mem: max_mem = memory
    #         if storage > max_storage: max_storage = storage
    #         if price > max_price: max_price = price

    surrogate_result = 6 # Secure Web Container
        #surrogate_result = 5  # Secure Billing Email
        #surrogate_result = 11  #Oryx2
        #surrogate_result = 8  # Wordpress3
        #surrogate_result = 10  # Wordpress4
        #surrogate_result = 12  # Wordpress5
        #surrogate_result = 13  # Wordpress6
        #surrogate_result = 15  # Wordpress7
        #surrogate_result = 19  # Wordpress8
    component_nodes = get_component_nodes(json_data)
    print("component nodes ", component_nodes)

    vm_nodes = get_vm_nodes(json_data, len(component_nodes) + 1, surrogate_result)
    print("vm_nodes ", len(vm_nodes), vm_nodes)
    return Graph(file_name, component_nodes, vm_nodes, restrictions, assign, json_data["output"], surrogate_result)

# for edge prediction
class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_dims, n_classes):
        super().__init__()
        self.W = nn.Linear(in_dims * 2, n_classes)

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)
        y = self.W(x)
        return {'score': y}

    def forward(self, graph, h):
        # h contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in the node classification
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names) # node prediction
        self.pred = HeteroMLPPredictor(out_features, len(rel_names)) # edge prediction

    def forward(self, g, x, dec_graph):
        h = self.sage(g, x)
        return self.pred(dec_graph, h)

# for node prediction
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

def to_assignment_matrix(graph, dec_graph, tensor, components_nr):
    vms_nr = int(len(tensor) / components_nr)
    #print("tensor ", len(tensor), vms_nr, tensor)
    assign_matrix = [[0 for _ in range(vms_nr)] for _ in range(components_nr)]
    #print("assign_matrix ", len(assign_matrix), len(assign_matrix[0]))

    for dec_ind in range(len(tensor)):
        type = dec_graph.edata[dgl.ETYPE][dec_ind].item()
        orig_index = dec_graph.edata[dgl.EID][dec_ind].item()
        type_edge = graph.etypes[type]
        component = graph.edges(form='uv', order='srcdst', etype=type_edge)[0][orig_index].item()
        vm =        graph.edges(form='uv', order='srcdst', etype=type_edge)[1][orig_index].item()
        value = tensor[dec_ind].item()
        print("component ", component, " vm ", vm, "value ", value)

        if value == 2:
            assign_matrix[component][vm] = 0
        else:
            assign_matrix[component][vm] = 1
    # print("Each row represents a component")
    # # hardcoding
    # # surrogate_result = 6 # Secure Web Container
    # # surrogate_result = 5  # Secure Billing Email
    # # surrogate_result = 11  #Oryx2
    # surrogate_result = 8  # Wordpress3
    # # surrogate_result = 10  # Wordpress4
    # # surrogate_result = 12  # Wordpress5
    # # surrogate_result = 13  # Wordpress6
    # # surrogate_result = 15  # Wordpress7
    # # surrogate_result = 19  # Wordpress8
    # # VM types = 7 from training
    # numb_VM_Types = 500
    # for i in range(components_nr):
    #     print(assign_matrix[i], " ")
    # print("Each matrix is for a component where rows represents a VM and cols types of a VM")
    # for i in range(components_nr):
    #     print("component ", i)
    #     print(np.array(assign_matrix[i]).reshape(surrogate_result, numb_VM_Types))
    #
    # assign_matrix = np.array(assign_matrix)
    #
    # for row in assign_matrix:
    #     for vm in range(surrogate_result):
    #         start = vm * numb_VM_Types
    #         end = start + numb_VM_Types
    #         batch = row[start:end]
    #
    #         if np.sum(batch) > 1:
    #             first_one = np.argmax(batch)  # Get the first occurrence of '1' in the batch
    #             row[start:end] = 0  # Reset the batch
    #             row[start + first_one] = 1  # Keep only the first '1'
    #
    # print("AFTER removal of duplicate 1s: Each row represents a component")
    # for i in range(components_nr):
    #     print(assign_matrix.tolist()[i], " ")
    # print("Each matrix is for a component where rows represents a VM and cols types of a VM")
    # for i in range(components_nr):
    #     print("component ", i)
    #     print(np.array(assign_matrix[i]).reshape(surrogate_result, numb_VM_Types))

    print("assign_matrix ", assign_matrix)

    return assign_matrix

    #return assign_matrix.tolist()

def count_matches_and_diffs(list1, list2):
    # Ensure the lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")

    # Initialize counters
    matches = 0
    diffs = 0

    # Iterate through the lists
    for i in range(len(list1)):
        if list1[i] == 1 and list2[i] == 1:
            matches += 1
        elif list1[i] != list2[i]:
            diffs += 1

    return matches, diffs


def split_into_batches(arr, batch_size):
    return [arr[i:i + batch_size] for i in range(0, len(arr), batch_size)]


if __name__ == '__main__':
    #print("BEFORE DIR READ")
    data = read_jsons('/Users/madalinaerascu/PycharmProjects/SAGE-GNN/Datasets/DatasetsImprovedGini/DsSecureWebContainer_20_7_improved_Gini')
    #print("AFTER DIR READ")

    graphs = []
    index = 0
    samples = 1000
    for json_graph_data in data[:samples]:
        index = index + 1
        #print(f"DURING Graphs construct {index}")
        filename = json_graph_data['filename']
        graphs.append(get_graph_data(json_graph_data, filename))
        print("Example of graph data (from dataset files)", get_graph_data(json_graph_data, filename))

    dgl_graphs = []
    index = 0

    #print("len graphs ", len(graphs))

    for graph in graphs:
        index = index + 1
        #print(f"DURING Graphs dgl convert cuda {index}")
        # print('\n\nGraph Nodes AND Edges')
        # print(graph)
        dataset = DGLGraph(graph)
        dgl_graph = dataset[0] # dataset[1], ..., dataset[len(samples)] is the same
        #print("dgl graph ", dgl_graph)
        #dgl_graph = dgl_graph.to('cuda')
        print("Example of dgl_graph")
        print_dataset(dgl_graph)
        print("End example of dgl_graph")
        dgl_graphs.append(dgl_graph)

    #print("dgl_graphs ", dgl_graphs)

    arr = np.array(dgl_graphs)
    # Calculate the sizes of the three parts
    n = len(arr)
    size1 = int(0.6 * n)
    size2 = int(0.2 * n)

    # Split the array into three parts
    train = arr[:size1].tolist()
    validation = arr[size1:size1 + size2].tolist()
    test = arr[size1 + size2:].tolist()

    model = Model(8, 300, 5, ['binding', 'linked', 'unlinked'])

    print("model = ", model)

    #model = model.to('cuda')
    opt = torch.optim.Adam(model.parameters())
    loss_list = []
    loss_list_valid = []

    acc_training_list = []
    acc_validation_list = []

    class_weights = torch.FloatTensor([0.0, 0.9, 0.1])
    #class_weights = class_weights.to('cuda')
    loss_func = FocalLoss(weights=class_weights, gamma=0.7)
    #loss_func = FocalLoss(weights=class_weights, gamma=0) #when gamma=0 we have cross entropy
    m = torch.nn.Softmax(dim=-1)
    startime = time.time()
    epochs = 100
    for epoch in range(epochs):
        ###########################################################################################################################################################
        ######################################################################## TRAINING #########################################################################
        ###########################################################################################################################################################
        # set the model to train mode
        model.train()

        # create empty lists to store the predictions and true labels
        y_pred = []
        y_true = []

        batch_size = 32
        batched_training = split_into_batches(train, batch_size)
        for train_graphs in batched_training:
            loss_list_batch = []
            total_logits = None
            total_labels = None

            for train_graph in train_graphs:
                #print("train_graph", train_graph)
                #train_graph = train_graph.to('cuda')
                dec_graph = train_graph['component', :, 'vm']
                #print("dec_graph ", dec_graph)
                #dec_graph = dec_graph.to('cuda')
                edge_label = dec_graph.edata[dgl.ETYPE]
                #print("edge label train", edge_label)
                #edge_label = edge_label.to('cuda')
                comp_feats = train_graph.nodes['component'].data['feat']
                #comp_feats = comp_feats.to('cuda')
                vm_feats = train_graph.nodes['vm'].data['feat']
                #vm_feats = vm_feats.to('cuda')

                node_features = {'component': comp_feats, 'vm': vm_feats}
                #print("node_features train ", node_features)

                # print("train graph ", train_graph)
                # print("node features ", node_features)
                # print("dec graph ", dec_graph)
                # the call "model(train_graph, ..." looks weird but this is how is on DGL library: https://www.dgl.ai/dgl_docs/guide/training-edge.html
                # 3 args instead of 2 in DGL website
                logits = model(train_graph, node_features, dec_graph)
                print("logits train after model ", len(logits), len(logits[0]), logits)
                #logits = logits.to('cuda')
                if total_logits == None:
                    total_logits = logits
                    #print("none total_logits", total_logits)
                else:
                    total_logits = torch.cat((total_logits, logits))
                    #print("not none total_logits", total_logits)
                if total_labels == None:
                    total_labels = edge_label
                    #print("none total_labels", total_labels)
                else:
                    total_labels = torch.cat((total_labels, edge_label))
                    #print("not none total_labels", total_labels)

                print("logits.argmax(dim=-1) ", logits.argmax(dim=-1)) # Select the most probable class (binding, linked, unlinked) for each prediction.
                y_pred.append(logits.argmax(dim=-1))
                y_true.append(edge_label)

            # loss = ((total_logits - total_labels) ** 2).mean()
            loss = loss_func(m(total_logits), total_labels)
            loss_list_batch.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        loss_list.append(loss_list_batch[0])
        #print(loss_list_batch[0])

        # concatenate the predictions and true labels into tensors
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        # compute the accuracy of the model on the training set
        accuracy = (y_pred == y_true).float().mean().item()
        acc_training_list.append(accuracy)
        print("Training accuracy:", accuracy)
        ###########################################################################################################################################################
        ####################################################################### VALIDATION ########################################################################
        ###########################################################################################################################################################

        # create empty lists to store the predictions and true labels
        y_pred_val = []
        y_true_val = []

        # set the model to evaluation mode
        model.eval()
        avg_loss = []
        # loop over the validation graphs and compute the predictions and true labels
        for validation_graph in validation:
            dec_graph = validation_graph['component', :, 'vm']

            edge_label = dec_graph.edata[dgl.ETYPE]
            #print("edge label validation", edge_label)
            comp_feats = validation_graph.nodes['component'].data['feat']
            vm_feats = validation_graph.nodes['vm'].data['feat']
            node_features = {'component': comp_feats, 'vm': vm_feats}
            with torch.no_grad():
                logits = model(validation_graph, node_features, dec_graph)
                # loss = ((m(logits) - edge_label) ** 2).mean()
                loss = loss_func(m(logits), edge_label)
                avg_loss.append(loss.item())

            y_pred_val.append(logits.argmax(dim=-1))
            y_true_val.append(edge_label)

        loss_avg = sum(avg_loss) / len(avg_loss)
        loss_list_valid.append(loss_avg)

        # concatenate the predictions and true labels into tensors
        y_pred_val = torch.cat(y_pred_val)
        y_true_val = torch.cat(y_true_val)

        # compute the accuracy of the model on the validation set
        accuracy = (y_pred_val == y_true_val).float().mean().item()
        acc_validation_list.append(accuracy)
        #print("Validation accuracy:", accuracy)

    stoptime = time.time()
    #print("training time ", stoptime - startime)
    #print(loss_list)
    #print("loss_list_valid ", loss_list_valid)


    plt.plot(range(epochs), loss_list, label='Loss Train')
    plt.plot(range(epochs), loss_list_valid, label='Loss Valid')
    # plt.plot(range(epochs), acc_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    #plt.show()
    plt.savefig(f'../plots/temp/loss_RGCN_{samples}_samples_{epochs}_epochs_{batch_size}_batchsize.png')
    plt.close()

    # plt.plot(range(epochs), loss_list, label='Loss')
    plt.plot(range(epochs), acc_training_list, label='Training Accuracy')
    plt.plot(range(epochs), acc_validation_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.show()
    plt.savefig(f'../plots/temp/acc_RGCN_{samples}_samples_{epochs}_epochs_{batch_size}_batchsize.png')
    plt.close()

    ###########################################################################################################################################################
    ######################################################################### TESTING #########################################################################
    ###########################################################################################################################################################
    # create empty lists to store the predictions and true labels
    y_pred = []
    y_true = []

    # set the model to evaluation mode
    model.eval()

    #print("test ", test)

    # loop over the test graphs and compute the predictions and true labels
    #print("len test ", len(test))
    matchesCount = 0
    diffsCount = 0
    for test_graph in test:
        dec_graph = test_graph['component', :, 'vm']
        #print("dec graph", dec_graph)

        edge_label = dec_graph.edata[dgl.ETYPE]
        #print("edge label test", edge_label)
        comp_feats = test_graph.nodes['component'].data['feat']
        vm_feats = test_graph.nodes['vm'].data['feat']
        node_features = {'component': comp_feats, 'vm': vm_feats}
        #print("node features ", node_features)
        with torch.no_grad():
            logits = model(test_graph, node_features, dec_graph)
        pred = logits.argmax(dim=-1) # return to the class

        #print("logits testing", len(logits), logits.shape, logits)
        #print("pred testing", len(pred), pred)

        y_pred.append(pred)
        # last argument is the # of components of the application, Secure Web Container=5
        # last argument is the # of components of the application, Secure Billing Email=5
        # last argument is the # of components of the application, Oryx2=10
        # last argument is the # of components of the application, Wordpress3=5
        print("=====================================")
        print(test_graph)
        print(dec_graph)
        print("pred", pred)
        print("=====================================")
        assingnament_pred   = to_assignment_matrix(test_graph, dec_graph, pred, 5)
        print("assingnament_pred", assingnament_pred)
        assingnament_actual = to_assignment_matrix(test_graph, dec_graph, edge_label, 5)
        print("assingnament_actual", assingnament_actual)
        matches, diffs = count_matches_and_diffs([element for row in assingnament_pred for element in row],
                                                 [element for row in assingnament_actual for element in row])
        matchesCount += matches
        diffsCount += diffs
        #print(f"{matches} values match; {diffs} don't")
        #print(f"Prediction {assingnament_pred}")
        y_true.append(edge_label)
        #print(f"Actual {assingnament_actual}")

    matchesCount /= len(test)
    diffsCount /=len(test)
    print("matchesCount ", matchesCount, "diffsCount", diffsCount)
    # concatenate the predictions and true labels into tensors
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    # compute the accuracy of the model on the validation set
    accuracy = (y_pred == y_true).float().mean().item()
    acc_validation_list.append(accuracy)
    print("Testing accuracy:", accuracy)

    path_to_gnn_model = ''
    gnn_model = 'model_RGCN_{samples}_samples_{epochs}_epochs_{batch_size}_batchsize.pth'
    torch.save(model, f'../Models/GNNs/Models_20_7_Datasets-improved-Gini/Models_20_7_Wordpress-improved-Gini/model_RGCN_{samples}_samples_{epochs}_epochs_{batch_size}_batchsize.pth')