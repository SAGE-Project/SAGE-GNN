import dgl
from dgl.data import DGLDataset
import torch
import numpy as np

class DGLGraph(DGLDataset):
    def __init__(self, graph_init):
        self.graph_init = graph_init
        super().__init__(name='dgl_graph')

        #print("??? ", self.graph_init)

    def process(self):
        components_src = []
        components_dest = []
        # print("self.graph_init.nodes", len(self.graph_init.nodes))
        # print("self.graph_init.edges", len(self.graph_init.edges))
        # print("self.graph_init.links", self.graph_init.links)
        for edge in self.graph_init.edges:
            components_src.append(edge.node1.id - 1)
            components_dest.append(edge.node2.id - 1)

        size_components = len([n for n in self.graph_init.nodes if n.get_type() == "component"])
        size_vms = len([n for n in self.graph_init.nodes if n.get_type() == "vm"])
        # print("size_components", size_components)
        # print("size_vms", size_vms)

        link_src = []
        link_dest = []
        unlink_src = []
        unlink_dest = []
        for link in self.graph_init.links:
            if (link.features == 1):
                link_src.append(link.node1.id - 1)
                link_dest.append(link.node2.id - self.graph_init.vm_index_start)
            else:
                unlink_src.append(link.node1.id - 1)
                unlink_dest.append(link.node2.id - self.graph_init.vm_index_start)
        # print("link_src ", link_src)
        # print("link_dest ", link_dest)

        deployed_src = [j for j in range(size_components) for i in range(size_vms)]
        deployed_dest = [j for i in range(size_components) for j in range(size_vms)]

        self.graph = dgl.heterograph({
            ('component', 'conflict', 'component'): (torch.tensor(components_src), torch.tensor(components_dest)), # component linkedCC
            ('component', 'linked', 'vm'): (torch.tensor(link_src), torch.tensor(link_dest)), # component linkedCM
            ('component', 'unlinked', 'vm'): (torch.tensor(unlink_src), torch.tensor(unlink_dest))  # component unlinkedCM
        })
        #print("self", self.graph)

        features_components = [x.features for x in self.graph_init.nodes if x.get_type() == "component"]
        features_vms = [x.features for x in self.graph_init.nodes if x.get_type() == "vm"]

        labels = [1 for x in self.graph_init.nodes]
        edge_features_conflicts = [x.features for x in self.graph_init.edges]

        self.graph.nodes['component'].data['feat'] = torch.from_numpy(np.array(features_components).astype(np.float32))
        self.graph.nodes['vm'].data['feat'] = torch.from_numpy(np.array(features_vms).astype(np.float32))
        self.graph.edges['conflict'].data['feat'] = torch.from_numpy(np.array(edge_features_conflicts).astype(np.float32))

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

def print_dataset(dgl_graph):
    print('\n\nDGL graph dataset')
    print(dgl_graph)
    print("Component nodes")
    print(dgl_graph.nodes('component'))
    print("VM nodes")
    print(dgl_graph.nodes('vm'))
    print("CC LINKS")
    print(dgl_graph.edges(etype='conflict'))
    print("CM LINKS")
    print(dgl_graph.edges(etype='linked'))
    print("CM UNLINKS")
    print(dgl_graph.edges(etype='unlinked'))
    print("Node Features")
    print(dgl_graph.ndata['feat']['component'])
    print(dgl_graph.ndata['feat']['vm'])
    print("Edge Features")
    print(dgl_graph.edata)