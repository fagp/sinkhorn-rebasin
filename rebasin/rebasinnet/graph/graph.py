from collections import defaultdict
from torchviz import make_dot
import graphviz

import warnings

list_node_without_perm = ["CatBackward0","MulBackward0"]
# these particular cases must be treated individually in the auto_graph.py
# list of perv nodes will be used for thoses cases
list_node_perm = ["ConvolutionBackward0", "AddmmBackward0","MmBackward0"]
# int perm_id
list_node = list_node_perm + list_node_without_perm
# layers where we want to set a node to
list_node_fuse = ["NativeBatchNormBackward0","NativeGroupNormBackward0"]
# layers where we need to use the prev perm 

class graph:
    def __init__(self):
        self.nodes = dict()
        self.edges = defaultdict(list)
        self.naming = dict()

    def add_node(self, name, value, is_output=False, is_param=False):
        if name in self.nodes.keys():
            #already in graph
            warnings.warn("Node {} already in graph".format(name))
        self.nodes[name] = dict(type=value, is_output=is_output, is_param=is_param)
        self.naming[name] = int(len(self.naming))

    def index2name(self, index):
        for key, value in self.naming.items():
            if value == index:
                return key
        return None

    def add_edge(self, from_node, to_node):
        if to_node not in self.edges[from_node]:
            self.edges[from_node].append(to_node)

    def paramid(self, name):
        for key, node in self.nodes.items():
            name_from_type = node["type"].split("\n")[0][1:]
            if node["is_param"] and name == name_from_type:
                return key

    def parents(self, name):
        parents = []
        for key, value in self.edges.items():
            if name in value:
                parents.append(key)
        return parents

    def closer_perm(self, key):
        if self.nodes[key]["type"] in list_node:
            return key
        if self.nodes[key]["type"] in list_node_fuse:
            return key

        child = self.edges[key]
        assert len(child) == 1
        return self.closer_perm(child[0])

    def child_perm(self, key, perms):
        queue = [key]
        visited = [key]
        childs = []
        fused_nodes = []

        notfirst = False
        while queue:
            node = queue.pop(0)
            type_node = self.nodes[node]["type"]

            if (
                node in perms
                and notfirst
                and type_node in list_node
            ):
                childs.append(node)
            else:
                if type_node in list_node_fuse:
                    fused_nodes.append(node)
                for child in self.edges[node]:
                    if child not in visited:
                        queue.append(child)
                        visited.append(child)
            notfirst = True
        return childs, fused_nodes

    def from_dot(self, dot):
        for n in dot.body:
            n = n.strip()
            is_edge = "->" in n

            # node
            if not is_edge:
                idnode, label = n.split(" [label=")
                # remove ]
                label = label[:-1]
                # output node
                is_output = "fillcolor=darkolivegreen1" in label
                label = label.replace(" fillcolor=darkolivegreen1", "")
                # param node
                is_param = "fillcolor=lightblue" in label
                label = label.replace(" fillcolor=lightblue", "")

                self.add_node(idnode, label, is_output, is_param)

            # edge
            else:
                from_node, to_node = n.split(" -> ")
                self.add_edge(from_node, to_node)


def permutation_graph(model, input):
    prev_dev = next(model.parameters()).device
    model.to("cpu")
    if isinstance(input,list):
        input = [x.to("cpu") for x in input]
        y = model(*input)
    else:
        input = input.to("cpu")
        y = model(input)
    dot = make_dot(y, params=dict(model.named_parameters()))
    g = graph()
    g.from_dot(dot)
    # map param name to permutation
    permutation_param = dict()
    for name, param in model.named_parameters():
        key = g.paramid(name)
        if key is not None :
            permutation_param[name] = g.closer_perm(key)

    # add concat node
    list_concat_node = []
    for key,v in g.nodes.items():
        if v["type"] in list_node_without_perm :
            list_concat_node.append(key)

    # set is_output
    for k in [k for k,v in g.nodes.items() if v["is_output"]]:
        p = g.parents(k)
        if len(p) == 0 :
            # input is an output
            continue
        assert len(p) == 1 ," more than on parent node to the output shape {}".format(g.nodes[k]["type"])
        g.nodes[p[0]]["is_output"] = True
        if g.nodes[p[0]]["type"] in list_node_fuse:
            for k in [k for k in g.parents(p[0]) if g.nodes[k]["type"] in list_node]:
                g.nodes[k]["is_output"] = True

    permutation_list = list(permutation_param.values()) + list_concat_node
    
    permutation_graph = graph()
    visited = set()

    # construct permutation params graph
    permutation_graph = graph()
    for p in permutation_list:
        if p in visited:
            continue
        if g.nodes[p]["type"] in list_node_fuse:
            continue

        permutation_graph.add_node(
            p, g.nodes[p]["type"], g.nodes[p]["is_output"], g.nodes[p]["is_param"]
        )
        childs, fused_nodes = g.child_perm(p, permutation_list)

        if fused_nodes:
            for k, v in permutation_param.items():
                if v in fused_nodes:
                    permutation_param[k] = p

        for c in childs:
            permutation_graph.add_edge(p, c)

        visited.add(p)
    model.to(prev_dev)
    return permutation_graph, permutation_param
