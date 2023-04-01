from numpy import unique,argmax,arange
from .graph import permutation_graph
import warnings

def get_connected_from(idx,permutation_g):
    """
    get the ids of the parents of the node idx
    """
    return [permutation_g.naming[k] for k,l in permutation_g.edges.items() if permutation_g.index2name(idx) in l]

def get_perm_dict(permutation_g):
    """
    get the permutation dict
    """
    perm_dict = {}
    i = 0
    for node in permutation_g.naming.values():
        p = get_connected_from(node,permutation_g)
        j = i
        i += 1
        if permutation_g.nodes[permutation_g.index2name(node)]["type"] == "MulBackward0":
            # if elementw mult : no perm & disable prev perm
            # in theory we can handle that case (not yet for us)
            perm_dict[node] = None
            for p_ in p :
                perm_dict[p_] = None
        elif permutation_g.nodes[permutation_g.index2name(node)]["type"] == "CatBackward0":
            # if concatenation : no perm but list of perm_id to use
            # sort list to match concat order
#             perm_dict[node] = sorted(p, key=lambda p_id: int(permutation_g.index2name(p_id)),
#                                     )#reverse=True)
            perm_dict[node] = p 
    
            for p_ in p:
                i = max([p_id for p_id in perm_dict.values() if isinstance(p_id,int)]) + 1
                perm_dict[p_] = i
        else :
            for p_ in p:
                if p_ in perm_dict.keys():
                    # p_id already in perm_dict
                    j = perm_dict[p_] # old p_id
                    i = max([p_id for p_id in perm_dict.values() if isinstance(p_id,int)]) + 1 # next new p_id
                perm_dict[p_] = j
                
    # add last output nodes with no perm
    output_nodes = [k for k,v in permutation_g.nodes.items() if v["is_output"]]
    for node in output_nodes:
        name = permutation_g.naming[node]
        perm_dict[name] = None
    return perm_dict

def remove_nodes_from_perm_dict(nodes_id,perm_dict):
    """
    removes the permutation associated with the nodes as well as other nodes using the same permutation
    """
    for node_id in nodes_id:
        if not node_id in perm_dict.keys():
            warnings.warn("Node_id {} cannot be removed, this node is not in the graph".format(node_id))
            continue
        perm_id = perm_dict[node_id]
        list_to_remove = [n_id for n_id in perm_dict.keys() if perm_dict[n_id]==perm_id]
        for node in list_to_remove:
            perm_dict[node] = None
    return perm_dict

def re_id_perm(perm_dict):
    """
    fill in the gaps in the perm_ids
    """
    list_perm_id = unique([p_id for p_id in perm_dict.values() if isinstance(p_id,int)])
    assert  (list_perm_id>=0).all(),"Negative p_id"
    if len(list_perm_id) == 0:
        # no permutation left
        return perm_dict
    first_gap = argmax((list_perm_id != arange(len(list_perm_id))).astype(int))
    if list_perm_id[first_gap] == 0 :
        # no re_id needed 
        return perm_dict
    for n_id,p_id in perm_dict.items():
        if p_id is None:
            continue
        if isinstance(p_id,int) and p_id > first_gap :
            # fill the gap
            perm_dict[n_id] -= 1
    # if there is more that one gap :
    return re_id_perm(perm_dict)

def solve_graph(model,input,remove_nodes=list()):
    permutation_g, parameter_map = permutation_graph(model, input)
    perm_dict = get_perm_dict(permutation_g)
    # remove the nodes disabled by the user
    perm_dict = remove_nodes_from_perm_dict(remove_nodes,perm_dict)
    # fill the gaps
    perm_dict = re_id_perm(perm_dict)
    # remove perm in concat where there is at least one None
    # in theory we can handle that case (not yet for us)
    for k,p_id in perm_dict.items():
        if isinstance(p_id,list):
            if None in [perm_dict[x] for x in p_id]:
                perm_dict[k] = None
                perm_dict = remove_nodes_from_perm_dict(p_id,perm_dict)
    # fill the gaps
    perm_dict = re_id_perm(perm_dict)
    n_perm = len(unique([p_id for p_id in perm_dict.values() if isinstance(p_id,int)]))
    if n_perm == 0:
        warnings.warn("No permutation left in graph, you might let more nodes free")
    return perm_dict,n_perm,permutation_g,parameter_map

