from numpy import unique, argmax, arange
from .graph import permutation_graph
import warnings


def get_connected_from(idx, permutation_g):
    """
    get the ids of the parents of the node idx
    """
    return [
        permutation_g.naming[k]
        for k, l in permutation_g.edges.items()
        if permutation_g.index2name(idx) in l
    ]


def get_perm_dict(permutation_g):
    """
    get the permutation dict
    """
    perm_dict = {}
    i = -1
    for node in permutation_g.naming.values():
        p = get_connected_from(node, permutation_g)
        j = i
        i += 1
        for p_ in p:
            if p_ in perm_dict.keys():
                j = perm_dict[p_]
                i = max(perm_dict.values()) + 1
            perm_dict[p_] = j
    # add last node (output) with no perm
    perm_dict[node] = None
    return perm_dict


def remove_nodes_from_perm_dict(nodes_id, perm_dict):
    """
    removes the permutation associated with the nodes as well as other nodes using the same permutation
    """
    for node_id in nodes_id:
        if not node_id in perm_dict.keys():
            warnings.warn(
                "Node_id {} cannot be removed, this node is not in the graph".format(
                    node_id
                )
            )
            continue
        perm_id = perm_dict[node_id]
        list_to_remove = [
            n_id for n_id in perm_dict.keys() if perm_dict[n_id] == perm_id
        ]
        for node in list_to_remove:
            perm_dict[node] = None
    return perm_dict


def re_id_perm(perm_dict):
    """
    fill in the gaps in the perm_ids
    """
    list_perm_id = unique([p_id for p_id in perm_dict.values() if p_id is not None])
    if len(list_perm_id) == 0:
        # no permutation left
        return perm_dict
    first_gap = argmax((list_perm_id != arange(len(list_perm_id))).astype(int))
    if list_perm_id[first_gap] == 0:
        # no re_id needed
        return perm_dict
    for n_id, p_id in perm_dict.items():
        if p_id is not None and p_id > first_gap:
            # fill the gap
            perm_dict[n_id] = p_id - 1
    # if there is more that one gap :
    return re_id_perm(perm_dict)


def solve_graph(model, input, remove_nodes=list()):
    permutation_g, parameter_map = permutation_graph(model, input, False, [], [])
    perm_dict = get_perm_dict(permutation_g)
    # remove the nodes disabled by the user
    perm_dict = remove_nodes_from_perm_dict(remove_nodes, perm_dict)
    # fill the gaps
    perm_dict = re_id_perm(perm_dict)
    n_perm = len(unique([p_id for p_id in perm_dict.values() if p_id is not None]))
    if n_perm == 0:
        warnings.warn("No permutation left in graph, you might let more nodes free")
    return perm_dict, n_perm, permutation_g, parameter_map
