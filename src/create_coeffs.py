import os
import time

from tqdm import tqdm

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import pydot
import numpy as np

def show_graph(g):
    with open("my.dot", "w") as w:
        g.dot(w)
    (graph,) = pydot.graph_from_dot_file("my.dot")
    graph.del_node('"\\n"')
    graph.write_png('somefile.png')
    os.startfile('somefile.png')



def fixed_unique(array_or_arrays, K_1_additions):
    array_or_arrays, idx = np.unique(array_or_arrays, return_index=True)
    if len(array_or_arrays) == 0:
        return []
    try:
        some_object_iterator = iter(array_or_arrays[0])
    except TypeError as te:
        return [array_or_arrays], [K_1_additions[idx[0]]]
    adds = []
    for i in idx:
        adds.append(K_1_additions[i])
    return array_or_arrays, adds


def gen_K_0_l(l:int):
    result = [0]
    for i in range(l+1):
        result.append(2**i)
        result.append(-(2**i))
    result = sorted(result)
    return result

def gen_K_1_l(l:int):
    K_0 = gen_K_0_l(l)
    B_1_l = []
    B_1_l_additions = []
    for k_a in K_0:
        for k_b in K_0:
            new_ele = np.unique([1, k_a + k_b]).tolist()
            if not new_ele in B_1_l:
                B_1_l.append(new_ele)
                B_1_l_additions.append([[k_a, k_b]])
    # B_1_l = fixed_unique(B_1_l)
    K_1 = []
    K_1_additions = []
    for b_index, B in enumerate(B_1_l):
        new_K = [0]
        for n in range(l+1):
            for b in B:
                new_value = b * (2**n)
                new_value = max(min(2**l, new_value), -2**l)
                new_K.append(new_value)
                new_K.append(-new_value)
        K_1.append(np.unique(new_K).tolist())
        K_1_additions.append(B_1_l_additions[b_index])

    K_1, K_1_additions = fixed_unique(K_1, K_1_additions)
    # K_1 = sorted(K_1, key=lambda x: len(x), reverse=True)
    return K_1, K_1_additions

def gen_K_2_l(l:int):
    K_0 = gen_K_0_l(l)
    K_1, K_1_l_additions = gen_K_1_l(l)
    B_2_l = []
    B_2_l_additions = []
    max_val = 2**l
    print("Iterating over half of", len(K_0))
    for k_a_index, k_a in tqdm(enumerate(K_0)):
        if k_a_index > len(K_0)/2:
            break
        for K_1_addition_item, K_1_item in zip(K_1_l_additions, K_1):
            for k_b in K_1_item:
                new_ele = np.unique([1, k_a + k_b, K_1_addition_item[0][0] + K_1_addition_item[0][1]]).tolist() #
                if not new_ele in B_2_l:
                    B_2_l.append(new_ele)
                    B_2_l_additions.append(K_1_addition_item + [[k_a, k_b]])
        # for k_b in K_0:
        #     if k_a == k_b:
        #         break
        #     if k_a + k_b > max_val or k_a + k_b < -max_val:
        #         break
        #     if k_a + k_b in K_0:
        #         break
        #     for k_c in K_0:
        #         for k_d in K_0:
        #             new_ele = np.unique([1, k_a + k_b, k_c + k_d]).tolist()
        #             if not new_ele in B_2_l:
        #                 print("ASD")
        #                 B_2_l.append(new_ele)
        #                 B_2_l_additions.append([[k_a, k_b], [k_c, k_d]])
    # B_2_l = fixed_unique(B_2_l)
    print(f"GOT ALL FOR B sets for K_2_{l}")
    K_2 = []
    K_2_additions = []
    for b_index, B in tqdm(enumerate(B_2_l)):
        new_K = [0]
        for n in range(l+1):
            for b in B:
                new_value = b * (2**n)
                new_value = max(min(2**l, new_value), -2**l)
                new_K.append(new_value)
                new_K.append(-new_value)
        K_2.append(np.unique(new_K).tolist())
        K_2_additions.append(B_2_l_additions[b_index])

    K_2, K_2_additions = fixed_unique(K_2, K_2_additions)
    # K_2 = sorted(K_2, key=lambda x: len(x), reverse=True)
    return K_2, K_2_additions


def draw_coeff_add_set(coeffs, additions, l, k):
    """
    This is just to debug the coeffs. The plot is not perfect. The plot should never contain nodes that are not connected anywhere.
    This function tries to draw connections between coeffs by itself with the amount of additions and inbetween the wordsize it gets.
    So just for testing!
    Args:
        coeffs:
        additions:
        l:
        k:

    Returns:

    """
    from gvgen import GvGen
    print("got coeffs", coeffs)
    print("with additions", additions)
    g = GvGen()
    coeff_node_map = {}
    # draw all coeffs as nodes
    for coeff in coeffs:
        coeff_node_map[coeff] = g.newItem(str(coeff))
    # draw start node
    coeff_node_map["start"] = g.newItem("Start")

    # draw connections to start node
    K_0 = gen_K_0_l(l)
    for k_0 in K_0:
        if k_0 not in coeffs:
            raise Exception("Not conatining all K_0 coeffs!")
        _ = g.newLink(coeff_node_map["start"], coeff_node_map[k_0])

    # this are all nodes that are not directly direved from the start node (base 2 like 16, 32 ...)
    rest = [x for x in coeffs if x not in K_0]
    rest_done = []
    # draw all additions
    for a, b in additions:
        if a+b in coeff_node_map:
            _ = g.newLink(coeff_node_map[a], coeff_node_map[a+b])
            _ = g.newLink(coeff_node_map[b], coeff_node_map[a+b])
            for r in rest:
                if r / (a+b) == r // (a+b):
                    if a+b != r:
                        _ = g.newLink(coeff_node_map[a+b], coeff_node_map[r])
                    rest_done.append(r)

    # draw all that are wrong!
    missing_connections = [x for x in rest if x not in rest_done]
    for very_wrong in missing_connections:
        g.propertyAppend(coeff_node_map[very_wrong], "fontcolor", "red")
    show_graph(g)

def writeAddFile(coeffs, prefix, l):
    coeffs = sorted(coeffs, key=lambda x: len(x), reverse=True)
    result = ""
    for index, x in enumerate(coeffs):
        result += " |  " + " ".join([str(_) for _ in x])
        if index + 1 < len(coeffs):
            result += "\n"
    filename = f"coeffs/{prefix}Add{l+1}.txt"
    with open(filename, "w") as f:
        f.write(result)


for l in tqdm(range(16)):
    cur = gen_K_0_l(l)
    print(f"K_0_{l} = Got {cur}")
    writeAddFile([cur], "10", l-1)


print()

for l in tqdm(range(16)): #range(16):
    cur, cur_adds = gen_K_1_l(l)
    # for i in range(len(cur)):
    #     draw_coeff_add_set(cur[i], cur_adds[i], l, 1)
    print(f"K_1_{l} = Got {len(cur)} sets: {cur}")
    writeAddFile(cur, "11", l-1)

print()

for l in tqdm(range(16)): # 14,16
    start = time.time()
    cur, cur_adds = gen_K_2_l(l)
    # for i in range(10):
    #     gen_K_2_l(l)
    # for i in range(len(cur)):
    #     draw_coeff_add_set(cur[i], cur_adds[i], l, 2)
    print(f"K_2_{l} = Got {len(cur)} sets: ", "took: ", time.time()-start)
    writeAddFile(cur, "12", l-1)

exit(0)

# Why those coeffs?
# The fully parallel architecture allows some tricks!
# Negative coeffs are for free. We can just subtract rather than add the result in the summation
# Same with zero. Just have one less value for the summation. Easy!
