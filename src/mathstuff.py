import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def add_to(set, item, debug=False):
    if item not in set:
        set.append(item)
        return True
    return False



# tab = ["a", "b", "c", "d"]
# result = []
# for a in tab:
#     for b in tab:
#         test = sorted([a, b])
#         if add_to(result, test):
#             print(test[0] + test[1])
#     print()
# print("results", len(result))
# exit(0)

l = 6

K_l_0 = [0]
for n in range(0,l+1):
    K_l_0.append(math.pow(2,n))
    K_l_0.append(-math.pow(2,n))
K_l_0 = sorted(K_l_0)
KK_l_0 = [K_l_0]
print("KK_l_0", len(KK_l_0), KK_l_0)

# BB_l_1 = []
# for K_i in [K_l_0]:
#     for K_j in [K_l_0]:
#         new_b_set = []
#         for k_i in K_i:
#             for k_j in K_j:
#                 add_to(new_b_set, k_j + k_i)
#                 add_to(new_b_set, -k_j + k_i)
#                 add_to(new_b_set, k_j + -k_i)
#                 add_to(new_b_set, -k_j + -k_i)
#         new_b_set = sorted(new_b_set)
#         BB_l_1.append(new_b_set)
#
# print("BB_l_1", BB_l_1)
# exit(0)



BB_l_1 = []
for B in KK_l_0:
    for ka in B:
        for k_ab in B:
            add_to(BB_l_1, [1, ka + k_ab])
BB_l_1 = sorted(BB_l_1)
print("BB_l_1", BB_l_1)


KK_l_1 = []
for B in BB_l_1:
    newk = [0]
    for n in range(0, l+1):
        for b in B:
            nvalue = max(min(b*math.pow(2,n), math.pow(2,l)), -math.pow(2,l))
            add_to(newk, nvalue)
            add_to(newk, -nvalue)
            # nvalue = -nvalue
            # if nvalue not in newk:
            #     newk.append(nvalue)
    newk = sorted(newk)
    if 0 not in newk:
        print(B, "results in", newk)
    KK_l_1.append(newk)
KK_l_1 = np.unique(KK_l_1)
KK_l_1 = sorted(KK_l_1)

print("KK_l_1", KK_l_1)

# BB_l_2 = []
# for B in KK_l_0:
#     for ka in tqdm(B):
#         for k_ab in B:
#             for kd in B:
#                 for k_cd in B:
#                     add_to(BB_l_2, [1, ka + k_ab, k_cd + kd])
BB_l_2 = []
for B in KK_l_0:
    for b_i, ka in tqdm(enumerate(B)):
        for b_i_kb in range(b_i, len(B)):
            k_ab = ka + B[b_i_kb]
            if k_ab in K_l_0:
                continue
            for b_j, kd in enumerate(B):
                for b_j_kc in range(b_j, len(B)):
                    k_cd = kd + B[b_j_kc]
                    if k_ab < k_cd:
                        BB_l_2.append([1, k_ab, k_cd])
                        #add_to(BB_l_2, [1, k_ab, k_cd])
for B in KK_l_0:
    for ka in tqdm(B):
        for B1 in KK_l_1:
            for kb in B1:
                BB_l_2.append([1, ka + kb])
                #add_to(BB_l_2, [1, ka + kb])
BB_l_2 = np.unique(BB_l_2)
BB_l_2 = sorted(BB_l_2)
print("BB_l_2", len(BB_l_2))

KK_l_2 = []
for B in tqdm(BB_l_2):
    newk = [0]
    for n in range(0, l+1):
        for b in B:
            nvalue = max(min(b*math.pow(2,n), math.pow(2,l)), -math.pow(2,l))
            newk.append(nvalue)
            newk.append(-nvalue)
            # add_to(newk, nvalue)
            # add_to(newk, -nvalue)
            # nvalue = -nvalue
            # if nvalue not in newk:
            #     newk.append(nvalue)
    newk = sorted(np.unique(newk))
    if 0 not in newk:
        print(B, "results in", newk)
    KK_l_2.append(newk)
    #add_to(KK_l_2, newk)
KK_l_2 = np.unique(KK_l_2)
KK_l_2 = sorted(KK_l_2)

print("KK_l_2", len(KK_l_2))


def flatten(list):
    result = []
    for tab in list:
        result.extend(tab)
    result = np.array(result)
    result = result[result>0]
    return np.unique(result).tolist()

all_cost_0 = flatten(KK_l_0)
all_cost_1 = flatten(KK_l_1)
all_cost_2 = flatten(KK_l_2)


def distinct(list, to_remove):
    for rem in to_remove:
        if rem in list:
            list.remove(rem)
    return list

all_cost_2 = distinct(all_cost_2, all_cost_1)
all_cost_1 = distinct(all_cost_1, all_cost_0)

print(all_cost_0)
print(all_cost_1)
print(all_cost_2)

numbers = []
adders = []
colors = ["green", "orange", "red"]

for num in range(1,33):
    if num in all_cost_0:
        adders.append(0)
        numbers.append(num)
    elif num in all_cost_1:
        pass
    elif num in all_cost_2:
        pass
    else:
        print("WOW", num)

def filter(adders):
    result = []
    for adder in adders:
        result.append(colors[adder])
    return result

from matplotlib.pyplot import figure
import matplotlib
figure(figsize=(19, 9))
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 26}

matplotlib.rc('font', **font)

plt.bar(numbers, np.array(adders)+1, width=0.5, color=filter(adders))  # marker="o",color=toFilter(flex_qat_mse, "#ffd700"), edgecolor="red", linewidth=toColor(flex_qat_stable, flex_qat_mse),
        #label="qat"
plt.ylim([0,3])
plt.yticks([1,2,3,4],[0,1,2,3])
plt.ylabel("adder-cost")
plt.xlabel("coefficients")
plt.xticks([1,2,4,8,16,32])
plt.xlim([0,33])
plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.1)
plt.savefig("zahlenstrahl_cost0.svg")
#plt.show()

