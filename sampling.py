import pandas as pd
import numpy as np
# Set the nodes in the topologic order H, W, A, J
nodes = np.array(["H", "W", "A", "J"])

probH = pd.DataFrame({True: [0.2], False: [0.8]})

probW = pd.DataFrame({True: [0.05], False: [0.95]})

probA_W = pd.DataFrame({True: [0.3, 0.1], False: [0.7, 0.9]})

probJ_HWA = pd.DataFrame({True: [0.1, 0.6, 0.3, 0.5, 0.95, 0.95, 0.95, 0.95],
                          False: [0.9, 0.4, 0.7, 0.05, 0.05, 0.05, 0.05, 0.5]})

net = {nodes[0]: probH, nodes[1]: probW, nodes[2]: probA_W, nodes[3]: probJ_HWA}


parents = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [1, 1, 1, 0]], dtype=bool)

#     P(n=sample[n] | parents(var))
# bn = [probWtrue, probHtrue, probAtrue_W, probJtrue_HWA]

# print(bn)
e = {nodes[1]: True, nodes[3]: False}
import copy


def weighted_sample(bn, parents, evidence):
    # suppose that the evidence is something like J=F, W=T and the query H=T
    # we want to calculate P(H=T|J=F, W=T)

    weight = 1
    sample = copy.deepcopy(evidence)

    for i, n in enumerate(nodes):
        par = nodes[parents[i]]
        parents_value = list()

        for p in par:
            parents_value.append(sample[p])

        acc = 0

        for j, el in enumerate(parents_value):
            exp = len(parents_value) - j - 1

            acc = el * (2**exp)

        if n in evidence: # (is an evidence variable with value x in e)
            probability = bn[n].loc[acc, sample[n]]

            weight = weight * probability  # weight * P(n=sample[n] | parents(var))

        else:
            random = round(np.random.random_sample(), 2)  #  random sample from P(var | parents(var))
            probability = bn[n].loc[acc, True]

            if random <= probability:
                sample[n] = True
            else:
                sample[n] = False

    return sample, weight


print(weighted_sample(net, parents, e))

#
