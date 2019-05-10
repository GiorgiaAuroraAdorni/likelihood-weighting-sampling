from util import likelihood_weighting
import pandas as pd
import numpy as np

# Set the nodes in topological order H, W, A, J
nodes = np.array(["H", "W", "A", "J"])

probH = pd.DataFrame({True: [0.2], False: [0.8]})

probW = pd.DataFrame({True: [0.05], False: [0.95]})

probA_W = pd.DataFrame({True: [0.3, 0.1], False: [0.7, 0.9]})

probJ_HWA = pd.DataFrame({True: [0.1, 0.6, 0.3, 0.5, 0.95, 0.95, 0.95, 0.95],
                          False: [0.9, 0.4, 0.7, 0.5, 0.05, 0.05, 0.05, 0.05]})

adj = np.array([[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 1, 1, 0]],
               dtype=bool)

net = [nodes, adj, {nodes[0]: probH, nodes[1]: probW, nodes[2]: probA_W, nodes[3]: probJ_HWA}]

# evidence
e = {nodes[2]: True}

# query
q = [nodes[3], True]

normalized = likelihood_weighting(q, e, net, 1000)

print("Normalized: ", normalized)

if q[1]:
    print("\nThe probability that ", q[0], " is ", q[1], " given the evidence ", e,  " is ", normalized[1])
else:
    print("The probability that ", q[0], " is ", q[1], " given the evidence ", e, " is ", normalized[0])
