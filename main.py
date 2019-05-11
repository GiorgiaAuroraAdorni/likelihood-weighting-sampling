from util import likelihood_weighting
from datetime import datetime
import pandas as pd
import numpy as np

start_time = datetime.now()

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

print("Probability table Rush Hour\n", net[2][nodes[0]])
print("\nProbability table Bad Weather\n", net[2][nodes[1]])
print("\nProbability table Accident\n", net[2][nodes[2]])
print("\nProbability table Traffic Jam\n", net[2][nodes[3]])

# evidence
e = {nodes[2]: True}
print("\nEvidence: ", list(e.keys())[0], "= ", e[nodes[2]])

# query
q = [nodes[3], True]
print("\nQuery: ", q[0], "= ", q[1])

normalized = likelihood_weighting(q, e, net, 1000000)

likelihood = pd.DataFrame({True: [normalized[1]], False: [normalized[0]]})

print("\nLikelihood: \n", likelihood)

if q[1]:
    print("\nP(", q[0], "=", q[1], "|", list(e.keys())[0], "=", e[nodes[2]], ")=", normalized[1])
else:
    print("\nP(", q[0], "=", q[1], "|", list(e.keys())[0], "=", e[nodes[2]], ") =", normalized[1])

print("\n\nTime to complete the execution: ", datetime.now() - start_time, " seconds.")
