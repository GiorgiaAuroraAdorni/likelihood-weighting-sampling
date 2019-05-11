import numpy as np
import copy


def weighted_sample(net, evidence):
    """
    :param net: a list composed of:
                - the list of the nodes of the net in topological order,
                - a matrix that contains the parents of each nodes
                - the bayesian network specifying joint distribution P(X1, ..., Xn)
    :param evidence: observed values for variables E

    :return: sample, weight
    """

    nodes, parents, bn = net

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

        if n in evidence:  # (is an evidence variable with value x in e)
            probability = bn[n].loc[acc, sample[n]]

            weight = weight * probability  # weight * P(n=sample[n] | parents(var))

        else:
            random = np.random.random_sample()  # random sample from P(var | parents(var))
            probability = bn[n].loc[acc, True]

            if random <= probability:
                sample[n] = True
            else:
                sample[n] = False

    return sample, weight


def likelihood_weighting(query, evidence, net, n_sample):
    """
    :param query: query variable
    :param evidence: observed values for variables E
    :param net: a list composed of:
                - the list of the nodes of the net in topological order,
                - a matrix that contains the parents of each nodes
                - the bayesian network specifying joint distribution P(X1, ..., Xn)
    :param n_sample: the total number of samples to be generated
    :return: an estimate of P(X | e)

    Local variables:
    - weights: a vector of weighted count for each value of X, initialy zero
    """

    weights = [0, 0]

    print("\nGenerating ", n_sample, " samples...")

    for i in range(n_sample):
        sample, weight = weighted_sample(net, evidence)

        if not sample[query[0]]:
            weights[0] = weights[0] + weight
        else:
            weights[1] = weights[1] + weight

    return weights/np.sum(weights)
