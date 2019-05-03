# Input

probWtrue = 0.05

probHtrue = 0.2

probAtrue_W = [ # W  Atrue
                [ 1, 0.1 ],
                [ 0, 0.3 ]
              ]

probJtrue_HWA = [# H  W  A  Jtrue
                 [ 1, 1, 1, 0.95 ],
                 [ 1, 1, 0, 0.95 ],
                 [ 1, 0, 1, 0.95 ],
                 [ 1, 0, 0, 0.95 ],
                 [ 0, 1, 1, 0.5  ],
                 [ 0, 1, 0, 0.3  ],
                 [ 0, 0, 1, 0.6  ],
                 [ 0, 0, 0, 0.1  ]
                ]

bn = [probWtrue, probHtrue, probAtrue_W, probJtrue_HWA]
# def weighted_sample(bn, e):
#
#     w = 1
#     x
#
#     for var in X:
#         if var #(is an evidence variable with value x in e)
#             w = w * #P(var = x | parents(var))
#         else
#             x = #random sample from P(var | parents(var))
#
#
#
#     return x, w
#
#
# def likelihood_weighting(X, e, bn, N):
#     """
#     :param X: query variable
#     :param e: observed values fro variables E
#     :param bn: a bayesian network specifying joint distribution P(X1, ..., Xn)
#     :param N: the total number of samples to be generated
#     :return: an estimate of P(X | e)
#
#     Local variables:
#     - weights: a vector of weighted count fro eac value of X, initialy zero
#     """
#
#     weights = list()
#
#     for i in range (1, N):
#         x, w = weighted_sample(bn, e)
#         weights[x] = weights[x] + w
#         # x is the value of X in x
#
#
#     # normalized = normalize(weights)
#
#     return normalized
