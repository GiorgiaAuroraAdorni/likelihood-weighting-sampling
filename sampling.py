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
