import numpy as np

prediction_list = np.array([])
for i in range(3):
    accuracy = np.random.rand()
    prediction_list = np.append(prediction_list, accuracy)

print(prediction_list)
















# import matplotlib.pyplot as plt

# import numpy as np
#
# np.random.seed(19680801)
#
#
# fig, ax = plt.subplots()
# for color in ['tab:blue', 'tab:orange', 'tab:green']:
#     n = 750
#     x, y = np.random.rand(2, n)
#     scale = 200.0 * np.random.rand(n)
#     ax.scatter(x, y, c=color, s=scale, label=color,
#                alpha=0.3, edgecolors='none')
#
# ax.legend()
# ax.grid(True)
#
# plt.show()