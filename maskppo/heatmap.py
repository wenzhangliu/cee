import matplotlib.pyplot as plt
import numpy as np


def heatmap(data,row_num=11,col_num=6,figsize=None):
    figure = plt.figure(figsize=figsize)
    matrix = np.zeros((row_num, col_num))
    for k, v in data.items():
        matrix[k[0], k[1]] = v

    # fig, ax = plt.subplots()
    plt.imshow(matrix.transpose(), cmap='hot')

    plt.yticks(np.arange(len(matrix[0])))
    plt.xticks(np.arange(len(matrix)))
    # plt.ylabel(np.arange(0, len(matrix[0])))
    # plt.xlabel(np.arange(0, len(matrix)))

    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            text = plt.text(j, i, int(matrix[j, i]), ha='center', va='center', color='black')

    # plt.colorbar(im)
    # plt.show()
    return figure

if __name__ == '__main__':
    data = {(4, 3): 1728, (4, 4): 1572, (3, 4): 1238, (2, 4): 1183, (2, 3): 913, (3, 3): 1229, (4, 2): 1351, (4, 1): 1562, (3, 1): 1389, (2, 1): 1169, (3, 2): 1234, (2, 2): 1014, (1, 2): 1001, (1, 3): 1132, (1, 4): 1266, (1, 1): 1499}
    fig = heatmap(data)
    fig.show()