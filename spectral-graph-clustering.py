import numpy as np
import time

from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt

import networkx as nx   # thư viện để tạo, thao tác, học cấu trúc.... của các mạng phức tạp
import seaborn as sns  # thư viện để trực quan hóa dữ liệu

sns.set()  # set context, style, pallete, font ....

float_formatter = lambda x: "%.3f" % x

np.set_printoptions(formatter={'float_kind':float_formatter})

def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 3)
    return str(mins) + " mins and " + str(secs) + " seconds"

def get_node_color(label):
    switcher = {
        0: 'red',
        1: 'blue',
        2: 'yellow'
    }
    return switcher.get(label, 'Invalid label')

def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)


def draw_graph_cluster(G, labels):
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        with_labels=True,
    )
    

start_time = time.time()
G = nx.Graph()
G.add_edges_from([
    [1, 2],
    [1, 3],
    [1, 4],
    [2, 3],
    [3, 4],
    [4, 5],
    [1, 5],
    [6, 7],
    [7, 8],
    [6, 8],
    [6, 9],
    [9, 6],
    [7, 10],
    [7, 2],
    [11, 12],
    [12, 13],
    [7, 12],
    [11, 13]
])

draw_graph(G)
plt.show()

W = nx.adjacency_matrix(G)
print(W.todense())

# degree matrix
D = np.diag(np.sum(np.array(W.todense()), axis=1))
print('degree matrix:')
print(D)# laplacian matrix
L = D - W
print('laplacian matrix:')
print(L)

eigval, eigvec = np.linalg.eig(L)# eigenvalues
print('eigenvalues:')
print(eigval)# eigenvectors
print('eigenvectors:')
print(eigvec)


i = np.where(eigval < 0.5)[0]
U = np.array(eigvec[:, i[1]])
km = KMeans(init='k-means++', n_clusters=3)
km.fit(U)
print(km.labels_)


node_colors = []
for label in km.labels_:
    node_colors.append(get_node_color(label))

draw_graph_cluster(G, node_colors)
plt.show()

end_time = time.time()
print(time_diff_str(start_time, end_time))

print('Finish!')