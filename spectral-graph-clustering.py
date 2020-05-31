import numpy as np
import time
from pathlib import Path

from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt

from pyspark import SparkContext, SparkConf

import networkx as nx   # thư viện để tạo, thao tác, học cấu trúc.... của các mạng phức tạp
import seaborn as sns  # thư viện để trực quan hóa dữ liệu

sns.set()  # set context, style, pallete, font ....

float_formatter = lambda x: "%.6f" % x

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
        2: 'orange',
        3: 'gray',
        4: 'violet',
        5: 'pink',
        6: 'purple',
        7: 'brown',
        8: 'yellow',
        9: 'lime',
        10: 'cyan'
    }
    return switcher.get(label, 'Invalid label')

def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=10)
    # nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.5)


def draw_graph_cluster(G, labels):
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        node_size=10,
        width=0.1,
        alpha=0.5,
        with_labels=False,
    )
    

if __name__ == "__main__":
    base_path = Path(__file__).parent
    file_input = (base_path / "./facebook_combined.txt").resolve() 
    file_output = (base_path / "./output/").resolve()
	# create Spark context with necessary configuration
    sc = SparkContext("local","PySpark Spectral Clustering")

    # read data from text file and split each line into words
    lines = sc.textFile(str(file_input)).map(lambda line: line.strip().split(" ")).collect()

    data = []
    for l in lines:
        data.append(l)
    # print(data[0:10])
    G = nx.Graph()
    # G.add_edges_from([
    #     [1, 2],
    #     [1, 3],
    #     [1, 4],
    #     [2, 3],
    #     [3, 4],
    #     [4, 5],
    #     [1, 5],
    #     [6, 7],
    #     [7, 8],
    #     [6, 8],
    #     [6, 9],
    #     [9, 6],
    #     [7, 10],
    #     [7, 2],
    #     [11, 12],
    #     [12, 13],
    #     [7, 12],
    #     [11, 13]
    # ])

    


    G.add_edges_from(data)
    plt.figure(1)
    draw_graph(G)

    W = nx.adjacency_matrix(G)
    # print(W.todense())

    # degree matrix
    D = np.diag(np.sum(np.array(W.todense()), axis=1))
    # print('degree matrix:')
    # print(D)# laplacian matrix
    L = D - W
    # print('laplacian matrix:')
    # print(L)

    eigval, eigvec = np.linalg.eig(L)# eigenvalues
    # print('eigenvalues:')
    # print(eigval)# eigenvectors
    # print('eigenvectors:')
    # print(eigvec)

    print(eigval.real)
    # print(eigvec.real)


    i = np.where(eigval.real < 0.1)[0]
    # print(i)

    fig = plt.figure(2)

    ax1 = plt.subplot(321)
    ax1.plot(eigval.real)
    ax1.title.set_text('eigenvalues')
    
    ax2 = plt.subplot(322)
    ax2.plot(eigvec[:, i[0]].real)
    ax2.title.set_text('1st eigvec')

    ax3 = plt.subplot(323)
    ax3.plot(eigvec[:, i[1]].real)
    ax3.title.set_text('2nd eigvec')

    ax4 = plt.subplot(324)
    ax4.plot(eigvec[:, i[2]].real)
    ax4.title.set_text('3rd eigvec')

    # ax5 = plt.subplot(325)
    # ax5.plot(eigvec[:, i[3]].real)
    # ax5.title.set_text('4th eigvec')

    # ax6 = plt.subplot(326)
    # ax6.plot(eigvec[:, i[4]].real)
    # ax6.title.set_text('5th eigvec')

    fig.tight_layout()
    # plt.show()

    
    U = np.array(eigvec[:, i[0]].real)
    km = KMeans(init='k-means++', n_clusters=10)
    km.fit(U)
    print(km.labels_)

    node_colors = []
    for label in km.labels_:
        node_colors.append(get_node_color(label))

    plt.figure(3)
    draw_graph_cluster(G, node_colors)
    
    
    plt.show()

    # end_time = time.time()
    # print(time_diff_str(start_time, end_time))

    print('Finish!')
    print()

    

    
# start_time = time.time()
# G = nx.Graph()
# G.add_edges_from([
#     [1, 2],
#     [1, 3],
#     [1, 4],
#     [2, 3],
#     [3, 4],
#     [4, 5],
#     [1, 5],
#     [6, 7],
#     [7, 8],
#     [6, 8],
#     [6, 9],
#     [9, 6],
#     [7, 10],
#     [7, 2],
#     [11, 12],
#     [12, 13],
#     [7, 12],
#     [11, 13]
# ])

# draw_graph(G)
# plt.show()

# W = nx.adjacency_matrix(G)
# print(W.todense())

# # degree matrix
# D = np.diag(np.sum(np.array(W.todense()), axis=1))
# print('degree matrix:')
# print(D)# laplacian matrix
# L = D - W
# print('laplacian matrix:')
# print(L)

# eigval, eigvec = np.linalg.eig(L)# eigenvalues
# print('eigenvalues:')
# print(eigval)# eigenvectors
# print('eigenvectors:')
# print(eigvec)


# i = np.where(eigval < 0.5)[0]
# U = np.array(eigvec[:, i[1]])
# km = KMeans(init='k-means++', n_clusters=3)
# km.fit(U)
# print(km.labels_)


# node_colors = []
# for label in km.labels_:
#     node_colors.append(get_node_color(label))

# draw_graph_cluster(G, node_colors)
# plt.show()

# end_time = time.time()
# print(time_diff_str(start_time, end_time))

# print('Finish!')


