from libnrl.lle import LLE
from libnrl.hope import HOPE
from libnrl.node2vec import Node2vec
from libnrl.classify import Classifier, read_node_label
from sklearn.linear_model import LogisticRegression
# model = LLE(d=2)
# model.build_graph("/home/alan/Projects/VsCode/OpenNE/data/karate/karate.edgelist")
from libnrl import graph
g = graph.Graph()
g.read_edgelist("/home/alan/Projects/VsCode/OpenNE/data/wiki/Wiki_edgelist.txt")
# model = LLE(graph= g, d=4)
model = HOPE(graph=g, d=128)
# model = Node2vec(graph=g, path_length=80, num_paths=10, dim=128, p=0.25, q=0.25)
# Y, t = model.learn_embedding(is_weighted=True, no_python=True)
# print(model.vectors)
# model.save_embeddings("vec_out.txt")
#   # print(model._method_name+':\n\tTraining time: %f' % (time() - t1))
#   # Evaluate on graph reconstruction
# MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(
#     model._graph, model, Y, None)
# # Visualize
# viz.plot_embedding2D(model.get_embedding(),
#                       di_graph=model._graph, node_colors=None)
# plt.show()
# import numpy as np
# x = np.load("/home/alan/Projects/VsCode/OpenNE/src/out.vec.1.npy")
# def vectors(x):
#   vectors = {}
#   for i in range(len(x)):
#     vectors[str(i)] = x[i, :]
#   return vectors
vectors = model.vectors
X, Y = read_node_label("/home/alan/Projects/VsCode/OpenNE/data/wiki/wiki_labels.txt")
clf_ratio = 0.5
print("Training classifier using {:.2f}% nodes...".format(clf_ratio*100))
clf = Classifier(vectors=vectors, clf=LogisticRegression())
clf.split_train_evaluate(X, Y, clf_ratio)
# print(model.getAdj())