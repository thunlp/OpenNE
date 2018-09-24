import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import os

LOG_DIR = 'log'

def fetch_data(path):
    from sklearn.datasets import fetch_20newsgroups
    categories = ['comp.graphics', 'rec.sport.baseball', 'talk.politics.guns']
    dataset = fetch_20newsgroups(path, categories=categories)
    return dataset


def text_to_graph(text):
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import kneighbors_graph

    # use tfidf to transform texts into feature vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text)

    # build the graph which is full-connected
    N = vectors.shape[0]
    mat = kneighbors_graph(vectors, N, metric='cosine', mode='distance', include_self=True)
    mat.data = 1 - mat.data  # to similarity

    g = nx.from_scipy_sparse_matrix(mat, create_using=nx.Graph())

    return g

dataset = fetch_data('data')

if not os.path.exists(LOG_DIR+"/network.edgelist"):
    import networkx as nx
    graph = text_to_graph(dataset.data)
    graph = graph.to_directed()
    nx.write_edgelist(graph, LOG_DIR+"/network.edgelist", data=['weight'])

labels = dataset.target

############# CHANGE YOUR CODE HERE #############
from openne.lap import LaplacianEigenmaps
from openne.graph import Graph
from openne import node2vec
g = Graph()
g.read_edgelist(filename=LOG_DIR+"/network.edgelist", weighted=True,
                directed=True)

model = LaplacianEigenmaps(g)
vectors = model.vectors
embeddings = np.zeros((g.G.number_of_nodes(), 128))
for i, embedding in vectors.items():
    embeddings[int(i), :] = embedding
# embeddings = model.num_embeddings  # numpy array
########################## END ############################

# save embeddings and labels
emb_df = pd.DataFrame(embeddings)
emb_df.to_csv(LOG_DIR + '/embeddings.tsv', sep='\t', header=False, index=False)

lab_df = pd.Series(labels, name='label')
lab_df.to_frame().to_csv(LOG_DIR + '/node_labels.tsv', index=False, header=False)

# save tf variable
embeddings_var = tf.Variable(embeddings, name='embeddings')
sess = tf.Session()

saver = tf.train.Saver([embeddings_var])
sess.run(embeddings_var.initializer)
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)

# configure tf projector
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = 'embeddings'
embedding.metadata_path = 'node_labels.tsv'

projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

# type "tensorboard --logdir=log" in CMD and have fun :)