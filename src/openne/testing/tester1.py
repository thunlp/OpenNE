# pylint: disable=E1101, E0611, E0401

import pandas as pd
import tensorflow as tf
import libnrl.lle as lle
import libnrl.hope as hope
import libnrl.graph as gh
from tensorflow.contrib.tensorboard.plugins import projector
import os


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
    mat = kneighbors_graph(vectors, N, metric='cosine',
                           mode='distance', include_self=True)
    mat.data = 1 - mat.data  # to similarity

    g = nx.from_scipy_sparse_matrix(mat, create_using=nx.Graph())

    return g, mat.toarray()


dataset = fetch_data('data')
graph, adj_mat = text_to_graph(dataset.data)

labels = dataset.target


############# INSERT YOUR CODE HERE (≈1 line) #############
g = gh.Graph()
g.G = graph
g.node_size = graph.number_of_nodes
# model = lle.LLE(graph=g, d=256)
model = hope.HOPE(graph=g, d=128)
import numpy as np
# embeddings = np.load("src/testing/lle_vec.npy")  # numpy array
embeddings = model._X  # numpy array
########################## END ############################

LOG_DIR = 'log'

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
