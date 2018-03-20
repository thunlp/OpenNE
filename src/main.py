import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from libnrl.graph import *
from libnrl import node2vec
from libnrl.classify import Classifier, read_node_label
from libnrl import line
from libnrl import tadw
from libnrl.gcn import gcnAPI
from libnrl.grarep import GraRep
import time

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output', required=True,
                        help='Output representation file')
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--q', default=1.0, type=float)
    parser.add_argument('--method', required=True, choices=['node2vec', 'deepWalk', 'line', 'gcn', 'grarep', 'tadw'],
                        help='The learning method')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    parser.add_argument('--feature-file', default='',
                        help='The file of node features')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='the negative ratio of LINE')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--no-auto-save', action='store_true',
                        help='no save the best embeddings when training LINE')
    parser.add_argument('--dropout', default=0.5, type=float, 
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--hidden', default=16, type=int,
                        help='Number of units in hidden layer 1')
    parser.add_argument('--kstep', default=4, type=int,
                        help='Use k-step transition probability matrix')
    parser.add_argument('--lamb', default=0.2, type=float,
                        help='lambda is a hyperparameter in TADW')
    args = parser.parse_args()
    return args


def main(args):
    t1 = time.time()
    g = Graph()
    print "Reading..."
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted, directed=args.directed)
    if args.method == 'node2vec':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                 num_paths=args.number_walks, dim=args.representation_size,
                                 workers=args.workers, p=args.p, q=args.q, window=args.window_size)
    elif args.method == 'line':
        if args.label_file and not args.no_auto_save:
            model = line.LINE(g, epoch = args.epochs, rep_size=args.representation_size, order=args.order, 
                label_file=args.label_file, clf_ratio=args.clf_ratio)
        else:
            model = line.LINE(g, epoch = args.epochs, rep_size=args.representation_size, order=args.order)
    elif args.method == 'deepWalk':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                 num_paths=args.number_walks, dim=args.representation_size,
                                 workers=args.workers, window=args.window_size, dw=True)
    elif args.method == 'tadw':
        assert args.label_file != ''
        assert args.feature_file != ''
        g.read_node_label(args.label_file)
        g.read_node_features(args.feature_file)
        model = tadw.TADW(graph=g, dim=args.representation_size, lamb=args.lamb)
    elif args.method == 'gcn':
        assert args.label_file != ''
        assert args.feature_file != ''
        g.read_node_label(args.label_file)
        g.read_node_features(args.feature_file)
        model = gcnAPI.GCN(graph=g, dropout=args.dropout,
                            weight_decay=args.weight_decay, hidden1=args.hidden,
                            epochs=args.epochs, clf_ratio=args.clf_ratio)
    elif args.method == 'grarep':
        model = GraRep(graph=g, Kstep=args.kstep, dim=args.representation_size)
    t2 = time.time()
    print t2-t1
    if args.method != 'gcn':
        print "Saving embeddings..."
        model.save_embeddings(args.output)
    if args.label_file and args.method != 'gcn':
        vectors = model.vectors
        X, Y = read_node_label(args.label_file)
        print "Training classifier using {:.2f}% nodes...".format(args.clf_ratio*100)
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, args.clf_ratio)



if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())
