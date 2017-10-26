# LibNRL

This  repository provide a NRL(network representation learning) framework. It contains implementation of "node2vec", "LINE", "GraRep", "TADW" and "GCN".

The NRL algorithm learns continuous representations for nodes in any (un)directed, (un)weighted graph.

## Requirements

-  numpy
-  networkx==2.0
-  scipy (if you want to use cython to spead up node2vec, please set scipy==0.15.1)
-  tensorflow
-  gensim
-  sklearn

## Usage

#### Example

To run "node2vec" on BlogCatalog network and evaluate the feature representations on multi-label classification for nodes, execute the following command from the project home directory:

    python src/main.py --method node2vec --label-file data/blogCatalog/bc_labels.txt --input data/blogCatalog/bc_adjlist.txt --graph-format adjlist --output vec_all.txt --q 0.25 --p 0.25

To run "gcn" on Cora network and evaluate the feature representations on classification for nodes, execute the following command from the project home directory:

    python src/main.py --method gcn --label-file data/cora/cora_labels.txt --input data/cora/cora_edgelist.txt --graph-format edgelist --feature-file data/cora/cora.features  --epochs 200 --output vec_all.txt --clf-ratio 0.1

#### Options
You can check out the other options available to use with *LibNRL* using:

    python src/main.py --help

General Options:

- --input, the input file of a network;
- --output, the output file of representation;
- --directed, treat graph as directed; this is an action;
- --representation-size, the number od latent dimensions to learn for each node; the default is 128;
- --method, the learning method including node2vec and line;
- --label-file, the file of node label;
- --graph-format, the format of input graph including adjlist and edgelist;
- --weighted, treat the graph as weighted; this is an action;
- --clf-ratio, the ratio of training data in the classification; the default is 0.5;

Node2vec Options:

- --number-walks, the number of random walks to start at each node; the default is 10;
- --walk-length, the length of random walk started at each node; the default is 80;
- --workers, the number od parallel processes to train Word2vec; the default is 8;
- --window-size, the window size of skipgram model; the default is 10;
- --q, the default is 1.0;
- --p, the default is 1.0;

LINE Options:

- --epochs, the training epochs of LINE; the default is 5;
- --negative-ratio, the default is 5;
- --order, the default is 3 which means 1st + 2nd and 1 means 1st, 2 means 2nd.
- --no-auto-stop, no early stop when training LINE; this is an action; When training LINE, we will calculate micro-F1 every epoch; If current micro-F1 is smaller than last micro-F1, the training process will stop early.

GCN Options:

- --feature-file, The file of node features;
- --epochs, the training epochs of GCN; the default is 5;
- --dropout, Dropout rate (1 - keep probability);
- --weight-decay, Weight for L2 loss on embedding matrix;
- --hidden, Number of units in hidden layer 1.

GraRep Options:

- --kstep, Use k-step transition probability matrix（make sure representation-size%k-step == 0).

#### Input
The supported input format is an edgelist or an adjlist:

    edgelist: node1 node2 <weight_float, optional>
    adjlist: node n1 n2 n3 ... nk
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

If the model need node's feature, the supported feature input format is as follow(**feature_i** should be a float number):

    node feature_1 feature_2 ... feature_n


#### Output
The output file has *n+1* lines for a graph with *n* vertices. 
The first line has the following format:

    num_of_nodes dim_of_representation

The next *n* lines are as follows:
    
    node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *libnrl*.

#### Evaluate

If you want to evaluate the feature representations, you can have the labels of node as input. It will use 50%(default) nodes to train a classifier and calculate F1-score on the rest dataset.

The supported input label format is

    node label1 label2 label3...

## Compare with other packages

We list the classification result of various methods in different datasets. We set the dimension of vectors = 128, **p=1, q=1** in node2vec and **kstep=4** in GraRep . DeepWalk can walk faster than node2vec because it only accepts binary edges. As a result, the time of DeepWalk is shorter than node2vec's.

GCN is an approach for semi-supervised learning on graph-structured data. GCN performs batch gradient descent using the full dataset for every training iteration, so it can't work with GPU on a large dataset. And GCN is suitable for the dataset whose nodes have features. In Cora, we use 10% data to train our model. For TADW, we use SVM as classifier.

CPU: Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz

[BlogCatalog](https://github.com/phanein/deepwalk/tree/master/example_graphs): 10312 nodes 333983 edges 39 labels undirected:

- data/blogCatalog/bc_adjlist.txt
- data/blogCatalog/bc_edgelist.txt
- data/blogCatalog/bc_labels.txt

|Algorithm | Time| Micro-F1 | Macro-F1|
|:------------|-------------:|------------:|-------:|
|LibNRL(node2vec) | 1522s  | 0.403 | 0.268|
|LibNRL(LINE 1st+2nd) | 943s | 0.368 | 0.192|
|LibNRL(GraRep) | 4178s | 0.393 | 0.230 |
|[Node2vec](https://github.com/aditya-grover/node2vec) | 1687s  | 0.390| 0.230|
|[LINE 1st+2nd](https://github.com/tangjianpku/LINE) | 2008s | 0.398 | 0.235|
|[DeepWalk](https://github.com/phanein/deepwalk) | 271s | 0.385 | 0.238|

[Wiki](https://github.com/thunlp/MMDW/tree/master/data): 2405 nodes 17981 edges 19 labels directed:

- data/wiki/Wiki_edgelist.txt
- data/wiki/Wiki_category.txt

|Algorithm | Time| Micro-F1 | Macro-F1|
|:------------|-------------:|------------:|-------:|
|LibNRL(Node2vec) | 49s  | 0.633 | 0.543|
|LibNRL(LINE 2nd) | 111s | 0.641 | 0.461|
|LibNRL(GraRep) | 75s | 0.654 | 0.509 |
|[node2vec](https://github.com/aditya-grover/node2vec) | 51s  | 0.623 | 0.537|
|[LINE 2nd](https://github.com/tangjianpku/LINE) | 103s | 0.584 | 0.396|
|[DeepWalk](https://github.com/phanein/deepwalk) | 50s | 0.667 | 0.566|

[cora](https://github.com/tkipf/gcn/tree/master/gcn/data): 2708 nodes 5429 edges 7 labels directed:

- data/cora/cora_edgelist.txt
- data/cora/cora.features
- data/cora/cora_labels.txt

|Algorithm | Dropout | Weight_decay | Hidden | Dimension | Time| Accuracy |
|:------------|-------------:|-------:|-------:|-------:|-------:|-------:|
| LibNRL(GCN) | 0.5 | 5e-4 | 16 | - | 5.7s | 0.780 |
| LibNRL(GCN) | 0 | 5e-4 | 16 | - | 6.5s | 0.779 |
| LibNRL(GCN) | 0.5 | 1e-4 | 16 | - | 5.5s | 0.782 |
| LibNRL(GCN) | 0.5 | 5e-4 | 64 | - | 6.0s | 0.785 |
| LibNRL(TADW) | - | - | - | 80*2 | 23.0s | 0.785 |
| [TADW](https://github.com/thunlp/TADW) | - | - | - | 80*2 | - | 0.780 |

## Citing

If you find *LibNRL* is useful for your research, please consider citing the following papers:

    @InProceedings{perozzi2014deepwalk,
      Title                    = {Deepwalk: Online learning of social representations},
      Author                   = {Perozzi, Bryan and Al-Rfou, Rami and Skiena, Steven},
      Booktitle                = {Proceedings of KDD},
      Year                     = {2014},
      Pages                    = {701--710}
    }

    @InProceedings{tang2015line,
      Title                    = {Line: Large-scale information network embedding},
      Author                   = {Tang, Jian and Qu, Meng and Wang, Mingzhe and Zhang, Ming and Yan, Jun and Mei, Qiaozhu},
      Booktitle                = {Proceedings of WWW},
      Year                     = {2015},
      Pages                    = {1067--1077}
    }

    @InProceedings{grover2016node2vec,
      Title                    = {node2vec: Scalable feature learning for networks},
      Author                   = {Grover, Aditya and Leskovec, Jure},
      Booktitle                = {Proceedings of KDD},
      Year                     = {2016},
      Pages                    = {855--864}
    }

    @article{kipf2016semi,
      Title                    = {Semi-Supervised Classification with Graph Convolutional Networks},
      Author                   = {Kipf, Thomas N and Welling, Max},
      journal                  = {arXiv preprint arXiv:1609.02907},
      Year                     = {2016}
    }

    @InProceedings{cao2015grarep,
      Title                    = {Grarep: Learning graph representations with global structural information},
      Author                   = {Cao, Shaosheng and Lu, Wei and Xu, Qiongkai},
      Booktitle                = {Proceedings of CIKM},
      Year                     = {2015},
      Pages                    = {891--900}
    }

    @InProceedings{yang2015network,
    Title                    = {Network representation learning with rich text information},
    Author                   = {Yang, Cheng and Liu, Zhiyuan and Zhao, Deli and Sun, Maosong and Chang, Edward},
    Booktitle                = {Proceedings of IJCAI},
    Year                     = {2015}
    }

## Sponsor

This research has been supported in part by NSFC and Tencent.

<img src="http://logonoid.com/images/tencent-logo.png" width = "300" height = "30" alt="tencent" align=center />

<img src="http://www.dragon-star.eu/wp-content/uploads/2014/04/NSFC_logo.jpg" width = "100" height = "80" alt="NSFC" align=center />






