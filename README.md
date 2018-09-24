# OpenNE: An open source toolkit for Network Embedding

This repository provides a standard NE/NRL(Network Representation Learning）training and testing framework. In this framework, we unify the input and output interfaces of different NE models and provide scalable options for each model. Moreover, we implement typical NE models under this framework based on tensorflow, which enables these models to be trained with GPUs.

We develop this toolkit according to the settings of DeepWalk. The implemented or modified models include [DeepWalk](https://github.com/phanein/deepwalk), [LINE](https://github.com/tangjianpku/LINE), [node2vec](https://github.com/aditya-grover/node2vec), [GraRep](https://github.com/ShelsonCao/GraRep), [TADW](https://github.com/thunlp/TADW), [GCN](https://github.com/tkipf/gcn), HOPE, GF, SDNE and LE. We will implement more representative NE models continuously according to our released [NRL paper list](https://github.com/thunlp/nrlpapers). Specifically, we welcome other researchers to contribute NE models into this toolkit based on our framework. We will announce the contribution in this project.

## Usage

#### Installation

- Clone this repo.
- enter the directory where you clone it, and run the following code
    ```bash
    pip install -r requirements.txt
    cd src
    python setup.py install
    ```

#### General Options

You can check out the other options available to use with *OpenNE* using:

    python -m openne --help


- --input, the input file of a network;
- --graph-format, the format of input graph, adjlist or edgelist;
- --output, the output file of representation (GCN doesn't need it);
- --representation-size, the number of latent dimensions to learn for each node; the default is 128
- --method, the NE model to learn, including deepwalk, line, node2vec, grarep, tadw, gcn, lap, gf, hope and sdne;
- --directed, treat the graph as directed; this is an action;
- --weighted, treat the graph as weighted; this is an action;
- --label-file, the file of node label; ignore this option if not testing;
- --clf-ratio, the ratio of training data for node classification; the default is 0.5;
- --epochs, the training epochs of LINE and GCN; the default is 5;

#### Example

To run "node2vec" on BlogCatalog network and evaluate the learned representations on multi-label node classification task, run the following command in the home directory of this project:

    python -m openne --method node2vec --label-file data/blogCatalog/bc_labels.txt --input data/blogCatalog/bc_adjlist.txt --graph-format adjlist --output vec_all.txt --q 0.25 --p 0.25

To run "gcn" on Cora network and evaluate the learned representations on multi-label node classification task, run the following command in the home directory of this project:

    python -m openne --method gcn --label-file data/cora/cora_labels.txt --input data/cora/cora_edgelist.txt --graph-format edgelist --feature-file data/cora/cora.features  --epochs 200 --output vec_all.txt --clf-ratio 0.1

#### Specific Options

DeepWalk and node2vec:

- --number-walks, the number of random walks to start at each node; the default is 10;
- --walk-length, the length of random walk started at each node; the default is 80;
- --workers, the number of parallel processes; the default is 8;
- --window-size, the window size of skip-gram model; the default is 10;
- --q, only for node2vec; the default is 1.0;
- --p, only for node2vec; the default is 1.0;

LINE:

- --negative-ratio, the default is 5;
- --order, 1 for the 1st-order, 2 for the 2nd-order and 3 for 1st + 2nd; the default is 3;
- --no-auto-save, no early save when training LINE; this is an action; when training LINE, we will calculate F1 scores every epoch. If current F1 is the best F1, the embeddings will be saved.

GraRep:

- --kstep, use k-step transition probability matrix（make sure representation-size%k-step == 0).

TADW:

- --lamb, lamb is a hyperparameter in TADW that controls the weight of regularization terms.

GCN:

- --feature-file, The file of node features;
- --epochs, the training epochs of GCN; the default is 5;
- --dropout, dropout rate;
- --weight-decay, weight for l2-loss of embedding matrix;
- --hidden, number of units in the first hidden layer.

GraphFactorization:

- --epochs, the training epochs of GraphFactorization; the default is 5;
- --weight-decay, weight for l2-loss of embedding matrix;
- --lr, learning rate, the default is 0.01

SDNE:

- --encoder-list, a list of numbers of the neuron at each encoder layer, the last number is the dimension of the output node representation, the default is [1000, 128]
- --alpha, alpha is a hyperparameter in SDNE that controls the first order proximity loss, the default is 1e-6
- --beta, beta is used for construct matrix B, the default is 5
- --nu1, parameter controls l1-loss of weights in autoencoder, the default is 1e-5
- --nu2, parameter controls l2-loss of weights in autoencoder, the default is 1e-4
- --bs, batch size, the default is 200
- --lr, learning rate, the default is 0.01

#### Input
The supported input format is an edgelist or an adjlist:

    edgelist: node1 node2 <weight_float, optional>
    adjlist: node n1 n2 n3 ... nk
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

If the model needs additional features, the supported feature input format is as follow (**feature_i** should be a float number):

    node feature_1 feature_2 ... feature_n


#### Output
The output file has *n+1* lines for a graph with *n* nodes. 
The first line has the following format:

    num_of_nodes dim_of_representation

The next *n* lines are as follows:
    
    node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *OpenNE*.

#### Evaluation

If you want to evaluate the learned node representations, you can input the node labels. It will use a portion (default: 50%) of nodes to train a classifier and calculate F1-score on the rest dataset.

The supported input label format is

    node label1 label2 label3...

#### Embedding visualization

To show how to apply dimension reduction methods like t-SNE and PCA to embedding visualization, we choose the 20 newsgroups dataset. Using the text feature, we built the news network by `kneighbors_graph` in scikit-learn. We uploaded the results of different methods in **t-SNE-PCA.pptx** where the colors of nodes represent the labels of nodes. A simple script is shown as follows:

    cd visualization_example
    python 20newsgroup.py
    tensorboard --logdir=log/

After running the tensorboard, visit `localhost:6006` to view the result.

## Comparisons with other implementations

Running environment:  <br />
BlogCatalog: CPU: Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz. <br />
Wiki, Cora: CPU: Intel(R) Core(TM) i5-7267U CPU @ 3.10GHz. <br />

We show the node classification results of various methods in different datasets. We set representation dimension to 128, **kstep=4** in GraRep. 

Note that, both GCN(a semi-supervised NE model) and TADW need additional text features as inputs. Thus, we evaluate these two models on Cora in which each node has text information. We use 10% labeled data to train GCN.

[BlogCatalog](http://leitang.net/social_dimension.html): 10312 nodes, 333983 edges, 39 labels,  undirected:

- data/blogCatalog/bc_adjlist.txt
- data/blogCatalog/bc_edgelist.txt
- data/blogCatalog/bc_labels.txt

|Algorithm | Time| Micro-F1 | Macro-F1|
|:------------|-------------:|------------:|-------:|
|[DeepWalk](https://github.com/phanein/deepwalk) | 271s | 0.385 | 0.238|
|[LINE 1st+2nd](https://github.com/tangjianpku/LINE) | 2008s | 0.398 | 0.235|
|[Node2vec](https://github.com/aditya-grover/node2vec) | 2623s  | 0.404| 0.264|
|[GraRep](https://github.com/ShelsonCao/GraRep) | - | - | - |
|OpenNE(DeepWalk) | 986s  | 0.394 | 0.249|
|OpenNE(LINE 1st+2nd) | 1555s | 0.390 | 0.253|
|OpenNE(node2vec) | 3501s  | 0.405 | 0.275|
|OpenNE(GraRep) | 4178s | 0.393 | 0.230 |

[Wiki](https://github.com/thunlp/MMDW/tree/master/data) (Wiki dataset is provided by [LBC project](http://www.cs.umd.edu/~sen/lbc-proj/LBC.html). But the original link failed.): 2405 nodes, 17981 edges, 19 labels, directed:

- data/wiki/Wiki_edgelist.txt
- data/wiki/Wiki_category.txt

|Algorithm | Time| Micro-F1 | Macro-F1|
|:------------|-------------:|------------:|-------:|
|[DeepWalk](https://github.com/phanein/deepwalk) | 52s | 0.669 | 0.560|
|[LINE 2nd](https://github.com/tangjianpku/LINE) | 70s | 0.576 | 0.387|
|[node2vec](https://github.com/aditya-grover/node2vec) | 32s  | 0.651 | 0.541|
|[GraRep](https://github.com/ShelsonCao/GraRep) | 19.6s | 0.633 | 0.476|
|OpenNE(DeepWalk) | 42s  | 0.658 | 0.570|
|OpenNE(LINE 2nd) | 90s | 0.661 | 0.521|
|OpenNE(Node2vec) | 33s  | 0.655 | 0.538|
|OpenNE(GraRep) | 23.7s | 0.649 | 0.507 |
|OpenNE(GraphFactorization) | 12.5s | 0.637 | 0.450 |
|OpenNE(HOPE) | 3.2s | 0.601 | 0.438 |
|OpenNE(LaplacianEigenmaps) | 4.9s | 0.277 | 0.073 |
|OpenNE(SDNE) | 39.6s | 0.643 | 0.498 |


[Cora](https://linqs.soe.ucsc.edu/data): 2708 nodes, 5429 edges, 7 labels, directed:

- data/cora/cora_edgelist.txt
- data/cora/cora.features
- data/cora/cora_labels.txt

|Algorithm | Dropout | Weight_decay | Hidden | Dimension | Time| Accuracy |
|:------------|-------------:|-------:|-------:|-------:|-------:|-------:|
| [TADW](https://github.com/thunlp/TADW) | - | - | - | 80*2 | 13.9s | 0.780 |
| [GCN](https://github.com/tkipf/gcn) | 0.5 | 5e-4 | 16 | - | 4.0s | 0.790 |
| OpenNE(TADW) | - | - | - | 80*2 | 20.8s | 0.791 |
| OpenNE(GCN) | 0.5 | 5e-4 | 16 | - | 5.5s | 0.789 |
| OpenNE(GCN) | 0 | 5e-4 | 16 | - | 6.1s | 0.779 |
| OpenNE(GCN) | 0.5 | 1e-4 | 16 | - | 5.4s | 0.783 |
| OpenNE(GCN) | 0.5 | 5e-4 | 64 | - | 6.5s | 0.779 |


## Citing

If you find *OpenNE* is useful for your research, please consider citing the following papers:

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
    
    @Article{tu2017network,
      Title                    = {Network representation learning: an overview},
      Author                   = {TU, Cunchao and YANG, Cheng and LIU, Zhiyuan and SUN, Maosong},
      Journal                  = {SCIENTIA SINICA Informationis},
      Volume                   = {47},
      Number                   = {8},
      Pages                    = {980--996},
      Year                     = {2017}
    }
    
    @inproceedings{ou2016asymmetric,
      title                    = {Asymmetric transitivity preserving graph embedding},
      author                   = {Ou, Mingdong and Cui, Peng and Pei, Jian and Zhang, Ziwei and Zhu, Wenwu},
      booktitle                = {Proceedings of the 22nd ACM SIGKDD},
      pages                    = {1105--1114},
      year                     = {2016},
      organization             = {ACM}
    }

    @inproceedings{belkin2002laplacian,
      title                    = {Laplacian eigenmaps and spectral techniques for embedding and clustering},
      author                   = {Belkin, Mikhail and Niyogi, Partha},
      booktitle                = {Advances in neural information processing systems},
      pages                    = {585--591},
      year                     = {2002}
    }

    @inproceedings{ahmed2013distributed,
      title                    = {Distributed large-scale natural graph factorization},
      author                   = {Ahmed, Amr and Shervashidze, Nino and Narayanamurthy, Shravan and Josifovski, Vanja and Smola, Alexander J},
      booktitle                = {Proceedings of the 22nd international conference on World Wide Web},
      pages                    = {37--48},
      year                     = {2013},
      organization             = {ACM}
    }

    @inproceedings{wang2016structural,
      title                    = {Structural deep network embedding},
      author                   = {Wang, Daixin and Cui, Peng and Zhu, Wenwu},
      booktitle                = {Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining},
      pages                    = {1225--1234},
      year                     = {2016},
      organization             = {ACM}
    }

## Sponsor

This research is supported by Tencent, MSRA, NSFC and [BBDM-Lab](http://www.bioinfotech.cn).

<img src="http://logonoid.com/images/tencent-logo.png" width = "300" height = "30" alt="tencent" align=center />

<img src="http://net.pku.edu.cn/~xjl/images/msra.png" width = "200" height = "100" alt="MSRA" align=center />

<img src="http://www.dragon-star.eu/wp-content/uploads/2014/04/NSFC_logo.jpg" width = "100" height = "80" alt="NSFC" align=center />
