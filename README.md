# OpenNE (sub-project of OpenSKL)

OpenNE is a sub-project of OpenSKL, providing an **Open**-source **N**etwork **E**mbedding toolkit for network representation learning (NRL), with [TADW](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) as key features to incorporate text attributes of nodes.

## Overview

OpenNE provides a standard training and testing toolkit for network embedding. We unify the input and output interfaces of different NE models and provide scalable options for each model. Moreover, we implement typical NE models based on tensorflow, which enables these models to be trained with GPUs.

## Models
Besides [TADW](https://github.com/thunlp/TADW) for learning network embeddings with text attributes, we also implement typical models including [DeepWalk](https://github.com/phanein/deepwalk)  [LINE](https://github.com/tangjianpku/LINE), [node2vec](https://github.com/aditya-grover/node2vec), [GraRep](https://github.com/ShelsonCao/GraRep), , [GCN](https://github.com/tkipf/gcn), HOPE, GF, SDNE and LE. 

If you want to learn more about network embedding, visit another project of ours [NRL paper list](https://github.com/thunlp/nrlpapers).

## Evaluation

To validate the effectiveness of this toolkit, we employ the node classification task for evaluation.

### Settings

We show the node classification results of various methods in different datasets. We set representation dimension to 128, **kstep=4** in GraRep. Note that, both GCN(a semi-supervised NE model) and TADW need additional text features as inputs. Thus, we evaluate these two models on Cora in which each node has text information. We use 10% labeled data to train GCN.

[Wiki](https://github.com/thunlp/MMDW/tree/master/data) (Wiki dataset is provided by [LBC project](http://www.cs.umd.edu/~sen/lbc-proj/LBC.html). But the original link failed.): 2405 nodes, 17981 edges, 19 labels, directed:

- data/wiki/Wiki_edgelist.txt
- data/wiki/Wiki_category.txt

[Cora](https://linqs.soe.ucsc.edu/data): 2708 nodes, 5429 edges, 7 labels, directed:

- data/cora/cora_edgelist.txt
- data/cora/cora.features
- data/cora/cora_labels.txt

Running environment:  <br />
BlogCatalog: CPU: Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz. <br />
Wiki, Cora: CPU: Intel(R) Core(TM) i5-7267U CPU @ 3.10GHz. <br />

### Results

We report the Micro-F1 and Macro-F1 performance to quantify the effectiveness, and the running time for efficiency evaluation. Overall, OpenNE can provide comparable effectiveness and efficiency as the original papers.

Wiki:

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

Cora:

|Algorithm | Dropout | Weight_decay | Hidden | Dimension | Time| Accuracy |
|:------------|-------------:|-------:|-------:|-------:|-------:|-------:|
| [TADW](https://github.com/thunlp/TADW) | - | - | - | 80*2 | 13.9s | 0.780 |
| [GCN](https://github.com/tkipf/gcn) | 0.5 | 5e-4 | 16 | - | 4.0s | 0.790 |
| OpenNE(TADW) | - | - | - | 80*2 | 20.8s | 0.791 |
| OpenNE(GCN) | 0.5 | 5e-4 | 16 | - | 5.5s | 0.789 |
| OpenNE(GCN) | 0 | 5e-4 | 16 | - | 6.1s | 0.779 |
| OpenNE(GCN) | 0.5 | 1e-4 | 16 | - | 5.4s | 0.783 |
| OpenNE(GCN) | 0.5 | 5e-4 | 64 | - | 6.5s | 0.779 |

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

## Citation

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


******************
## About OpenSKL
OpenSKL project aims to harness the power of both structured knowledge and natural languages via representation learning. All sub-projects of OpenSKL, under the categories of **Algorithm**, **Resource** and **Application**, are as follows.

- **Algorithm**: 
  - [OpenKE](https://www.github.com/thunlp/OpenKE)
    - An effective and efficient toolkit for representing structured knowledge in large-scale knowledge graphs as embeddings, with <a href="https://ojs.aaai.org/index.php/AAAI/article/view/9491/9350"> TransR</a> and  <a href="https://aclanthology.org/D15-1082.pdf">PTransE</a> as key features to handle complex relations and relational paths.
    - This toolkit also includes three repositories:
       - [KB2E](https://www.github.com/thunlp/KB2E)
       - [TensorFlow-Transx](https://www.github.com/thunlp/TensorFlow-Transx)
       - [Fast-TransX](https://www.github.com/thunlp/Fast-TransX)
  - [ERNIE](https://github.com/thunlp/ERNIE)
    - An effective and efficient toolkit for augmenting pre-trained language models with knowledge graph representations.
  - [OpenNE](https://www.github.com/thunlp/OpenNE)
    - An effective and efficient toolkit for representing nodes in large-scale graphs as embeddings, with [TADW](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) as key features to incorporate text attributes of nodes.
  - [OpenNRE](https://www.github.com/thunlp/OpenNRE)
    - An effective and efficient toolkit for implementing neural networks for extracting structured knowledge from text, with [ATT](https://aclanthology.org/P16-1200.pdf) as key features to consider relation-associated text information.
    - This toolkit also includes two repositories:
      - [JointNRE](https://www.github.com/thunlp/JointNRE)
      - [NRE](https://github.com/thunlp/NRE)
- **Resource**:
  - The embeddings of large-scale knowledge graphs pre-trained by OpenKE, covering three typical large-scale knowledge graphs: Wikidata, Freebase, and XLORE. The embeddings are free to use under the [MIT license](https://opensource.org/license/mit/), and please click the following link to submit [download requests](http://139.129.163.161/download/wikidata).
  - OpenKE-Wikidata
    - Wikidata is a free and collaborative database, collecting structured data to provide support for Wikipedia. The original Wikidata contains 20,982,733 entities, 594 relations and 68,904,773 triplets. In particular, Wikidata-5M is the core subgraph of Wikidata, containing  5,040,986 high-frequency entities from Wikidata with their corresponding 927 relations and 24,267,796 triplets.
    - TransE version: Knowledge embeddings of Wikidata pre-trained by OpenKE. 
    - [TransR version](https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/transr.npy) of Wikidata-5M: Knowledge embeddings of Wikidata-5M pre-trained by OpenKE for the project [Knowledge-Plugin](https://github.com/THUNLP/Knowledge-Plugin).
  - OpenKE-Freebase
    - Freebase was a large collaborative knowledge base consisting of data composed mainly by its community members. It was an online collection of structured data harvested from many sources. Freebase contains 86,054,151 entities, 14,824 relations and 338,586,276 triplets.
    - TransE version: Knowledge embeddings of Freebase pre-trained by OpenKE. 
  - OpenKE-XLORE
    - XLORE is one of the most popular Chinese knowledge graphs developed by THUKEG. XLORE contains 10,572,209 entities, 138,581 relations and 35,954,249 triplets.
    - TransE version: Knowledge embeddings of XLORE pre-trained by OpenKE.
- **Application**:   
    - [Knowledge-Plugin](https://github.com/THUNLP/Knowledge-Plugin)
      - An effective and efficient toolkit of plug-and-play knowledge injection for pre-trained language models. Knowledge-Plugin is general for all kinds of knowledge graph embeddings mentioned above. In the toolkit, we provide the example of plugging OpenKE-Wikidata embeddings into BERT.
