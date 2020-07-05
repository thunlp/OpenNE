# OpenNE-PyTorch

This is an open-source framework for self-supervised/unsupervised graph embedding implemented by PyTorch, migrated from the earlier version implemented by Tensorflow. 


## Overview
#### New Features

- **A unified framework**: We provide a unified framework for self-supervised/unsupervised node representation learning. Our models include unsupervised network embedding (NE) methods (DeepWalk, Node2vec, HOPE, GraRep, LLE, Lap, TADW, GF, LINE, SDNE) and recent self-supervised graph embedding methods (GAE, VGAE).

- **Efficiency**: We provide faster and more efficient models than those in the previous version.

| Method | Time | | Accuracy | | 
| | OpenNE | OpenNE-PyTorch | OpenNE | OpenNE-PyTorch |
| ---- | ---- | ---- | ---- | ---- |
DeepWalk | 77.26 | **73.87** | **.828** | .827 
Node2vec | 42.25 | **33.86** | **.820** | .812
HOPE | 132.52 | **3.50** | .308 | **.751**
GraRep | 60.64 | **4.90** | .770 | **.785**
LLE | 792.28 | **17.67** | .301 | **.306**
Lap | **6.13** | 6.47 | .305 | **.315**
TADW | 35.71 | **21.28** | **.853** | .850
GF | **15.64** | 22.24 | .521 | **.577**
LINE | 86.75 | **64.72** | **.611** | .597
SDNE | **11.07** | 16.5 | .683 | **.725**


- **Modularity**: We entangle the codes into three parts: Dataset, Model and Task. Users can easily customize the datasets, methods and tasks. It is also easy to define their specific datasets and methods.

#### Future Plan
We plan to add more models and tasks in our framework. Our future plan includes:

- More self-supervised models such as ARGA/ARVGA, GALA and AGE.

- New tasks for link prediction, graph clustering and graph classification.

You are welcomed to add your own datasets and methods by proposing new pull requests. 

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

- --model {deepwalk, line, node2vec, grarep, tadw, gcn, lap, gf, hope and sdne} the specified NE model;
- --dataset {ppi, wikipedia, flickr, blogcatalog, wiki, pubmed, cora, citeseer} standard dataset as provided by OpenNE;

If instead you want to create a dataset from file:

- --input, the input file of a network;
- --graph-format, the format of input graph, adjlist or edgelist;
- --output, the output file of representation (GCN doesn't need it);
- --representation-size, the number of latent dimensions to learn for each node; the default is 128
- --directed, treat the graph as directed; this is an action;
- --weighted, treat the graph as weighted; this is an action;
- --label-file, the file of node label; ignore this option if not testing;
- --clf-ratio, the ratio of training data for node classification; the default is 0.5;
- --epochs, the training epochs of LINE and GCN; the default is 5;

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

- --feature-file, The file of node features;
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

