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

If instead you want to create a dataset from file, you can provide your own graph by using switch
- --local-dataset (action store_true; mutually exclusive with --dataset)

and the following arguments:
- --root-dir, root directory of input files. If empty, you should provide absolute paths for graph files;
- --edgefile, description of input graph in edgelist format;
- --adjfile, description of input graph in adjlist format (mutually exclusive with --edgefile);
- --label-file, node label file; 
- --features, node feature file for certain models (optional);
- --name, dataset name, "SelfDefined" by default;
- --weighted, view graph as weighted (action store_true);
- --directed, view graph as directed (action store_true);

For general training options:
- --dim, dimension of node representation, 128 by default;
- --clf-ratio, the ratio of training data for node classification, 0.5 by default;
- --no-save, choose not to save the result (action store_false, dest=save);
- --output, output file for vectors, which will be saved to "results" by default;
- --sparse, calculate by sparse matrices (action store_true) (only supports lle & gcn);

For models with multiple epochs:
- --epochs, number of epochs;
- --validate, True if validation is needed; by default it is False except with GCN;
- --validation-interval, number of epochs between two validations, 5 by default;
- --debug-output-interval, number of epochs between two debug outputs, 5 by default;


#### Specific Options

GraphFactorization:
- --weight-decay, weight for l2-loss of embedding matrix (1.0 by default);
- --lr, learning rate (0.003 by default)

GraRep:

- --kstep, use k-step transition probability matrix（requires dim % kstep == 0).


HOPE:
- --measurement {katz, cn, rpr, aa}  mesurement matrix, katz by default;
- --beta, parameter with katz measurement, 0.02 by default;
- --alpha, parameter with rpr measurement, 0.5 by default;

LINE:
- --lr, learning rate, 0.001 by default;
- --batch-size, 1000 by default;
- --negative-ratio, 5 by default;
- --order, 1 for the 1st-order, 2 for the 2nd-order and 3 for 1st + 2nd, 3 by default;

SDNE:

- --encoder-list, list of neuron numbers at each encoder layer. The last number is the dimension of the output node representation. [1000, 128] by default. See "Input Instructions";
- --alpha, parameter that controls the first-order proximity loss, 1e-6 by default;
- --beta, parameter used for construct matrix B, 5 by default;
- --nu1, parameter that controls l1-loss of weights in autoencoder, 1e-8 by default;
- --nu2, parameter that controls l2-loss of weights in autoencoder, 1e-5 by default;
- --bs, batch size, 200 by default;
- --lr, learning rate, 0.01 by default;
- --decay, allow decay in learning rate (action store_true);

TADW: (requires attributed graph, eg. cora, pubmed, citeseer)
- --lamb, parameter that controls the weight of regularization terms, 0.2 by default;

GCN: (requires attributed graph)
- --lr, learning rate, 0.01 by default;
- --dropout, dropout rate, 0.5 by default;
- --weight-decay, weight for l2-loss of embedding matrix, 0.0001 by default;
- --hiddens, list of neuron numbers in each hidden layer, [16] by default;
- --max-degree, maximum Chebyshev polynomial degree. 0 (disable Chebyshev polynomial) by default;


DeepWalk and node2vec:
- --num-paths, number of random walks that starts at each node, 10 by default;
- --path-length, length of random walk started at each node, 80 by default;
- --window, window size of skip-gram model; 10 by default;
- --q (only node2vec), 1.0 by default;
- --p (only node2vec), 1.0 by default.

#### Input Instructions

##### Use default values

For the simplest use, if you want to run GraphFactorization on BlogCatalog, input the following command:

    python -m openne --model gf --dataset blogcatalog

##### store_true and store_false parameters

Parameters like --sparse have action "store_true", which means they are False by default, and should be specified if you want to assign True. Run GCN with sparsed matrices by the following command:
    
    python -m openne --model gcn --dataset cora --sparse

You can use store_false parameters, eg. --no-save, in a similar way:

    python -m openne --model gcn --dataset cora --sparse --no-save
    
The above command asks for not saving the trained results (while it is saved by default).
    
##### Use your own datasets

Use --local-dataset (which is also a store_true parameter!) and specify --root-dir, --edgefile or --adjfile, --labelfile, --features and --status to import dataset from file. 

Optionally, specify store_true parameters --weighted and --directed to view the graph as weighted and/or directed.

If you wish to use your dataset in "~/mydataset", which includes edges.txt, an edgelist file, and labels.txt, a label file, input the following:
    
    python -m openne --model gf --local-dataset --root-dir ~/mydataset --edgefile edges.txt --labelfile labels.txt


##### Input values

While all parameter names must be provided in lower case, string input values are case insensitive:

    python -m openne --model SDnE --dataset coRA

The simplest way to provide a Python list (as of --encoder-layer-list in SDNE and --hiddens in GCN) is to directly input it without space. You can also wrap the list in double quotes (") to input spaces. The following commands are the same:

    python -m openne --model sdne --dataset cora --encoder-layer-list [1000,128]
    python -m openne --model sdne --dataset cora --encoder-layer-list "[1000,128]"
    python -m openne --model sdne --dataset cora --encoder-layer-list "[1000, 128]"
    


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

