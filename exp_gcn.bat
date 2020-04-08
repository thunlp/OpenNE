cd src
py setup.py install
cd ..
python -m openne --method gcn --label-file data/cora/cora_labels.txt --input data/cora/cora_edgelist.txt --graph-format edgelist --weight-decay 5e-4 --feature-file data/cora/cora.features --epochs 200 --output vec_all.txt --clf-ratio 0.1