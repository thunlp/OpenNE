cd src
python setup.py install
cd ..
python -m openne --method sdne --label-file data/cora/cora_labels.txt --input data/cora/cora_edgelist.txt --graph-format edgelist --feature-file data/cora/cora.features  --output sdne.txt --epoch 0 --lr -10.001
python -m openne --method sdne --label-file data/cora/cora_labels.txt --input data/cora/cora_edgelist.txt --graph-format edgelist --feature-file data/cora/cora.features  --output sdne.txt --epoch 25 --lr -10.001
python -m openne --method sdne --label-file data/cora/cora_labels.txt --input data/cora/cora_edgelist.txt --graph-format edgelist --feature-file data/cora/cora.features  --output sdne.txt --epoch 20 --lr -10.001
python -m openne --method sdne --label-file data/cora/cora_labels.txt --input data/cora/cora_edgelist.txt --graph-format edgelist --feature-file data/cora/cora.features  --output sdne.txt --epoch 15 --lr -10.001
python -m openne --method sdne --label-file data/cora/cora_labels.txt --input data/cora/cora_edgelist.txt --graph-format edgelist --feature-file data/cora/cora.features  --output sdne.txt --epoch 10 --lr -10.001
python -m openne --method sdne --label-file data/cora/cora_labels.txt --input data/cora/cora_edgelist.txt --graph-format edgelist --feature-file data/cora/cora.features  --output sdne.txt --epoch 5 --lr -10.001



