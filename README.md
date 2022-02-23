# Community-based Distributed Training of Graph Convolutional Networks via ADMM

## Requirements
The codebase is implemented in Python 3.8.10. package versions used for development are just below.
```
torch                1.8.1
torch-cluster        1.5.9
torch-geometric      1.7.1
torch-scatter        2.0.7
torch-sparse         0.6.10
torch-spline-conv    1.2.1
pyarrow              3.0.0
tornado              6.1
```

## How to use
###1 community
 run
```
python ADMM_subnet_ppi.py
```
on PPI dataset, or run 
```
python ADMM_subnet.py
```
on other datasets.

###N (N>1) communities
 (N+1) nodes are needed to run the N-community algorithm. Specifically, the 0 to N-1 nodes are responsible for updating node representations for different communities in parallel, and the N-th community updates weight parameters and computes accuracy.
 1.  Modify config.ini on each node. Here is the example for the node for Community 0:
 
```
[currentCommunity]
community = 0

[common]
seed =  0
epochs = 20 
hidden = 16
# Number of hidden units
dataset = ogbn_arxiv
;Dataset: cora, pubmed, citeseer, amazon_computers, amazon_photo, coauthor_cs, coauthor_physics, reddit,  ogbn_arxiv, ppi
class_num = 40
# class_num should be at least 2
num_parts = 2
# number of partitions of a graph

rho = 1
mu = 0
plasma_path = /tmp/plasma
#modify ‘/tmp/plasma’ to an existing path
device = cuda
# cpu or cuda
chunks = 1
#how many chunks do you want to split the weights
epoch_for_comm = 1
# communities send 1st and 2nd info to its neighbor communities every "epoch_for_comm" epochs.
[community0]
server = 10.65.187.246

[community1]
server = 10.65.187.225

[community2]
server = 10.65.187.236

[community3]
server = 10.65.187.249
```
2. On each node, run 
```
plasma_store -s /tmp/plasma -m 300000000
```
3. On each node, run 
```
python3 ADMM_subnet_server.py
```
4. On each node, run the following command
```
python3 ADMM_subnet_parallel_ppi.py
```
if using PPI dataset, or run
```
python3 ADMM_subnet_parallel.py
```
on other datasets.

