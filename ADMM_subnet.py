import torch
import os
import numpy as np
import argparse
import time
import common, input_data
import psutil

# =================== parameters ===================
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset: cora, pubmed, citeseer, amazon_computers, amazon_photo, coauthor_cs,coauthor_physics,'
                         'flickr, reddit2, ogbn_arxiv.')
parser.add_argument('--device', type=str, default='cpu',
                    help='cpu or cuda')
args = parser.parse_args()
args.device = 'cuda'
device = torch.device(args.device)
# =================== fixed random seed ===================
torch.manual_seed(args.seed)
multi_label = None
# =================== import dataset ===================
if args.dataset == 'cora':
    data = input_data.cora()
    rho = 0.1
    mu = 100
elif args.dataset == 'pubmed':
    data = input_data.pubmed()
    rho = 0.1
    mu = 1
elif args.dataset == 'citeseer':
    data = input_data.citeseer()
    rho = 0.1
    mu = 100
elif args.dataset == 'amazon_computers':
    data = input_data.amazon_computers()
    rho = 0.1
    mu = 0
elif args.dataset == 'amazon_photo':
    data = input_data.amazon_photo()
    rho = 0.1
    mu = 0
elif args.dataset == 'coauthor_cs':
    data = input_data.coauthor_cs()
    rho = 0.1
    mu = 100
elif args.dataset == 'coauthor_physics':
    data = input_data.coauthor_physics()
    rho = 0.1
    mu = 100
elif args.dataset == 'flickr':
    data = input_data.flickr()
    rho = 0.1
    mu = 0
elif args.dataset == 'reddit2':
    data = input_data.reddit2()
    rho = 0.1
    mu = 0
elif args.dataset == 'ogbn_arxiv':
    data = input_data.ogbn_arxiv(class_list=[i for i in range(40)])
    rho = 0.1
    mu = 0

print("Dataset: {}".format(args.dataset))


if not multi_label:
    data.x = data.x.to(device)
    data.adj = data.adj.to(device)
    data.label_train_onehot = data.label_train_onehot.to(device)
    data.label_train = data.label_train.to(device)
    data.label_test = data.label_test.to(device)
# num_layers = [1, 2, 3]  # if only 1 subnet, then it should have at least 2 layers
num_layers = [2]  # if only 1 subnet, then it should have at least 2 layers

w, z = common.gcn(data.adj, data.x, data.num_features, args.hidden, data.num_classes, args.seed, num_layers)

name = '_'  # checkpoint name

for i in range(len(num_layers)):
    for j in range(num_layers[i]):
        w[i][j] = w[i][j].to(device)
    z[i][data.test_mask] = 0
    z[i] = z[i].to(device)
    name = name + str(num_layers[i]) + '_'

y = torch.zeros(z[-1].shape, device=device)

admm_train_loss = np.zeros(args.epochs)
admm_train_acc = np.zeros(args.epochs)
admm_test_loss = np.zeros(args.epochs)
admm_test_acc = np.zeros(args.epochs)
pres = np.zeros(args.epochs)
dres = np.zeros(args.epochs)
obj = np.zeros(args.epochs)
#
time_avg = 0
min_epoch = 0

if os.path.exists('gcn_admm_' + args.dataset + '_' + repr(args.hidden) + name + '.pt') \
        and os.path.exists('gcn_admm_' + args.dataset + '_performance_' + repr(args.hidden) + name + '.pt'):
    admm_var = torch.load('gcn_admm_' + args.dataset + '_' + repr(args.hidden) + name + '.pt', map_location=device)
    rho = admm_var['rho']
    num_layers = admm_var['num_layers']
    w=admm_var['w']
    z=admm_var['z']
    for i in range(len(num_layers)):
        for j in range(num_layers[i]):
            w[i][j] = admm_var['w'][i][j].to(device)
        z[i] = admm_var['z'][i].to(device)
    y = admm_var['y'].to(device)
    min_epoch = admm_var['epoch']
    admm_per = torch.load('gcn_admm_' + args.dataset + '_performance_' + repr(args.hidden) + name + '.pt',
                          map_location=device)
    admm_train_acc = admm_per["train_acc"]
    admm_train_loss = admm_per["train_cost"]
    admm_test_acc = admm_per["test_acc"]
    admm_test_loss = admm_per["test_cost"]
    pres = admm_per["pres"]
    dres = admm_per["dres"]
    obj = admm_per["obj"]

    del admm_var
    del admm_per
    print("Loaded num_layers: {}".format(num_layers))

for epoch in range(min_epoch, args.epochs):
    pre = time.time()
    z_old = z[-1].clone()
    print("----------------------------------------------")
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    print("iter={:3d}".format(epoch))
    pre_w = time.time()
    w = common.update_w(data.adj, data.x, z, w, rho, data.train_mask, mu, num_layers, y)
    print('w time: {}'.format(time.time()-pre_w))
    pre_z= time.time()
    z = common.update_z(data.adj, w, rho, data.train_mask, mu, data.x, z, num_layers, y, data.label_train_onehot, multi_label=multi_label)
    print('z time: {}'.format(time.time() - pre_z))
    y, pres[epoch] = common.update_y(data.adj, data.x, z, w, y, rho, num_layers, data.train_mask)

    pre_res = time.time()
    time_avg+=pre_res-pre
    d = rho * (z[-1] - z_old)[data.train_mask]
    dres[epoch] = torch.norm(d ** 2)
    admm_train_acc[epoch], admm_test_acc[epoch], admm_train_loss[epoch], admm_test_loss[epoch] \
        = common.test(data.adj, data.x, w, num_layers, data.label_train, data.label_test, data.train_mask,
                      data.test_mask)
    obj[epoch], merge_index = common.objective(data.adj, data.x, z, w, num_layers, y, mu, rho, data.label_train_onehot,
                                               data.train_mask)
    print("obj={}".format(obj[epoch]))

    print("rho=", rho)
    print("pres: {:.6f}. dres: {:.6f}.".format(pres[epoch], dres[epoch]))
    print("Train loss: {:.6f}. Train acc: {:.6f}."
          "Test loss: {:.6f}. Test acc: {:.6f}.".format(
        admm_train_loss[epoch], admm_train_acc[epoch], admm_test_loss[epoch], admm_test_acc[epoch]))
    torch.save({"z": z, "w": w, "y": y, "rho": rho, "epoch": epoch + 1,"num_layers":num_layers},
               'gcn_admm_' + args.dataset + '_' + repr(args.hidden) + name + '.pt')

    torch.save(
        {"train_acc": admm_train_acc, "train_cost": admm_train_loss,
         "test_acc": admm_test_acc, "test_cost": admm_test_loss,
         "pres": pres, "dres": dres, "obj": obj},
        'gcn_admm_' + args.dataset + '_performance_' + repr(args.hidden) + name + '.pt')
time_avg=time_avg/args.epochs
print("Average time per iteration:{:.6f}.".format(time_avg))


