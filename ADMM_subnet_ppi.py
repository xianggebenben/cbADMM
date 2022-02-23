import torch
import os
import numpy as np
import argparse
import time
try:
    import subnet.common as common
    import subnet.input_data as input_data
except:
    import common, input_data
import psutil

# =================== parameters ===================
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5,
                    help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='ppi',
                    help='Dataset: ppi')

parser.add_argument('--device', type=str, default='cuda',
                    help='cpu or cuda')
args = parser.parse_args()
device = torch.device(args.device)
# =================== fixed random seed ===================
torch.manual_seed(args.seed)
multi_label = True
# =================== import dataset ===================
if args.dataset == 'ppi':
    data = input_data.ppi()
    rho = 1e-5
    mu = 0

num_layers = [2]  # if only 1 subnet, then it should have at least 2 layers
w, z = common.gcn(data.adj_train, data.data_train.x, data.data_train.num_features, args.hidden, data.num_classes, args.seed, num_layers)

name = '_'  # checkpoint name

for i in range(len(num_layers)):
    for j in range(num_layers[i]):
        w[i][j] = w[i][j].to(device)
    if not multi_label:
        z[i][data.test_mask] = 0
    z[i] = z[i].to(device)
    name = name + str(num_layers[i]) + '_'

y = torch.zeros(z[-1].shape, device=device)


admm_train_loss = np.zeros(args.epochs)
admm_train_acc = np.zeros(args.epochs)
admm_test_loss = np.zeros(args.epochs)
admm_test_f1 = np.zeros(args.epochs)
pres = np.zeros(args.epochs)
dres = np.zeros(args.epochs)
obj = np.zeros(args.epochs)
#
time_avg_train = 0
time_avg_infer = 0
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
    admm_test_f1 = admm_per["test_acc"]
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
    w = common.update_w(data.adj_train, data.data_train.x, z, w, rho, None, mu, num_layers, y)
    print('w time: {}'.format(time.time()-pre_w))
    pre_z= time.time()
    z = common.update_z(data.adj_train, w, rho, None, mu, data.data_train.x, z, num_layers, y, data.data_train.y, multi_label=multi_label)
    print('z time: {}'.format(time.time() - pre_z))
    y, pres[epoch] = common.update_y(data.adj_train, data.data_train.x, z, w, y, rho, num_layers, None)

    pre_train = time.time()
    time_avg_train += pre_train - pre
    d = rho * (z[-1] - z_old)
    dres[epoch] = torch.norm(d ** 2)
    admm_train_acc[epoch], admm_train_loss[epoch]\
        = common.test(data.adj_train, data.data_train.x, w, num_layers,
                      data.data_train.y, multi_label=True)
    admm_test_f1[epoch], admm_test_loss[epoch]\
        = common.test(data.adj_test, data.data_test.x, w, num_layers,
                      data.data_test.y, multi_label=True)
    admm_train_loss[epoch] /= (data.data_train.y.size()[0] * data.data_train.y.size()[1])
    admm_test_loss[epoch] /= (data.data_test.y.size()[0] * data.data_test.y.size()[1])
    time_avg_infer += time.time() - pre_train


    print("rho=", rho)
    print("pres: {:.6f}. dres: {:.6f}.".format(pres[epoch], dres[epoch]))
    print("Train loss: {:.6f}. Train f1: {:.6f}."
          "Test loss: {:.6f}. Test f1: {:.6f}.".format(
        admm_train_loss[epoch], admm_train_acc[epoch], admm_test_loss[epoch], admm_test_f1[epoch]))
    torch.save({"z": z, "w": w, "y": y, "rho": rho, "epoch": epoch + 1, "num_layers": num_layers},
               'gcn_admm_' + args.dataset + '_' + repr(args.hidden) + name + '.pt')

    torch.save(
        {"train_acc": admm_train_acc, "train_cost": admm_train_loss,
         "test_acc": admm_test_f1, "test_cost": admm_test_loss,
         "pres": pres, "dres": dres, "obj": obj},
        'gcn_admm_' + args.dataset + '_performance_' + repr(args.hidden) + name + '.pt')
time_avg_train = time_avg_train / args.epochs
time_avg_infer = time_avg_infer / args.epochs
print("Average training time per iteration:{:.6f}.".format(time_avg_train))
print("Average inference time per iteration:{:.6f}.".format(time_avg_infer))


