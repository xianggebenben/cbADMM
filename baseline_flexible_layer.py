import torch
import numpy as np
import argparse
import time
import common
import input_data
from torch.nn import Parameter
import torch.nn.functional as F
import psutil

mem_init = psutil.virtual_memory()
print("Initial memeory used:{:.6f}GB".format(mem_init.used / 1024 / 1024 / 1024))
# =================== parameters ===================
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=4, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr_adam', type=float, default=1e-3,
                    help='Learning rate for Adam.')
parser.add_argument('--lr_adagrad', type=float, default=1e-2,
                    help='Learning rate for Adagrad.')
parser.add_argument('--lr_gd', type=float, default=1,
                    help='Learning rate for GD.')
# gd amazon photos,amazon computers 0.1
parser.add_argument('--lr_adadelta', type=float, default=0.1,
                    help='Learning rate for Adadelta.')
# adadelta cora 0.01
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='ppi',
                    help='Dataset: cora, pubmed, citeseer, amazon_computers, amazon_photo, '
                         'coauthor_cs, coauthor_physics, ogbn_arxiv, reddit2, flickr, ppi.')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# =================== fixed random seed ===================
# np.random.seed(args.seed)

# =================== import dataset ===================
if args.dataset == 'cora':
    data = input_data.cora()
elif args.dataset == 'pubmed':
    data = input_data.pubmed()
elif args.dataset == 'citeseer':
    data = input_data.citeseer()
elif args.dataset == 'amazon_computers':
    data = input_data.amazon_computers()
elif args.dataset == 'amazon_photo':
    data = input_data.amazon_photo()
elif args.dataset == 'coauthor_cs':
    data = input_data.coauthor_cs()
elif args.dataset == 'coauthor_physics':
    data = input_data.coauthor_physics()
elif args.dataset == 'ogbn_arxiv':
    data = input_data.ogbn_arxiv()
elif args.dataset == 'reddit2':
    data = input_data.reddit2()
elif args.dataset == 'flickr':
    data = input_data.flickr()
elif args.dataset == 'ppi':
    data = input_data.ppi()

if args.dataset != 'ppi':
    data.x = data.x.to(device)
    data.adj = data.adj.to(device)
    data.label_train = data.label_train.to(device)
    data.label_test = data.label_test.to(device)
    data.label_train_onehot = data.label_train_onehot.to(device)
    adj_train = common.sparse_mask(data.adj, data.train_mask).to(device)



num_layers = [2]  # it should have at least 2 layers
if args.dataset != 'ppi':
    w = common.gcn(data.adj, data.x, data.num_features, args.hidden, data.num_classes, args.seed, num_layers, return_z=False)
else:
    w = common.gcn(data.adj_train, data.data_train.x, data.data_train.num_features, args.hidden, data.num_classes, args.seed, num_layers, return_z=False)



def adam_train():
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 1e-8
    adam_eta = 1 / args.lr_adam

    adam_train_loss = np.zeros(args.epochs)
    adam_train_acc = np.zeros(args.epochs)
    adam_test_loss = np.zeros(args.epochs)
    adam_test_acc = np.zeros(args.epochs)
    min_epoch = 0
    # for Adam
    adam_m_w = []
    adam_v_w = []
    adam_m_w_hat = []
    adam_v_w_hat = []
    for i in range(num_layers[0]):
        adam_m_w.append(torch.zeros_like(w[0][i]))
        adam_v_w.append(torch.zeros_like(w[0][i]))
        adam_m_w_hat.append(torch.zeros_like(w[0][i]))
        adam_v_w_hat.append(torch.zeros_like(w[0][i]))

    # if os.path.exists( 'gcn_adam_' + args.dataset + '_performance_' + repr(args.hidden) + '_' + repr(num_layers[0])+'.pt')\
    #         and os.path.exists('gcn_adam_' + args.dataset + '_variable_' + repr(args.hidden) + '_' + repr(num_layers[0]) + '.pt'):
    #     var = torch.load('gcn_adam_' + args.dataset + '_variable_' + repr(args.hidden) + '_' + repr(num_layers[0]) + '.pt',map_location=device)
    #     for i in range(num_layers[0]):
    #         min_epoch = var['epoch']
    #         w[0][i] = var['w'][0][i].to(device)
    #         adam_m_w[i] = var['adam_m_w'][i].to(device)
    #         adam_v_w[i] = var['adam_v_w'][i].to(device)
    #         adam_m_w_hat = var['adam_m_w_hat'][i].to(device)
    #         adam_v_w_hat = var['adam_m_w_hat'][i].to(device)
    #     per = torch.load('gcn_adam_' + args.dataset + '_performance_' + repr(args.hidden) + '_' + repr(num_layers[0]) + '.pt',map_location=device)
    #     adam_train_acc = per['adam_train_acc'].to(device)
    #     adam_train_loss = per['adam_train_loss'].to(device)
    #     adam_test_acc = per['adam_test_acc'].to(device)
    #     adam_test_loss = per['adam_test_loss'].to(device)

    time_sum = 0
    for epoch in range(min_epoch, args.epochs):
        print("---------------------------")
        print('iter={}'.format(epoch))
        pre = time.time()
        if args.dataset != 'ppi':
            pred = common.forwardprop(data.adj, data.x, w, num_layers)[data.train_mask]
            loss, _ = common.cross_entropy_with_softmax(pred, data.label_train_onehot)
            loss /= pred.size()[0]
        else:
            pred = common.forwardprop(data.adj_train, data.data_train.x, w, num_layers)
            loss = F.binary_cross_entropy_with_logits(pred, data.data_train.y, reduction='sum')

        gradient = torch.autograd.grad(loss, w[0])
        for i in range(num_layers[0]):
            adam_m_w[i] = adam_beta1 * adam_m_w[i] + (1 - adam_beta1) * gradient[i]
            adam_v_w[i] = adam_beta2 * adam_v_w[i] + (1 - adam_beta2) * gradient[i] ** 2
            adam_m_w_hat[i] = adam_m_w[i] / (1 - adam_beta1 ** (epoch + 1))
            adam_v_w_hat[i] = adam_v_w[i] / (1 - adam_beta2 ** (epoch + 1))
            w[0][i] = Parameter(w[0][i] - adam_m_w_hat[i] / (torch.sqrt(adam_v_w_hat[i]) + adam_eps) / adam_eta)
        time_iter = time.time() - pre
        #print("Epoch {} takes: {:.4f}".format(epoch, time_iter))
        time_sum += time_iter
        if (epoch + 1) % 5 == 0:
            if args.dataset != 'ppi':
                adam_train_acc[epoch], adam_test_acc[epoch], adam_train_loss[epoch], adam_test_loss[epoch] \
                    = common.test(data.adj, data.x, w, num_layers, data.label_train, data.label_test, data.train_mask, data.test_mask)
            else:
                adam_train_acc[epoch], adam_train_loss[epoch]\
                    = common.test(data.adj_train, data.data_train.x, w, num_layers, data.data_train.y, None, None,
                                  None, multi_label=True)
                adam_test_acc[epoch], adam_test_loss[epoch] \
                    = common.test(data.adj_test, data.data_test.x, w, num_layers, data.data_test.y, None, None,
                                  None, multi_label=True)

            print("adam_train_acc={}".format(adam_train_acc[epoch]))
            print("adam_test_acc={}".format(adam_test_acc[epoch]))

        # save performance and variables
        torch.save(
            {"adam_train_acc": adam_train_acc, "adam_train_loss": adam_train_loss, "adam_test_acc": adam_test_acc,
             "adam_test_loss": adam_test_loss},
            'gcn_adam_' + args.dataset + '_performance_' + repr(args.hidden) + '_' + repr(num_layers[0]) + '.pt')
        #
        # torch.save(
        #     {"w": w, "adam_m_w": adam_m_w, "adam_v_w": adam_v_w,
        #      "adam_m_w_hat": adam_m_w_hat, "adam_v_w_hat": adam_v_w_hat,
        #      "epoch": epoch + 1},
        #     'gcn_adam_' + args.dataset + '_variable_' + repr(args.hidden) + '_' + repr(num_layers[0]) + '.pt')
    mem = psutil.virtual_memory()
    print("adam memeory used:{:.6f}GB".format((mem.used - mem_init.used) / 1024 / 1024 / 1024))
    print("training time per epoch={}".format(time_sum / args.epochs))
    # adam_train_acc = adam_train_acc[1:]
    # adam_test_acc = adam_test_acc[1:]
    # adam_train_loss = adam_train_loss[1:]
    # adam_test_loss = adam_test_loss[1:]




def adagrad_train():
    eps = 1e-8
    lr = args.lr_adagrad

    adagrad_train_loss = np.zeros(args.epochs)
    adagrad_train_acc = np.zeros(args.epochs)
    adagrad_test_loss = np.zeros(args.epochs)
    adagrad_test_acc = np.zeros(args.epochs)

    # for Adagrad
    v_w = []


    for i in range(num_layers[0]):
        v_w.append(torch.zeros_like(w[0][i]))

    time_sum = 0
    for epoch in range(0, args.epochs):
        print("---------------------------")
        print('iter={}'.format(epoch))
        pre = time.time()
        if args.dataset != 'ppi':
            pred = common.forwardprop(data.adj, data.x, w, num_layers)[data.train_mask]
            loss, _ = common.cross_entropy_with_softmax(pred, data.label_train_onehot)
            loss /= pred.size()[0]
        else:
            pred = common.forwardprop(data.adj_train, data.data_train.x, w, num_layers)
            loss = F.binary_cross_entropy_with_logits(pred, data.data_train.y, reduction='sum')
        gradient = torch.autograd.grad(loss, w[0])
        for i in range(num_layers[0]):
            v_w[i] += torch.square(gradient[i])
            w[0][i] = Parameter(w[0][i] - lr * gradient[i] / (torch.sqrt(v_w[i]) + eps))
        time_iter = time.time() - pre
        #print("Epoch {} takes: {:.4f}".format(epoch, time_iter))
        time_sum += time_iter
        if (epoch + 1) % 5 == 0:
            if args.dataset != 'ppi':
                adagrad_train_acc[epoch], adagrad_test_acc[epoch], adagrad_train_loss[epoch], adagrad_test_loss[epoch] \
                    = common.test(data.adj, data.x, w, num_layers, data.label_train, data.label_test, data.train_mask,
                                  data.test_mask)
            else:
                adagrad_train_acc[epoch], adagrad_train_loss[epoch] \
                    = common.test(data.adj_train, data.data_train.x, w, num_layers, data.data_train.y, None, None,
                                  None, multi_label=True)
                adagrad_test_acc[epoch], adagrad_test_loss[epoch] \
                    = common.test(data.adj_test, data.data_test.x, w, num_layers, data.data_test.y, None, None,
                                  None, multi_label=True)
            print("adagrad_train_acc={}".format(adagrad_train_acc[epoch]))
            print("adagrad_test_acc={}".format(adagrad_test_acc[epoch]))
        torch.save(
            {"adagrad_train_acc": adagrad_train_acc, "adagrad_train_loss": adagrad_train_loss,
             "adagrad_test_acc": adagrad_test_acc, "adagrad_test_loss": adagrad_test_loss},
            'gcn_adagrad_' + args.dataset + '_performance_' + repr(args.hidden) + '_' + repr(num_layers[0]) + '.pt')

        # torch.save(
        #     {"w": w, "v_w": v_w, "epoch":epoch+1},
        #     'gcn_adagrad_' + args.dataset + '_variable_' + repr(args.hidden) + '_' + repr(num_layers[0]) + '.pt')

    mem = psutil.virtual_memory()
    print("adagrad memeory used:{:.6f}GB".format((mem.used - mem_init.used) / 1024 / 1024 / 1024))
    print("training time per epoch={}".format(time_sum / args.epochs))
    # adagrad_train_loss = adagrad_train_loss[1:]
    # adagrad_test_loss = adagrad_test_loss[1:]
    # adagrad_train_acc = adagrad_train_acc[1:]
    # adagrad_test_acc = adagrad_test_acc[1:]


def gd_train():
    lr = args.lr_gd

    gd_train_loss = np.zeros(args.epochs)
    gd_train_acc = np.zeros(args.epochs)
    gd_test_loss = np.zeros(args.epochs)
    gd_test_acc = np.zeros(args.epochs)

    time_sum = 0
    for epoch in range(0, args.epochs):
        print("---------------------------")
        print('iter={}'.format(epoch))
        pre = time.time()
        if args.dataset != 'ppi':
            pred = common.forwardprop(data.adj, data.x, w, num_layers)[data.train_mask]
            loss, _ = common.cross_entropy_with_softmax(pred, data.label_train_onehot)
            loss /= pred.size()[0]
        else:
            pred = common.forwardprop(data.adj_train, data.data_train.x, w, num_layers)
            loss = F.binary_cross_entropy_with_logits(pred, data.data_train.y, reduction='sum')
        gradient = torch.autograd.grad(loss, w[0])
        for i in range(num_layers[0]):
            w[0][i] = Parameter(w[0][i] - lr * gradient[i])
        time_iter = time.time() - pre
        #print("Epoch {} takes: {:.4f}".format(epoch, time_iter))
        time_sum += time_iter
        if (epoch + 1) % 5 == 0:
            if args.dataset != 'ppi':
                gd_train_acc[epoch], gd_test_acc[epoch], gd_train_loss[epoch], gd_test_loss[epoch]\
                = common.test(data.adj, data.x, w, num_layers, data.label_train, data.label_test, data.train_mask,
                              data.test_mask)
            else:
                gd_train_acc[epoch], gd_train_loss[epoch] \
                    = common.test(data.adj_train, data.data_train.x, w, num_layers, data.data_train.y, None, None,
                                  None, multi_label=True)
                gd_test_acc[epoch], gd_test_loss[epoch] \
                    = common.test(data.adj_test, data.data_test.x, w, num_layers, data.data_test.y, None, None,
                                  None, multi_label=True)
            print("gd_train_acc={}".format(gd_train_acc[epoch]))
            print("gd_test_acc={}".format(gd_test_acc[epoch]))
        torch.save(
            {"gd_train_acc": gd_train_acc, "gd_train_loss": gd_train_loss, "gd_test_acc": gd_test_acc,
             "gd_test_loss": gd_test_loss},
            'gcn_gd_' + args.dataset + '_performance_' + repr(args.hidden) + '_' + repr(num_layers[0]) + '.pt')
        # torch.save(
        #     {"w": w, "epoch": epoch+1},
        #     'gcn_gd_' + args.dataset + '_variable_' + repr(args.hidden) + '_' + repr(num_layers[0]) + '.pt')

    mem = psutil.virtual_memory()
    print("gd memeory used:{:.6f}GB".format((mem.used - mem_init.used) / 1024 / 1024 / 1024))
    print("training time per epoch={}".format(time_sum / args.epochs))
    # gd_train_loss = gd_train_loss[1:]
    # gd_test_loss = gd_test_loss[1:]
    # gd_train_acc = gd_train_acc[1:]
    # gd_test_acc = gd_test_acc[1:]



def adadelta_train():
    lr = args.lr_adadelta
    eps = 1e-8
    eta = 0.9

    adadelta_train_loss = np.zeros(args.epochs)
    adadelta_train_acc = np.zeros(args.epochs)
    adadelta_test_loss = np.zeros(args.epochs)
    adadelta_test_acc = np.zeros(args.epochs)

    # for Adadelta
    v_w = []

    for i in range(num_layers[0]):
        v_w.append(torch.zeros_like(w[0][i]))

    time_sum = 0
    for epoch in range(1, args.epochs):
        print("---------------------------")
        print('iter={}'.format(epoch))
        pre = time.time()
        if args.dataset != 'ppi':
            pred = common.forwardprop(data.adj, data.x, w, num_layers)[data.train_mask]
            loss, _ = common.cross_entropy_with_softmax(pred, data.label_train_onehot)
            loss /= pred.size()[0]
        else:
            pred = common.forwardprop(data.adj_train, data.data_train.x, w, num_layers)
            loss = F.binary_cross_entropy_with_logits(pred, data.data_train.y, reduction='sum')
        gradient = torch.autograd.grad(loss, w[0])
        for i in range(num_layers[0]):
            v_w[i] += eta * v_w[i] + (1 - eta) * torch.square(gradient[i])
            w[0][i] = Parameter(w[0][i] - lr * gradient[i] / (torch.sqrt(v_w[i]) + eps))
        time_iter = time.time() - pre
        #print("Epoch {} takes: {:.4f}".format(epoch, time_iter))
        time_sum += time_iter
        if (epoch + 1) % 5 == 0:
            if args.dataset != 'ppi':
                adadelta_train_acc[epoch], adadelta_test_acc[epoch], adadelta_train_loss[epoch], adadelta_test_loss[epoch]\
                = common.test(data.adj, data.x, w, num_layers, data.label_train, data.label_test, data.train_mask,
                              data.test_mask)
            else:
                adadelta_train_acc[epoch], adadelta_train_loss[epoch] \
                    = common.test(data.adj_train, data.data_train.x, w, num_layers, data.data_train.y, None, None,
                                  None, multi_label=True)
                adadelta_test_acc[epoch], adadelta_test_loss[epoch] \
                    = common.test(data.adj_test, data.data_test.x, w, num_layers, data.data_test.y, None, None,
                                  None, multi_label=True)
            print("adadelta_train_acc={}".format(adadelta_train_acc[epoch]))
            print("adadelta_test_acc={}".format(adadelta_test_acc[epoch]))
        torch.save(
            {"adadelta_train_acc": adadelta_train_acc, "adadelta_train_loss": adadelta_train_loss,
             "adadelta_test_acc": adadelta_test_acc, "adadelta_test_loss": adadelta_test_loss},
            'gcn_adadelta_' + args.dataset + '_performance_' + repr(args.hidden) + '_' + repr(num_layers[0]) + '.pt')
        # torch.save(
        #     {"w": w, "v_w": v_w, 'epoch':epoch+1},
        #     'gcn_adadelta_' + args.dataset + '_variable_' + repr(args.hidden) + '_' + repr(num_layers[0]) + '.pt')
        #

    mem = psutil.virtual_memory()
    print("gd memeory used:{:.6f}GB".format((mem.used - mem_init.used) / 1024 / 1024 / 1024))
    print("training time per epoch={}".format(time_sum / args.epochs))
    # adadelta_train_loss = adadelta_train_loss[1:]
    # adadelta_test_loss = adadelta_test_loss[1:]
    # adadelta_train_acc = adadelta_train_acc[1:]
    # adadelta_test_acc = adadelta_test_acc[1:]


if __name__ == "__main__":
    adam_train()
    # adagrad_train()
    # gd_train()
    # adadelta_train()
