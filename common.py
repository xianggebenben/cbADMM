import torch.nn.functional as F
import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from torch_sparse import SparseTensor, fill_diag, sum, mul, cat, masked_select, spmm
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from torch.autograd import Variable
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch.optim as optim
import copy
import os
from configparser import ConfigParser
from sklearn import metrics
# read config file
config = ConfigParser()
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config.read(os.path.join(BASE_DIR, 'config.ini'), encoding='utf-8')
except:
    config.read('config.ini', encoding='utf-8')


device = torch.device(config['common']['device'])

# normalize adj matrix
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        # return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        n = torch.max(edge_index[0]) + 1  # num of nodes
        adj_norm = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
                                sparse_sizes=(n, n))
        return adj_norm



class AdjDivider(torch.utils.data.DataLoader):
    r"""rewritten form 'ClusterLoader'
    Args:
        cluster_data (torch_geometric.data.ClusterData): The already
            partioned data object.
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, cluster_data, **kwargs):
        self.cluster_data = cluster_data

        super(AdjDivider,
              self).__init__(range(len(cluster_data)),
                             collate_fn=self.__collate__, **kwargs)

    def __collate__(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        N = self.cluster_data.data.num_nodes
        E = self.cluster_data.data.num_edges

        start = self.cluster_data.partptr[batch].tolist()
        end = self.cluster_data.partptr[batch + 1].tolist()
        node_idx = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])

        data = copy.copy(self.cluster_data.data)

        adj, data.adj = self.cluster_data.data.adj, None
        adj = cat([adj.narrow(0, s, e - s) for s, e in zip(start, end)], dim=0)

        adj1 = adj.index_select(1, node_idx)
        row, col, edge_idx = adj1.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        data.adj = adj1.to(device)
        data.node_index = node_idx

        for key, item in data:
            try:
                if item.size(0) == N:
                    data[key] = item[node_idx]
                if item.size(0) == E:
                    data[key] = item[edge_idx]
            except:
                pass
        data.adj_row = adj
        return data

def az(adj, z):
    if isinstance(adj, SparseTensor):
        row, col, adj_value = adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        row_size = adj.size(dim=0)
        col_size = adj.size(dim=1)
        return spmm(edge_index, adj_value, row_size, col_size, z)

def azw(adj, z, w):
    if isinstance(adj, SparseTensor):
        row, col, adj_value = adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        row_size = adj.size(dim=0)
        col_size = adj.size(dim=1)
        f1, f2 = w.size()
        if f1 - f2 > 0:  # A(ZW) would be more efficient
            temp = z.matmul(w)
            return spmm(edge_index, adj_value, row_size, col_size, temp)
        else:  # (AZ)W would be more efficient
            temp = spmm(edge_index, adj_value, row_size, col_size, z)
            return temp.matmul(w)
    elif adj == 0:
        return 0
    else:  # adj is dense matrix
        return adj.matmul(z.matmul(w))


def sparse_mask(adj, train_mask):
    return masked_select(adj, 0, train_mask)


def forwardprop(adj, x, w, num_layers):
    length = len(num_layers)
    if length > 0:
        if length == 1:  # only 1 subnet
            if num_layers[0] > 0:
                for i in range(num_layers[0] - 1):
                    x = F.relu(azw(adj, x, w[0][i]))
                return azw(adj, x, w[0][-1])
        else:  # multiple subnets
            for i in range(length - 1):
                if num_layers[i] > 0:
                    for j in range(num_layers[i]):  # for each layer in subnet i (i<length-1)
                        x = F.relu(azw(adj, x, w[i][j]))
            for j in range(num_layers[-1] - 1):  # last subnet
                x = F.relu(azw(adj, x, w[-1][j]))
            return azw(adj, x, w[-1][-1])


def forwardprop_parallel(adj, x, w, num_layers):
    length = len(num_layers)

    if length > 0:
        assert length == 1, 'only support 1 subnet currently'
        if num_layers[0] > 0:
            for i in range(num_layers[0] - 1):
                if i == 0:
                    x = F.relu(x.matmul(w[0][i]))
                else:
                    x = F.relu(azw(adj, x, w[0][i]))
            return azw(adj, x, w[0][-1])



def f1_score(label, pred, average = True):
    tp = (label * pred).sum().item()
    fp = ((1-label)*pred).sum().item()
    fn = (label * (1-pred)).sum().item()
    if average:
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
        return f1
    else:
        return tp, fp, fn


def multi_class_test(pred_train, label_train, average):
    pred_train = F.log_softmax(pred_train, dim=1)
    train_loss = F.nll_loss(pred_train, label_train, reduction='mean')
    pred_train = pred_train.argmax(dim=1)
    train_acc = pred_train.eq(label_train)
    train_acc = float(train_acc.sum().item())
    if average:
        train_acc /= len(pred_train)
    return train_acc, train_loss


def multi_label_test(pred_train, label_train, average):
    train_loss = F.binary_cross_entropy_with_logits(pred_train, label_train, reduction='sum')
    pred_train = (pred_train > 0).float()
    # micro_f1_train = metrics.f1_score(label_train.cpu(), pred_train.cpu(), average='micro')
    micro_f1_train = f1_score(label_train, pred_train, average)
    return micro_f1_train, train_loss


def test(adj, x, w, num_layers, label_train, label_test=None, mask_train=None, mask_test=None, multi_label=False,average=True):
    pred = forwardprop(adj, x, w, num_layers)
    val = test_parallel(pred, label_train, label_test, mask_train, mask_test, multi_label,average)
    if len(val) == 2:
        return val[0], val[1]
    else:
        assert len(val) == 4, 'length must be 2 or4'
        return val[0], val[1], val[2], val[3]


def test_parallel(z_aggr, label_train, label_test=None, mask_train=None, mask_test=None, multi_label=False, average=True):
    if label_test is not None and mask_train is not None and mask_test is not None:
        pred_train = z_aggr[mask_train]
        pred_test = z_aggr[mask_test]

        if not multi_label:
            train_acc, train_loss = multi_class_test(pred_train, label_train,average)
            test_acc, test_loss = multi_class_test(pred_test, label_test,average)
            return train_acc, test_acc, train_loss, test_loss
        else:
            micro_f1_train, train_loss = multi_label_test(pred_train, label_train,average)
            micro_f1_test, test_loss = multi_label_test(pred_test, label_test,average)
            return micro_f1_train, micro_f1_test, train_loss, test_loss

    else:
        assert multi_label, 'only support ppi'
        micro_f1_train, train_loss = multi_label_test(z_aggr, label_train,average)
        return micro_f1_train, train_loss





# augmented lagrangian
def objective(adj, x, z, w, num_layers, y, mu, rho, label_onehot, train_mask=None, multi_label=False):  # modified!
    if train_mask is not None:
        if not multi_label:
            loss, _ = cross_entropy_with_softmax(z[-1][train_mask], label_onehot)  # modified!
        else:
            loss = F.binary_cross_entropy_with_logits(z[-1][train_mask], label_onehot, reduction='sum')
    else:
        if not multi_label:
            loss, _ = cross_entropy_with_softmax(z[-1], label_onehot)  # modified!
        else:
            loss = F.binary_cross_entropy_with_logits(z[-1], label_onehot, reduction='sum')
    length = len(num_layers)
    if length == 1:
        obj = loss + phi_w_subnet(adj, w[0], rho, train_mask, mu, x, z[0], num_layers[0], y)
        merge_index=None
    else:
        obj = loss + phi_w_subnet(adj, w[0], rho, train_mask, mu, x, z[0], num_layers[0])
        merge_index=None
        max_gap=0
        for i in range(1, length - 1):
            gap=phi_w_subnet(adj, w[i], rho, train_mask, mu, z[i - 1], z[i], num_layers[i])
            obj += gap
            if gap>=max_gap:
                merge_index=i
                max_gap=gap
        obj+=phi_w_subnet(adj, w[length - 1], rho, train_mask, mu, z[length - 2], z[length - 1],
                            num_layers[length - 1], y)
        gap=phi_w_subnet(adj, w[length - 1], rho, train_mask, mu, z[length - 2], z[length - 1],
                            num_layers[length - 1], torch.zeros(size=y.size()).to(device))
        if gap >= max_gap:
            merge_index = length-1
    return obj, merge_index


# initialize w for a subnet
def subnet_w(num_layer, nhid, seed, nfeat=None, nclass=None):
    w = []
    if nfeat:  # first layer
        if num_layer == 1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            w.append(Parameter(torch.rand(nfeat, nhid)).to(device))
            glorot(w[0])
            return w
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            w.append(Parameter(torch.rand(nfeat, nhid)).to(device))
            glorot(w[0])
            for i in range(1, num_layer - 1):
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                w.append(Parameter(torch.rand(nhid, nhid)).to(device))
                glorot(w[i])
    else:
        for i in range(num_layer - 1):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            w.append(Parameter(torch.rand(nhid, nhid)).to(device))
            glorot(w[i])

    if nclass:  # last layer
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        w.append(Parameter(torch.rand(nhid, nclass)).to(device))
        glorot(w[-1])
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        w.append(Parameter(torch.rand(nhid, nhid)).to(device))
        glorot(w[-1])
    return w


# initialize z for all subnets
def subnet_z(adj, x, w, num_layers):
    '''
    :return: list of subnet outputs
    '''
    temp = x.clone()
    z = []
    length = len(num_layers)
    if length > 0:
        if length == 1:  # only 1 subnet
            if num_layers[0] > 0:
                for i in range(num_layers[0] - 1):
                    temp = F.relu(azw(adj, temp, w[0][i]))
                z.append(azw(adj, temp, w[0][-1]))
        else:  # multiple subnets
            for i in range(length - 1):
                if num_layers[i] > 0:
                    for j in range(num_layers[i]):  # for each layer in subnet i (i<length-1)
                        temp = F.relu(azw(adj, temp, w[i][j]))
                    z.append(temp)

            for j in range(num_layers[-1] - 1):  # last subnet
                temp = F.relu(azw(adj, temp, w[-1][j]))
            z.append(azw(adj, temp, w[-1][-1]))
    return z

# initialize z for all subnets (community-wise parallel version)
def subnet_z_parallel(batch_list, w, num_layers, num_parts):
    '''
        :return: list of subnet outputs
    '''
    z = []
    y = []
    length = len(num_layers)
    if length > 0:
        assert length == 1, 'only support one subnet currently!'
        if num_layers[0] > 0:
            for j in range(num_parts): # initialize z for each community
                temp = batch_list[j].x
                for i in range(1, num_layers[0]):
                    temp = F.relu(azw(batch_list[j].adj_full, temp, w[0][i-1]))
                z.append(azw(batch_list[j].adj_full, temp, w[0][-1]))
                y.append(torch.zeros_like(z[-1]))

    return z, y


def gcn(adj, x, nfeat, nhid, nclass, seed, num_layers, return_z=True):
    w = []
    length = len(num_layers)
    if length > 0:
        if length == 1:  # only 1 subnet
            if num_layers[0] > 0:
                w.append(subnet_w(num_layers[0], nhid, seed, nfeat=nfeat, nclass=nclass))

        else:  # multiple subnets
            for i in range(length):
                if num_layers[i] > 0:
                    if i == 0:  # first subnet
                        w.append(subnet_w(num_layers[i], nhid, seed, nfeat=nfeat, nclass=None))
                    elif i == length - 1:  # last subnet
                        w.append(subnet_w(num_layers[i], nhid, seed, nfeat=None, nclass=nclass))
                    else:  # middle subnets
                        w.append(subnet_w(num_layers[i], nhid, seed))
    if return_z == True:
        z = subnet_z(adj, x, w, num_layers)
        return w, z
    else:
        return w


def update_z(adj, w, rho, train_mask, mu, x, z, num_layers, y, label_train_one_hot, multi_label=None):
    length = len(num_layers)
    assert length > 0, "num_layers should be nonempty"
    if length == 1:  # only 1 subnet
        z[0]=update_z2_subnet(adj, w[0], rho, train_mask, mu, x, z[0], num_layers[0], y,
                              label_train_one_hot, multi_label)
    else:  # multiple subnets
        if length == 2:
            z[0]=update_z1_subnet(adj, [w[0], w[1]], rho, train_mask, mu, [x, z[0], z[1]],
                                 [num_layers[0], num_layers[1]], y)
        else:
            z[0]=update_z1_subnet(adj, [w[0], w[1]], rho, train_mask, mu, [x, z[0], z[1]],
                                 [num_layers[0], num_layers[1]])
            for i in range(1, length - 1):
                if i == length - 2:
                    z[i]=update_z1_subnet(adj, [w[i], w[i + 1]], rho, train_mask, mu, [z[i - 1], z[i], z[i + 1]],
                                         [num_layers[i], num_layers[i + 1]], y)
                else:
                    z[i]=update_z1_subnet(adj, [w[i], w[i + 1]], rho, train_mask, mu, [z[i - 1], z[i], z[i + 1]],
                                         [num_layers[i], num_layers[i + 1]])
        z[length-1]=update_z2_subnet(adj, w[length - 1], rho, train_mask, mu, z[length - 2], z[length - 1],
                             num_layers[length - 1], y,
                             label_train_one_hot)
    return z



def update_z_parallel(rho,  z, z_aggr, num_layers, y, label_train_one_hot, multi_mask=False):
    length = len(num_layers)
    assert length > 0, "num_layers should be nonempty"
    assert length == 1, 'only support 1 subnet'  # only 1 subnet
    z = update_z2_subnet_parallel(rho,  z, z_aggr,  y, label_train_one_hot, multi_mask)
    return z




def update_z2_subnet(adj, w, rho, train_mask, mu, z1, z2_old, num_layer, y, label_train_one_hot, multi_label=None):
    MAX_ITER = 200
    z2 = Variable(z2_old, requires_grad=True)
    optimizer = optim.Adam([z2], lr=1e-3)
    if multi_label:
        if train_mask is not None:
            loss=F.binary_cross_entropy_with_logits(z2[train_mask], label_train_one_hot, reduction='sum')
        else:
            loss = F.binary_cross_entropy_with_logits(z2, label_train_one_hot, reduction='sum')
    else:
        if train_mask is not None:
            loss=cross_entropy_with_softmax(z2[train_mask], label_train_one_hot)[0]
        else:
            loss = cross_entropy_with_softmax(z2, label_train_one_hot)[0]
    fz2_initial = loss + phi_w_subnet(adj, w, rho, train_mask, mu, z1, z2, num_layer, y)
    for i in range(MAX_ITER):
        optimizer.zero_grad()
        if multi_label:
            if train_mask is not None:
                loss = F.binary_cross_entropy_with_logits(z2[train_mask], label_train_one_hot, reduction='sum')
            else:
                loss = F.binary_cross_entropy_with_logits(z2, label_train_one_hot, reduction='sum')
        else:
            if train_mask is not None:
                loss = cross_entropy_with_softmax(z2[train_mask], label_train_one_hot)[0]
            else:
                loss = cross_entropy_with_softmax(z2, label_train_one_hot)[0]
        fz2 = loss + phi_w_subnet(adj, w, rho, train_mask, mu, z1, z2, num_layer, y)
        torch.autograd.backward(fz2, inputs=[z2])
        optimizer.step()
    if multi_label:
        if train_mask is not None:
            loss=F.binary_cross_entropy_with_logits(z2[train_mask], label_train_one_hot, reduction='sum')
        else:
            loss = F.binary_cross_entropy_with_logits(z2, label_train_one_hot, reduction='sum')
    else:
        if train_mask is not None:
            loss=cross_entropy_with_softmax(z2[train_mask], label_train_one_hot)[0]
        else:
            loss = cross_entropy_with_softmax(z2, label_train_one_hot)[0]
    fz2 = loss + phi_w_subnet(adj, w, rho, train_mask, mu, z1, z2, num_layer, y)
    if fz2>fz2_initial:
        z2=z2_old
    return z2


def update_z2_subnet_parallel(rho, z2_old, z_aggr, y, label_train_one_hot, multi_task):
    MAX_ITER = 200
    z2 = Variable(z2_old, requires_grad=True)
    optimizer = optim.Adam([z2], lr=1e-3)
    if multi_task:
        fz2_initial = F.binary_cross_entropy_with_logits(z2, label_train_one_hot, reduction='sum') \
                      + phi_z2_parallel(z2 - z_aggr, y, rho)
    else:
        fz2_initial=cross_entropy_with_softmax(z2, label_train_one_hot)[0]\
                + phi_z2_parallel(z2-z_aggr, y, rho)
    for i in range(MAX_ITER):
        # z2_train = Variable(z2_train, requires_grad=True)
        optimizer.zero_grad()
        if multi_task:
            fz2 = F.binary_cross_entropy_with_logits(z2, label_train_one_hot, reduction='sum') \
                          + phi_z2_parallel(z2 - z_aggr, y, rho)
        else:
            fz2=cross_entropy_with_softmax(z2, label_train_one_hot)[0]\
                    + phi_z2_parallel(z2-z_aggr, y, rho)
        #print("fz2: {}".format(fz2))
        # fz2.backward()
        torch.autograd.backward(fz2, inputs=[z2])
        optimizer.step()
    if multi_task:
        fz2 = F.binary_cross_entropy_with_logits(z2, label_train_one_hot, reduction='sum') \
              + phi_z2_parallel(z2 - z_aggr, y, rho)
    else:
        fz2 = cross_entropy_with_softmax(z2, label_train_one_hot)[0] \
              + phi_z2_parallel(z2 - z_aggr, y, rho)
    if fz2>fz2_initial:
        z2=z2_old
    return z2


def phi_z2_parallel(temp, y, rho):
    return torch.einsum("ij, ij ->", (y + rho / 2 * temp), temp)


def update_z1_subnet(adj, w, rho, train_mask, mu, z, num_layers, y=None):
    '''
    :param x: input for a subnet
    :param z1: output for a subnet
    :return: a list of updated parameters for a subnet
    '''
    z = [Variable(i.clone(), requires_grad=True) for i in z]
    MAX_ITER=200
    optimizer=optim.Adam([z[1]],lr=1e-3)
    initial_loss = phi_z_subnet(adj, w, rho, train_mask, mu, z, num_layers, y)
    for i in range(MAX_ITER):
        optimizer.zero_grad()  # 1.
        loss = phi_z_subnet(adj, w, rho, train_mask, mu, z, num_layers, y)
        torch.autograd.backward(loss, inputs=[z[1]])
        optimizer.step()
    loss = phi_z_subnet(adj, w, rho, train_mask, mu, z, num_layers, y)
    #if loss>initial_loss:
    #    z = [Variable(i.clone(), requires_grad=True) for i in z]
    return z[1]


def phi_z_subnet(adj, w, rho, train_mask, mu, z, num_layers, y=None):
    loss = phi_w_subnet(adj, w[0], rho, train_mask, mu, z[0], z[1], num_layers[0]) \
           + phi_w_subnet(adj, w[1], rho, train_mask , mu, z[1], z[2], num_layers[1],y)
    return loss



def cross_entropy_with_softmax(zl, label_onehot):
    # prob = F.log_softmax(zl, dim=1)
    # loss = F.nll_loss(prob, label, reduction="sum")
    prob = F.softmax(zl, dim=1)
    loss = - torch.einsum("ij, ij->", label_onehot, torch.log(prob + 1e-10))
    return loss, prob


def phi_w_subnet(adj, w, rho, train_mask, mu, x, z1,  num_layer, y=None):
    temp = x.clone()
    '''
    :param x: input for a subnet
    :param z1: output for a subnet
    '''
    if y is not None:  # the last subnet
        temp=forwardprop(adj,temp,[w],[num_layer])
        if train_mask is not None:
            temp = (z1[train_mask] - temp[train_mask])
        else:
            temp = z1 - temp
        if mu == 0:
            if train_mask is not None:
                return torch.einsum("ij, ij ->", (y[train_mask] + rho / 2 * temp), temp)
            else:
                return torch.einsum("ij, ij ->", (y + rho / 2 * temp), temp)

        else:
            if train_mask is not None:
                temp = torch.einsum("ij, ij ->", (y[train_mask] + rho / 2 * temp), temp)
            else:
                temp = torch.einsum("ij, ij ->", (y + rho / 2 * temp), temp)
            for i in range(num_layer):
                temp += mu / 2 * torch.norm(w[i] ** 2)
            return temp

    else:  # not the last subnet
        for i in range(num_layer):
            temp = F.relu(azw(adj, temp, w[i]))
        if train_mask is not None:
            temp = z1[train_mask] - temp[train_mask]
        else:
            temp = z1 - temp
        if mu == 0:
            return torch.einsum("ij, ij ->", (rho / 2 * temp), temp)
        else:
            temp = torch.einsum("ij, ij ->", (rho / 2 * temp), temp)
            for i in range(num_layer):
                temp += mu / 2 * torch.norm(w[i] ** 2)
            return temp



def phi_w_subnet_parallel(adj, w, rho, train_mask, mu, z_aggr, z1,  num_layer, y=None):
    temp = z_aggr.clone()
    '''
    :param x: input for a subnet
    :param z1: output for a subnet
    '''
    if y is not None:  # the last subnet
        temp=forwardprop_parallel(adj,temp,[w],[num_layer])
        if train_mask is not None:
            temp = (z1[train_mask] - temp[train_mask])
        else:
            temp = z1 - temp
        if mu == 0:
            if train_mask is not None:
                return torch.einsum("ij, ij ->", (y[train_mask] + rho / 2 * temp), temp)
            else:
                return torch.einsum("ij, ij ->", (y + rho / 2 * temp), temp)

        else:
            if train_mask is not None:
                temp =  torch.einsum("ij, ij ->", (y[train_mask] + rho / 2 * temp), temp)
            else:
                temp =  torch.einsum("ij, ij ->", (y + rho / 2 * temp), temp)
            for i in range(num_layer):
                temp += mu / 2 * torch.norm(w[i] ** 2)
            return temp


def u_w_subnet(w_old, w_new, tau, gradient, loss):
    # here w_old, w_new are all Tensors, not list.
    temp = w_new - w_old
    f = loss + torch.einsum("ij, ij ->", (gradient + tau / 2 * temp), temp)
    return f



def update_w_subnet(adj, x, z, w_old, rho, train_mask, mu, num_layer, y=None):
    '''
    :param x: input for a subnet
    :param z: output for a subnet
    :return: a list of updated parameters for a subnet
    '''
    beta = [Variable(w_old[j].clone(), requires_grad=True) for j in range(num_layer)]
    optimizer = optim.Adam(beta, lr=1e-3)
    MAX_ITER=50
    initial_loss=phi_w_subnet(adj, beta, rho, train_mask, mu, x, z, num_layer, y)
    for i in range(MAX_ITER):
        optimizer.zero_grad()  # 1.
        loss = phi_w_subnet(adj, beta, rho, train_mask, mu, x, z, num_layer, y)
        torch.autograd.backward(loss, inputs=beta)
        optimizer.step()
    loss = phi_w_subnet(adj, beta, rho, train_mask, mu, x, z, num_layer, y)
    if loss>initial_loss:
        beta=[Variable(w_old[j].clone(), requires_grad=True) for j in range(num_layer)]
    return beta


def update_w_subnet_parallel(adj, z_aggr, z, w_old, rho, train_mask, mu, num_layer, y=None):
    '''
    :param x: input for a subnet
    :param z: output for a subnet
    :return: a list of updated parameters for a subnet
    '''
    beta = [Variable(w_old[j].clone(), requires_grad=True) for j in range(num_layer)]
    optimizer = optim.Adam(beta, lr=1e-3)
    MAX_ITER=50
    initial_loss=phi_w_subnet_parallel(adj, beta, rho, train_mask, mu, z_aggr, z, num_layer, y)
    for i in range(MAX_ITER):
        optimizer.zero_grad()  # 1.
        loss = phi_w_subnet_parallel(adj, beta, rho, train_mask, mu, z_aggr, z, num_layer, y)
        torch.autograd.backward(loss, inputs=beta)
        optimizer.step()
    loss = phi_w_subnet_parallel(adj, beta, rho, train_mask, mu, z_aggr, z, num_layer, y)
    if loss>initial_loss:
        beta=[Variable(w_old[j].clone(), requires_grad=True) for j in range(num_layer)]
    return beta



def update_w(adj, x, z, w_old, rho, train_mask, mu, num_layers, y):
    w_new = []
    length = len(num_layers)
    if length > 0:
        if length == 1:  # only 1 subnet
            w_new.append(update_w_subnet(adj, x, z[0], w_old[0], rho, train_mask,
                                         mu, num_layers[0], y))
        else:  # multiple subnets
            w_new.append(update_w_subnet(adj, x, z[0], w_old[0], rho, train_mask,
                                         mu, num_layers[0]))
            for i in range(1, length - 1):
                w_new.append(update_w_subnet(adj, z[i - 1], z[i], w_old[i], rho, train_mask,
                                             mu, num_layers[i]))
            w_new.append(update_w_subnet(adj, z[-2], z[-1], w_old[-1], rho, train_mask,
                                         mu, num_layers[-1], y))
    return w_new



def update_w_parallel(adj,z_aggr, z, w_old, rho, train_mask, mu, num_layers, y):
    w_new = []
    length = len(num_layers)
    if length > 0:
        assert length == 1, 'only support 1 subnet currently'
        w_new.append(update_w_subnet_parallel(adj,z_aggr, z[0], w_old[0], rho, train_mask,
                                     mu, num_layers[0], y))
    return w_new


def update_y(adj, x, z, w, y, rho, num_layers, train_mask=None):
    if len(num_layers) == 1:
        temp = x
    else:
        temp = z[-2]
    temp =forwardprop(adj,temp,[w[-1]],[num_layers[-1]])
    if train_mask is not None:
        r = z[-1][train_mask] - temp[train_mask]
        y = y.detach()
        y[train_mask] = y[train_mask] + rho * r
    else:
        r = z[-1] - temp
        y = y.detach()
        y = y + rho * r
    r = torch.sum(r ** 2)
    return y, r


def update_y_parallel(z, z_aggr, y, rho, num_layers):
    assert len(num_layers) == 1, 'only support 1 subnet currently'
    r = z - z_aggr
    # y = y.detach()
    y = y + rho * r
    r = torch.sum(r ** 2)
    return y, r


def merge_subnet(w, z, num_layers, i):
    w_new = []
    z_new = []
    num_layers_new=[]
    assert len(w) > 1 and len(z) > 1 and len(num_layers) > 1,"should be at least two subnetworks"
    assert len(w)== len(z) and len(num_layers)==len(z)," the number of subnetworks should be consistent"
    assert i<len(num_layers), "should be within the number of subnetworks"
    for k in range(len(num_layers)):
        if k!=i-1 and k!=i:
            w_new.append(w[k])
            z_new.append(z[k])
            num_layers_new.append(num_layers[k])
        else:
            if k==i:
                w_new.append(w[k-1]+w[k])
                z_new.append(z[k])
                num_layers_new.append(num_layers[k-1]+num_layers[k])
    return w_new,z_new,num_layers_new


def concat_batch(t, z1, num_parts, node_perm_inverse):
    z1_all = torch.cat([z1[i] for i in range(num_parts)], dim=0)
    z1_all = z1_all[node_perm_inverse]
    return z1_all

def node_perm(batch_list, perm, num_batches, num_batches_per_part):
    node_perm_inverse_list = []
    node_perm_list = []
    temp2 = torch.range(0, num_batches - 1, num_batches_per_part, dtype=int)
    for i in range(num_batches_per_part):
        temp = torch.cat([batch_list[temp2[j].item()].node_index for j in range(len(temp2))], dim=0)
        temp3 = perm[temp]
        sortsort = torch.sort(temp3)
        temp3_value, temp3_inverse = sortsort.values, sortsort.indices
        node_perm_list.append(temp3_value)
        node_perm_inverse_list.append(temp3_inverse)
        temp2 += 1
    return node_perm_list, node_perm_inverse_list
