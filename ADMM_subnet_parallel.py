import threading
import psutil
from psutil._common import bytes2human

from configparser import ConfigParser
from tornado.tcpclient import TCPClient
from tornado.ioloop import IOLoop
from tornado import gen,concurrent
import pickle
import codecs
import pyarrow.plasma as plasma
import logging
from multiprocessing import Pool
import csv
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
import time
import common, input_data
from torch_geometric.data import ClusterData, Data
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# read config file
config = ConfigParser()
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config.read(os.path.join(BASE_DIR, 'config.ini'), encoding='utf-8')
except:
    config.read('config.ini', encoding='utf-8')
seed = config.getint('common', 'seed')
epochs_origin = config.getint('common', 'epochs')
hidden = config.getint('common', 'hidden')
dataset_name = config['common']['dataset']
num_parts = config.getint('common', 'num_parts')
num_batches_per_part = 1
chunks = config.getint('common','chunks')
torch.manual_seed(seed)
current_community = config.getint('currentCommunity', 'community')
client = plasma.connect(config['common']['plasma_path'])
epoch_per_comm = config.getint('common', 'epoch_for_comm')


logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(str(current_community) + '.log')
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

sentinel = b'---end---'
# stochastic ADMM
epochs = epochs_origin  # for full-batch update
num_batches = num_batches_per_part * num_parts



def init_io_stat():
    logger.info("I/O bytes:")
    stats = psutil.net_if_stats()
    io_counters = psutil.net_io_counters(pernic=True)
    for nic, addrs in psutil.net_if_addrs().items():
        print("%s:" % (nic))
        if nic in io_counters:
            io = io_counters[nic]
            print("    incoming       : ", end='')
            print("bytes=%s, pkts=%s, errs=%s, drops=%s" % (
                bytes2human(io.bytes_recv), io.packets_recv, io.errin,
                io.dropin))
            print("    outgoing       : ", end='')
            print("bytes=%s, pkts=%s, errs=%s, drops=%s" % (
                bytes2human(io.bytes_sent), io.packets_sent, io.errout,
                io.dropout))
        print("")
        return io_counters


def io_stat(io_counters_old):
    logger.info("I/O bytes:")
    io_counters = psutil.net_io_counters(pernic=True)
    for nic, addrs in psutil.net_if_addrs().items():
        print("%s:" % (nic))
        if nic in io_counters:
            io = io_counters[nic]
            io_old = io_counters_old[nic]
            print("    incoming       : ", end='')
            print("bytes=%s, pkts=%s, errs=%s, drops=%s" % (
                bytes2human(io.bytes_recv - io_old.bytes_recv),
                io.packets_recv - io_old.packets_recv,
                io.errin - io_old.errin,
                io.dropin - io_old.dropin))
            print("    outgoing       : ", end='')
            print("bytes=%s, pkts=%s, errs=%s, drops=%s" % (
                bytes2human(io.bytes_sent - io_old.bytes_sent),
                io.packets_sent - io_old.packets_sent,
                io.errout - io_old.errout,
                io.dropout - io_old.dropout))
        print("")
        return io_counters


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()

def relu(x):
    return np.maximum(x, 0)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
# return cross entropy
def cross_entropy(label, prob):
    loss = -np.sum(label * np.log(prob))
    return loss
# return the cross entropy loss function
def cross_entropy_with_softmax(label, z):
    prob = softmax(z)
    loss = cross_entropy(label, prob)
    return loss

async def send_parameter(ip, data):
    stream = await TCPClient().connect(ip, 8888)  # host: ip; port: 8888
    await stream.write(data)  # Asynchronously write the given data to this stream
    # await gen.sleep(0)


async def send_splitted_parameter(ip, data, parameter):
    logger.info('Sent parameter to %s ', ip)
    all_works = []
    chunked_array = np.vsplit(data, chunks)

    for index, arr in enumerate(chunked_array):
        new_para = parameter + '|' + str(index + 1).zfill(2)

        logger.info('Sent parameter %s to %s ', new_para, ip)
        all_works.append(send_parameter(ip, pickle.dumps(arr) + str.encode(new_para) + sentinel))
    await gen.multi(all_works)


# For the initialization, we use multiprocessing to send the whole data.
async def send_whole_parameter(ip, data, parameter):
    stream = await TCPClient().connect(ip, 8888)
    if parameter != 'xtrain' and parameter != 'ytrain' and parameter != 'xtest' and parameter != 'ytest' \
        and parameter[0] != "i"\
            and parameter[0:5] != "epoch"\
            and parameter[0:9] != "epoch_glo":
        new_para = parameter + '|00'
    else:
        new_para = parameter.zfill(12)
    logger.info('Sent parameter %s to %s', new_para, ip)
    await stream.write(pickle.dumps(data) + str.encode(new_para) + sentinel)
    # gen.sleep(10)


def start_send_whole_parameter(ip, data, parameter):
    # if layer != 0:
    #     data = tensor_to_numpy([data])[0]
    IOLoop.current().run_sync(lambda: send_whole_parameter(ip, data, parameter))





def start_send_splitted_parameter(ip, data, parameter):
    # if layer != 0:
    #     data = tensor_to_numpy([data])[0]
    logger.info("start to send splitted para : %s", parameter)
    IOLoop.current().run_sync(lambda: send_splitted_parameter(ip, data, parameter))



def check_existance(para_name):
    for i in range(chunks):
        if not client.contains(plasma_id(para_name + '|' + str(i + 1).zfill(2))):
            return False
    return True


def check_one_existance(para_name):
    return client.contains(plasma_id(para_name))


def aggregate_para(para_name):
    all = []
    for i in range(chunks):
        all.append(get_value(para_name + '|' + str(i + 1).zfill(2)))
        # client.delete([plasma_id(para_name+'|'+str(i+1).zfill(2))])
    aggregated_para = numpy_to_tensor([np.concatenate(all)])[0]
    return aggregated_para


# id in client.list()
def plasma_id(name):
    return plasma.ObjectID(8 * b'0' + str.encode(name))


def get_value(name):
    value = np.array(client.get(plasma_id(name)))
    delete_value(name)
    return value


def get_value_wo_delete(name):
    value = np.array(client.get(plasma_id(name)))
    # delete_value(name)
    return value


def delete_value(name):
    client.delete([plasma_id(name)])


def numpy_to_tensor(list):
    tran_start_time = time.time()
    new_list = [torch.from_numpy(i).float().to(device) for i in list]
    global tran_time
    tran_time += (time.time() - tran_start_time)
    return new_list


def tensor_to_numpy(list):
    tran_start_time = time.time()
    new_list = [i.detach().cpu().numpy().astype(np.float32) for i in list]
    global tran_time
    tran_time += (time.time() - tran_start_time)
    return new_list


def tensor_to_numpy_int(list):
    tran_start_time = time.time()
    new_list = [i.numpy().astype(np.int8) for i in list]
    global tran_time
    tran_time += (time.time() - tran_start_time)
    return new_list


class MyThread(threading.Thread):
    def __init__(self, target, args=()):
        super(MyThread, self).__init__()
        self.target = target
        self.args = args

    def run(self):
        # sleep(2)
        self.result = self.target(*self.args)

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except Exception:
            return None


if __name__ == '__main__':
    logger.info('Community number is : %d', current_community)

    # =================== import dataset ===================
    # device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device = torch.device(config['common']['device'])
    print('Dataset: {}'.format(dataset_name))
    multi_label = False
    if dataset_name == 'cora':
        dataset = input_data.cora()
        rho = 0.1
        mu = 100
    elif dataset_name == 'pubmed':
        dataset = input_data.pubmed()
        rho = 0.1
        mu = 1
    elif dataset_name == 'citeseer':
        dataset = input_data.citeseer()
        rho = 0.1
        mu = 100
    elif dataset_name == 'amazon_computers':
        dataset = input_data.amazon_computers()
        rho = 0.1
        mu = 0
    elif dataset_name == 'amazon_photo':
        dataset = input_data.amazon_photo()
        rho = 0.1
        mu = 0
    elif dataset_name == 'coauthor_cs':
        dataset = input_data.coauthor_cs()
        rho = 0.1
        mu = 100
    elif dataset_name == 'coauthor_physics':
        dataset = input_data.coauthor_physics()
        rho = 0.1
        mu = 100
    elif dataset_name == 'flickr':
        dataset = input_data.flickr()
        rho = 0.1
        mu = 0
    elif dataset_name == 'ppi':
        dataset = input_data.ppi()
        rho = 0.1
        mu = 0
        multi_label = True
    elif dataset_name == 'reddit2':
        dataset = input_data.reddit2()
        rho = 0.1
        mu = 0
    elif dataset_name == 'ogbn_arxiv':
        dataset = input_data.ogbn_arxiv(class_list=[i for i in range(40)])
        rho = 0.1
        mu = 0

    # ================================= dataset partition =================================

    print("Dataset: {}".format(dataset_name))
    print("{} communities".format(num_parts))

    # do not support multi_label currently
    if not multi_label:
        # dataset.x = dataset.x.to(device)
        #.adj = dataset.adj.to(device)
        dataset.label_train_onehot = dataset.label_train_onehot.to(device)
        dataset.label_train = dataset.label_train.to(device)
        dataset.label_test = dataset.label_test.to(device)
    num_layers = [2]  # if only 1 subnet, then it should have at least 2 layers

    data = Data(x=dataset.x.to('cpu'), y=dataset.label.to('cpu'), edge_index=dataset.edge_index.to('cpu'))
    data.train_mask = dataset.train_mask.to('cpu')
    data.test_mask = dataset.test_mask.to('cpu')
    data.label_train_onehot = dataset.label_train_onehot.to('cpu')

    cluster_data = ClusterData(data, num_parts=num_batches, recursive=False,
                           save_dir=dataset.processed_dir)
    data.train_mask = dataset.train_mask.to(device)
    data.test_mask = dataset.test_mask.to(device)
    data.label_train_onehot = dataset.label_train_onehot.to(device)
    data.edge_index = None
    num_classes = dataset.num_classes
    num_train = (data.train_mask == True).sum().item()
    num_test = (data.test_mask == True).sum().item()

    adj = dataset.adj.to(device)
    del dataset

    # permute 'adj' and substitute for cluster_data.data.adj
    # cluster_data.data.adj = permute(adj, cluster_data.perm)

    # add perm_node_inverse
    cluster_data.perm_node_inverse = torch.sort(cluster_data.perm).indices
    train_loader = common.AdjDivider(cluster_data, batch_size=1, shuffle=False,
                                     num_workers=0)
    # for all nodes: do graph partitioning
    logger.info('Finish graph partitioning!')


    para_dict = {}
    scheduler_ip = config['community'+str(num_parts)]['server']
    time_csv = str(current_community) + '.csv'
    file = open(time_csv, 'w')
    tran_time = 0
    rho_count = 0
    if current_community == num_parts: # for w
        io_counters_old = init_io_stat()
        avg_time = 0
        avg_train_time = 0
        avg_wait_time = 0
        avg_inference_time = 0
        p = Pool(5)  # process pool; max num of pool = 10
    else:
        p = Pool(5)
        avg_time = 0
        avg_train_time = 0
        avg_wait_time = 0

    # initialize parameters
    w = common.gcn(None, None, data.num_features, hidden, num_classes, seed, num_layers, return_z=False)
    data.x = None
    data.edge_index = None

    # for each community z
    if current_community < num_parts:
        # load partitioned graph
        batch = input_data.load_batch_parallel(train_loader, num_parts, num_batches_per_part, cluster_data.partptr, cluster_data.perm, current_community, num_classes=num_classes)

        logger.info("Current community has {} training samples.".format(batch.adj.size(dim=0)))
        logger.info("Current community has {} neighbors.".format(len(batch.nei_list)))

        #  initialize z and y
        z, y = common.subnet_z_parallel([batch], w, num_layers, 1)
        z = z[0]
        y = y[0]
        # forward propagation
        z_temptemp = []
        for j in range(len(batch.nei_list)):
            z_temptemp.append(common.az(batch.adj_list[j].t(), batch.x))

        # send 1st-layer info
        for j in range(len(batch.nei_list)):
            r = batch.nei_list[j]
            logger.info("Neighbor ID is {}".format(r))
            nei_community_name = str(r).zfill(2) + '_' + str(0).zfill(3)
            ip_address = config['community' + str(r)]['server']
            p.apply_async(start_send_splitted_parameter,
                                          args=(ip_address, tensor_to_numpy([z_temptemp[j]])[0],
                                                'f' + str(current_community).zfill(2) + nei_community_name,))
        del z_temptemp

        # receive 1st-layer info
        z_aggr = common.az(batch.adj.t(), batch.x)
        # print('z_aggr device: {}'.format(z_aggr.device))
        current_community_name = str(current_community).zfill(2) + '_' + str(0).zfill(3)
        for r in batch.nei_list:
            while (1):
                if check_existance('f' + str(r).zfill(2) + current_community_name):
                    z_aggr += aggregate_para('f' + str(r).zfill(2) + current_community_name)
                    break
                else:
                    logger.info('Waiting for {} from community {}!'.format(
                                    'f' + str(r).zfill(2) + current_community_name, r))
                    time.sleep(1)

        # send z_aggr to w
        logger.info('start to send z_aggr to w')
        scheduler_name = str(num_parts).zfill(2) + '_' + str(0).zfill(3)
        p.apply_async(start_send_splitted_parameter,
                      args=(scheduler_ip, tensor_to_numpy([z_aggr])[0],
                            'a' + str(current_community).zfill(2) + scheduler_name,))
        logger.info('finish sending z_aggr to w')

    if current_community == num_parts:
        # for training and test performance
        admm_train_loss = np.zeros(epochs)
        admm_train_acc = np.zeros(epochs)
        admm_test_loss = np.zeros(epochs)
        admm_test_acc = np.zeros(epochs)
        pres = np.zeros(epochs)
        dres = np.zeros(epochs)
        obj = np.zeros(epochs)
        time_avg = 0


        # record node_perm_inverse for the concatenation of the same batch from all partitions
        node_perm_list, node_perm_inverse_list = common.node_perm(input_data.load_batch(train_loader, num_parts,
                                                                                        num_batches_per_part,
                                                                                        cluster_data.partptr,
                                           cluster_data.perm), cluster_data.perm, num_batches,
                                                                  num_batches_per_part)

        # receive z_aggr
        z_aggr = []
        for i in range(num_parts):
            scheduler_name = str(num_parts).zfill(2)+ '_' + str(0).zfill(3)
            while(1):
                nei_community_name = str(i).zfill(2)
                if check_existance('a' + nei_community_name+ scheduler_name):
                    z_aggr.append(aggregate_para('a' + nei_community_name+ scheduler_name))
                    break
                else:
                    logger.info('Waiting for {} from community {}!'.format(
                                    'a' + nei_community_name+ scheduler_name, i))
                    time.sleep(1)

        # concatenate z_aggr
        z_aggr_all = common.concat_batch(0, z_aggr, num_parts, node_perm_inverse_list)

    # for all
    min_epoch = 0
    # print("min_epoch: {}".format(min_epoch))
    for epoch in range(min_epoch, epochs):
        logger.info('=========================== Iter %d ===========================', epoch)
        logger.info('rho is : %f', rho)
        time_dict = dict()
        time_dict['iteration'] = epoch
        start_time = time.time()
        tran_time = 0
        wait_time = 0
        time_send_para_for_training = 0
        t = 0

        # for each community
        if current_community < num_parts:

            # compute 2nd-layer information
            z_aggr_temptemp = []
            for j in range(len(batch.nei_list)):

                z_aggr_temptemp.append(common.az(batch.adj_list[j].t(), F.relu(z_aggr.matmul(w[0][0]))))

            # send 2nd-layer information
            for j in range(len(batch.nei_list)):
                r = batch.nei_list[j]
                logger.info("Neighbor ID is {}".format(r))
                nei_community_name = str(r).zfill(2) + '_' + str(epoch).zfill(3)
                ip_address = config['community' + str(r)]['server']
                p.apply_async(start_send_splitted_parameter,
                              args=(ip_address, tensor_to_numpy([z_aggr_temptemp[j]])[0],
                                    's' + str(current_community).zfill(2) + nei_community_name,))

            # receive 2nd-layer information
            z_aggr_temp = common.az(batch.adj, F.relu(z_aggr.matmul(w[0][0])))
            current_community_name = str(current_community).zfill(2) + '_' + str(epoch).zfill(3)
            for r in batch.nei_list:
                while (1):
                    if check_existance('s' + str(r).zfill(2) + current_community_name):
                        z_aggr_temp += aggregate_para('s' + str(r).zfill(2) + current_community_name)
                        break
                    else:
                        logger.info('Waiting for {} from community {}!'.format(
                            's' + str(r).zfill(2) + current_community_name, r))
                        time.sleep(1)
                        wait_time += 1
            z_aggr_temp = z_aggr_temp.matmul(w[0][1])

            # update z, y
            logger.info("Start updating z and y in parallel")
            t1 = MyThread(target=common.update_z_parallel, args=(rho,  z[batch.train_mask],
                                                                  z_aggr_temp[batch.train_mask],
                                                                  num_layers, y[batch.train_mask],
                                        input_data.onehot(batch.label_train, num_classes),))
            t2 = MyThread(target=common.update_y_parallel, args=(z[batch.train_mask],
                                                               z_aggr_temp[batch.train_mask],
                                                               y[batch.train_mask], rho,
                                                              num_layers))
            t1.start()
            t2.start()
            z.detach()[batch.train_mask] = t1.get_result()
            y[batch.train_mask], _ = t2.get_result()
            logger.info("Finish updating z and y in parallel")

            # compute loss and acc for the current community
            temp = torch.tensor(common.test_parallel(z_aggr_temp, batch.label_train, batch.label_test, batch.train_mask,
                                                     batch.test_mask))


            if (epoch + 1) % epoch_per_comm == 0 or epoch == 1:
                logger.info("Send info to w in epoch {}".format(epoch))


                #  send z, y to update w; send temp for inference
                logger.info('Start to send z, y to update w; send temp for inference')
                ip_address = config['community' + str(num_parts)]['server']
                curr_community_name = str(current_community).zfill(2) + '_' + str(epoch).zfill(3)#'_000'
                p.apply_async(start_send_splitted_parameter,
                          args=(ip_address, tensor_to_numpy([z])[0] , 'z'.zfill(3) + curr_community_name,))
                p.apply_async(start_send_splitted_parameter,
                          args=(ip_address, tensor_to_numpy([y])[0], 'y'.zfill(3) + curr_community_name,))
                p.apply_async(start_send_whole_parameter,
                              args=(ip_address, tensor_to_numpy([temp])[0], (('i'+str(current_community)+str(epoch).zfill(3))),))
                logger.info('Sent z, y, temp')
            else:
                logger.info("Don't exchange info with w in epoch {}".format(epoch))

            # receive w1, w2 from *community num_parts*
            if ((epoch + 1) % epoch_per_comm == 0) or epoch == 1:
                logger.info("Receive info from w in epoch {}".format(epoch))
                while (1):
                    prefix = str(current_community).zfill(2) + '_' + str(epoch).zfill(3)
                    if check_existance('w1'.zfill(3) + prefix) and check_existance('w2'.zfill(3) + prefix):
                        w = [[aggregate_para('w1'.zfill(3) + prefix), aggregate_para('w2'.zfill(3) + prefix)]]
                        break
                    else:
                        logger.info(
                            "Waiting for {} and {} !".format('w1'.zfill(3) + prefix, 'w2'.zfill(3) + prefix))
                        time.sleep(1)
                        wait_time += 1

            else:
                logger.info("Don't exchange info with w in epoch {}".format(epoch))

            t_time = time.time() - start_time - time_send_para_for_training
            logger.info('Iteration %d takes %f ', epoch, (t_time))
            logger.info('Iteration %d wait time %f ', epoch, wait_time)
            logger.info('Iteration %d compute time %f ', epoch,
                        (t_time - wait_time))
            avg_time += t_time
            avg_train_time += t_time - wait_time
            avg_wait_time += wait_time

            time_dict['compute'] = t_time - wait_time
            time_dict['wait'] = wait_time
            time_dict['total_time'] = t_time
            time_dict['tran_time'] = tran_time
            dw = csv.DictWriter(file, time_dict.keys())
            if epoch == 0:
                dw.writeheader()
            dw.writerow(time_dict)
            # finished!

        # for w
        if current_community == num_parts:

            # receive z, y
            # receive info from z
            if ((epoch + 1) % epoch_per_comm == 0 or epoch == 1) or epoch == 0:
                logger.info("Receive info from z in epoch {}".format(epoch))
                # receive z, y from *community 0: num_part -1 *
                z = []
                y = []
                for i in range(num_parts):
                    current_community_name = str(i).zfill(2) + '_' + str(epoch).zfill(3)
                    while (1):
                        if check_existance('z'.zfill(3) + current_community_name) \
                                and check_existance('y'.zfill(3) + current_community_name):
                            z.append(aggregate_para('z'.zfill(3) + current_community_name))
                            y.append(aggregate_para('y'.zfill(3) + current_community_name))
                            break
                        else:
                            logger.info('Waiting for {}  and {}from community {}!'.format(
                                'z'.zfill(3) + current_community_name,
                                'y'.zfill(3) + current_community_name, i))
                            time.sleep(1)
                            wait_time += 1
                try:
                    z = numpy_to_tensor(z)
                    y = numpy_to_tensor(y)
                except:
                    pass
            else:
                logger.info("Don't exchange info with z in epoch {}".format(epoch))


            # concat z1, z2, y2
            z_all = common.concat_batch(0, z, num_parts, node_perm_inverse_list)
            y_all = common.concat_batch(0, y, num_parts, node_perm_inverse_list)

            # update w1, w2
            logger.info("Start updating w")
            w = common.update_w_parallel(adj, z_aggr_all, [z_all], w, rho, data.train_mask, mu, num_layers, y_all)
            logger.info("Finish updating w")

            if ((epoch+1) % epoch_per_comm == 0 or epoch == 1) :
                logger.info("Send w to z in epoch {}".format(epoch))
                # send w1, w2 to *community 0: num_part-1*
                for i in range(num_parts):
                    ip_address = config['community' + str(i)]['server']
                    curr_community_name = str(i).zfill(2) + '_' + str(epoch).zfill(3)
                    p.apply_async(start_send_splitted_parameter,
                                  args=(ip_address, tensor_to_numpy([w[0][0]])[0],
                                        'w1'.zfill(3)+ curr_community_name,))
                    p.apply_async(start_send_splitted_parameter,
                                  args=(ip_address, tensor_to_numpy([w[0][1]])[0],
                                        'w2'.zfill(3) + curr_community_name,))


            # receive inference
            for i in range(num_parts):
                # curr_community_name = str(i).zfill(2) + '_' + str(epoch).zfill(3)  # '_000'
                while 1:
                    if check_one_existance(('i'+str(i)+str(epoch).zfill(3)).zfill(12)):
                        temp = get_value(('i'+str(i)+str(epoch).zfill(3)).zfill(12))
                        admm_train_acc[epoch] += temp[0]
                        admm_test_acc[epoch] += temp[1]
                        admm_train_loss[epoch] += temp[2]
                        admm_test_loss[epoch] += temp[3]
                        # logger.info('Got {}  from community {}!'.format(
                        #     (('i'+str(i)+str(epoch).zfill(3)).zfill(12)), i))
                        break
                    else:
                        logger.info('Waiting for {}  from community {}!'.format(
                            ('i'+str(i)+str(epoch).zfill(3)).zfill(12), i))
                        time.sleep(1)
                        wait_time += 1

            t_inference_start = time.time()
            # compute total loss and total acc
            admm_train_acc[epoch] /= num_train
            admm_train_loss[epoch] /= num_train
            admm_test_acc[epoch] /= num_test
            admm_test_loss[epoch] /= num_test

            logger.info("Train loss: {:.6f}. Train acc: {:.6f}."
                  "Test loss: {:.6f}. Test acc: {:.6f}.".format(
                admm_train_loss[epoch], admm_train_acc[epoch], admm_test_loss[epoch], admm_test_acc[epoch]))
            logger.info("Finish inference")
            t_inference = time.time() - t_inference_start




        # if current_community == num_parts:
            t_time = time.time() - start_time - time_send_para_for_training
            logger.info('Iteration %d takes %f ', epoch, (t_time))
            logger.info('Iteration %d wait time %f ', epoch, wait_time)
            logger.info('Iteration %d compute time %f ', epoch,
                        (t_time - wait_time - t_inference))
            logger.info('Iteration %d inference time %f', epoch, t_inference)
            avg_time += t_time
            avg_train_time += t_time - wait_time
            avg_wait_time += wait_time
            avg_inference_time += t_inference

            time_dict['compute'] = t_time - wait_time - t_inference
            time_dict['wait'] = wait_time
            time_dict['total_time'] = t_time
            time_dict['tran_time'] = tran_time
            time_dict['inference_time'] = t_inference
            dw = csv.DictWriter(file, time_dict.keys())
            if epoch == 0:
                dw.writeheader()
            dw.writerow(time_dict)

        if epoch == epochs-1:
            p.close()
            p.join()
        logger.info('Average time per iteration: {}'.format(avg_time / epochs_origin * (num_batches_per_part)))
        logger.info('Average training time per iteration: {}'.format(avg_train_time / epochs_origin * (num_batches_per_part)))
        logger.info('Average waiting time per iteration: {}'.format(avg_wait_time / epochs_origin * (num_batches_per_part)))
        if current_community == num_parts:
            logger.info('Average inference time per iteration: {}'.format(avg_inference_time / epochs_origin * (num_batches_per_part)))
                # finished!
