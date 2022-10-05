import numpy as np
import math
from abc import abstractmethod
from multiprocessing import Value

from .base_worker_process import BaseWorkerProcess
from .base_network_client import Client
from .utils import socket_recv, socket_send, split_examples
from .cpu_utils import divide_work


class KFoldWorker(BaseWorkerProcess):
    def __init__(
            self,
            conn,
            id,
            ready_val,
            lock,
            test_interval,
            k_fold,
            seed,
            dataset,
            dataset_dir,
            args_file,
            p_data,
            train_idxs,
            test_idxs,
            start_with_work=True):
        self.test_bool = False
        self.train_step = 0
        self._train_step = 0
        self.test_step = 0
        self.test_interval = test_interval
        self.k_fold = k_fold
        self.seed = seed
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.args_file = args_file
        self.p_data = p_data
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs
        super().__init__(
            conn=conn,
            id=id,
            ready_val=ready_val,
            lock=lock,
            start_with_work=start_with_work)
        self.on_init()
        self.load_model(dir="./tmp", name="tmp_net")

    def on_master_progress(self, msg):
        if msg == "train":
            self.test_bool = False
        elif msg == "test":
            self.test_bool = True
        #print("test_bool", self.test_bool)
        self.load_model(dir="./tmp", name="tmp_net")

    def progress(self):
        if self.test_bool:
            #print("test")
            self.test()
            self.test_step += 1
            self.reload_data()
            self._train_step = 0
        else:
            #print("train")
            self.train_step += 1
            self._train_step += 1
            self.train()
        #print("done")
        return "done"

    def on_worker_start(self):
        pass

    @abstractmethod
    def on_init(self):
        pass

    @abstractmethod
    def load_model(self, dir, name):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def reload_data(self):
        pass


class KFoldClient(Client):
    def __init__(
            self,
            server_ip,
            server_port,
            n_cpus,
            shared_value,
            critical_mem,
            init_buffer=4096,
            data_dir="./tmp",
            dataset_dir="/edges_0/s3dis/edges",
            buffer_size=4096):
        self.dataset_dir = dataset_dir
        self.k_fold_dir = self.dataset_dir + "/../folds"
        print("k fold directory: {0}".format(self.k_fold_dir))
        self.buffer_size = buffer_size
        model_dir = "./tmp"
        model_name = "tmp_net"
        self.net_file = model_dir + "/" + model_name + ".npz"
        self.net_msg_size = None
        self.test_loop = False
        self.train_step = 0
        self._train_step = 0
        self.test_step = 0
        self.k_fold = None
        super().__init__(
            n_cpus=n_cpus,
            shared_value=shared_value,
            critical_mem=critical_mem,
            server_ip=server_ip,
            server_port=server_port,
            init_buffer=init_buffer,
            data_dir=data_dir)

    @abstractmethod
    def get_worker(self, conn, id, ready_val, lock, train_idxs, test_idxs):
        pass

    def train_test_idxs(self, id):
        _, tr_start, tr_stop = divide_work(worker_id=id, n_workers=self.n_cpus, workload=self.train_n)
        train_idxs_w = np.array(self.train_idxs[tr_start:tr_stop], copy=True)
        _, te_start, te_stop = divide_work(worker_id=id, n_workers=self.n_cpus, workload=self.test_n)
        test_idxs_w = np.array(self.test_idxs[te_start:te_stop], copy=True)
        #print("worker id: {0}, n_cpus: {1}".format(id, self.n_cpus))
        #print("worker id: {0}, train start: {1}, train stop: {2}, test start: {3}, test stop: {4}".format(id, tr_start, tr_stop, te_start, te_stop))
        if train_idxs_w.shape[0] == 0:
            raise Exception("Zero train idxs in worker {0}, train_idxs shape: {1}, train_n: {2}".format(id, self.train_idxs.shape, self.train_n))
        if test_idxs_w.shape[0] == 0:
            raise Exception("Zero test idxs in worker {0}, test_idxs shape: {1}, test_n: {2}".format(id, self.test_idxs.shape, self.test_n))
        np.savez("./tmp/train_test_idxs_{0}.npz".format(id), train_idxs=train_idxs_w, test_idxs=test_idxs_w)
        return train_idxs_w, test_idxs_w 

    def create_worker(self, conn, id, ready_val, lock):
        train_idxs_w, test_idxs_w = self.train_test_idxs(id=id)
        return self.get_worker(conn=conn, id=id, ready_val=ready_val, lock=lock, train_idxs=train_idxs_w, test_idxs=test_idxs_w)

    def on_worker_progress(self, msg, id):
        return

    def on_loop(self):
        #print("send data")
        if self.test_loop:
            for i in range(self.n_cpus):
                socket_send(file="./tmp/test_stats_" + str(i) + ".npz", sock=self.sock, buffer_size=self.buffer_size)
                msg = self.sock.recv(128)
                #print("test, received:", msg.decode())
        else:
            for i in range(self.n_cpus):
                socket_send(file="./tmp/grads_" + str(i) + ".npz", sock=self.sock, buffer_size=self.buffer_size)
                msg = self.sock.recv(128)
                #print("train, received:", msg.decode())
            self.train_step += 1
            self._train_step += 1
        #print(msg)
        #print("done - wait for network update")
        ret = socket_recv(file=self.net_file, sock=self.sock, buffer_size=self.buffer_size, msg_size=self.net_msg_size)
        if self.net_msg_size is None:
            self.net_msg_size = ret
        #print("done")

    def on_loop_end(self):
        test = self._train_step == self.test_interval.value
        #print("client master, test_interval:", self.test_interval.value, test)
        if test:
            #print("test", self.test_loop, self.train_step, self.test_interval)
            self.test_loop = True
            self.test_step += 1
            self.load()
            # save the train/test idxs to file
            for id in range(self.n_cpus):
                self.train_test_idxs(id=id)
            self.msg_to_workers("test")
            self._train_step = 0
        else:
            #print("train", self.test_loop, self.train_step, self.test_interval)
            self.test_loop = False
            self.msg_to_workers("train")
    
    def unpack_msg(self, msg, i):
        ti = int(msg[i+1])
        self.test_interval = Value("i", ti)
        self.k_fold = int(msg[i+2])
        self.seed = int(msg[i+3])
        self.dataset = msg[i+4]
        self.p_data = float(msg[i+5])
        self.n_epochs = int(msg[i+6])

        print("test_interval: {0}".format(self.test_interval))
        print("k_fold: {0}".format(self.k_fold))
        print("seed: {0}".format(self.seed))
        print("dataset: {0}".format(self.dataset))
        print("p_data: {0}".format(self.p_data))
        print("n_epochs: {0}".format(self.n_epochs))

    @abstractmethod
    def get_data(self):
        pass

    def load(self, verbose=False):
        self.data_files, self.train_idxs, self.test_idxs = self.get_data()
        self.train_n = len(self.train_idxs)
        if self.k_fold is not None:
            self.test_interval.value = self.train_n * self.n_epochs
            #print("client master, update test_interval:", self.test_interval.value)
        self.test_n = len(self.test_idxs)
        #print("{0} examples for training, {1} examples for testing".format(self.train_n, self.test_n))
        
        self.train_idxs = split_examples(train_idxs=self.train_idxs, train_n=self.train_n,
            client_id=self.node_id, n_clients=self.n_nodes)
        self.train_n = self.train_idxs.shape[0]
        if self.train_idxs.shape[0] == 0:
            raise Exception("Zero train idxs, train_n: {0}, client_id: {1}, n_clients: {2}".format(self.train_n, self.node_id, self.n_nodes))
        self.test_idxs = split_examples(train_idxs=self.test_idxs, train_n=self.test_n,
            client_id=self.node_id, n_clients=self.n_nodes)
        self.test_n = self.test_idxs.shape[0]
        if verbose:
            print("train_n: {0}, test_n: {1}".format(self.train_n, self.test_n))
        if self.test_idxs.shape[0] == 0:
            raise Exception("Zero train idxs, train_n: {0}, client_id: {1}, n_clients: {2}".format(self.test_n, self.node_id, self.n_nodes))
        #print("use {0} examples for training and {1} examples for testing".format(self.train_idxs.shape[0], self.test_idxs.shape[0]))

    def on_init(self):
        self.load(verbose=True)
        
        ret = socket_recv(file=self.net_file, sock=self.sock, buffer_size=self.buffer_size, msg_size=self.net_msg_size)
        if self.net_msg_size is None:
            self.net_msg_size = ret
        self.start_loop()