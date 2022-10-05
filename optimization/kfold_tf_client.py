import numpy as np
import tensorflow as tf
import os
import math
from multiprocessing import Value
from abc import abstractmethod

from .kfold_network_client import KFoldWorker, KFoldClient
from .utils import get_type, load_args_file
from .tf_utils import setup_gpu


class KFoldTFWorker(KFoldWorker):
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
            start_with_work=True,
            verbose=False):
        self.verbose = verbose
        self.k_fold_dir = dataset_dir + "/../folds"
        tf.random.set_seed(seed)
        super().__init__(
            conn=conn,
            id=id,
            ready_val=ready_val,
            lock=lock,
            test_interval=test_interval,
            k_fold=k_fold,
            seed=seed,
            dataset=dataset,
            dataset_dir=dataset_dir,
            args_file=args_file,
            p_data=p_data,
            train_idxs=train_idxs,
            test_idxs=test_idxs,
            start_with_work=start_with_work,
            )

    def on_init(self):
        setup_gpu()
        print(self.args_file)
        params, _ = load_args_file(args_file=self.args_file, types_file="types.json")

        self.batch_size = params["batch_size"]
        self.seed = params["seed"]
        self.check_numerics = params["check_numerics"]
        np.random.seed(self.seed)

        # calculate the nr of batches
        self.n_batches = math.floor(self.train_idxs.shape[0] / self.batch_size)
        self.n_t_batches = self.get_n_t_batches()
        self.batch_id = 0

        # init the neural net
        model_args = self.compose_model_args(params)
        model_type = get_type(params["model_path"], params["model_type"])
        self.model = model_type(**model_args)
        self.data_files, _, _ = self.load_dataset()
        batch = self.load_batch(i=0, train_idxs=self.test_idxs, dir=self.k_fold_dir,
            files=self.data_files, batch_size=self.batch_size)
        #print(batch[0].nodes.shape)
        self.prediction(batch=batch)
        self.model.reset()
        print("neural net ready")

        # create arrays for the batches that are used in training and testing

        self.on_init_end()
        #self.test()

    @abstractmethod
    def compose_model_args(self):
        pass

    def get_n_t_batches(self):
        return math.floor(self.test_idxs.shape[0] / self.batch_size)

    def load_model(self, dir, name):
        # abstract method to load the variables
        self.model.load(directory=dir, filename=name)

    def train(self):
        # training phase
        #print("Compute gradients")
        with tf.GradientTape() as tape:
            try:
                batch = self.load_batch(i=self.batch_id, train_idxs=self.train_idxs, dir=self.dataset_dir,
                    files=self.data_files, batch_size=self.batch_size)
            except:
                self.batch_id = 0
                batch = self.load_batch(i=self.batch_id, train_idxs=self.train_idxs, dir=self.dataset_dir,
                    files=self.data_files, batch_size=self.batch_size)

            losses = self.compute_losses(batch=batch)
            if self.verbose:
                print(losses)
            loss = losses["loss"]

            vars_ = tape.watched_variables()
            grads = tape.gradient(loss, vars_)

            # save gradients
            save_dict = {}
            for i in range(len(grads)):
                grad = grads[i]
                grad_np = grad.numpy()
                name = vars_[i].name
                save_dict[name] = grad_np
                # uncomment to log the mean gradients of the variables
                #save_dict[name + "_grads_loss"] = np.mean(grad_np)
            for k, v in losses.items():
                save_dict[k] = v.numpy()
            np.savez("./tmp/grads_" + str(self.id) + ".npz", **save_dict)

        self.batch_id += 1
        if self.batch_id >= self.n_batches:
            self.batch_id = 0

    def test_prediction(self, i):
        batch = self.load_batch(i=i, train_idxs=self.test_idxs, dir=self.dataset_dir, 
            files=self.data_files, batch_size=self.batch_size)
        action = self.prediction(batch=batch)
        #action = action.numpy()
        y = batch[1]
        return y, action

    def test(self):
        accs = []
        precs = []
        recs = []
        f1s = []
        #print("Compute test")
        for i in range(self.n_t_batches):
            #print("File:", self.data_files[self.test_idxs[i]])
            y, action = self.test_prediction(i=i)
            #print(np.unique(action, return_counts=True))
            TP = np.sum((y == 1) & (action == 1))
            TN = np.sum((y == 0) & (action == 0))
            FN = np.sum((y == 1) & (action == 0))
            FP = np.sum((y == 0) & (action == 1))
            acc = (TP + TN) / y.shape[0]
            #print("TP: {0}, TN: {1}, FN: {2}, FP: {3}".format(TP, TN, FN, FP))
            if TP + FP == 0:
                prec = TP / (TP + FP + 1e-12)
            else:
                prec = TP / (TP + FP)
            if TP + FN == 0:
                rec = TP / (TP + FN + 1e-12)
            else:
                rec = TP / (TP + FN)
            if prec + rec == 0:
                f1 = 2*(prec * rec)/(prec + rec + 1e-12)
            else:
                f1 = 2*(prec * rec)/(prec + rec)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
        save_dict = {}
        save_dict["Mean_Acc"] = np.mean(accs)
        save_dict["Mean_Prec"] = np.mean(precs)
        save_dict["Mean_Rec"] = np.mean(recs)
        save_dict["Mean_F1"] = np.mean(f1s)
        save_dict["Std_Acc"] = np.std(accs)
        save_dict["Std_Prec"] = np.std(precs)
        save_dict["Std_Rec"] = np.std(recs)
        save_dict["Std_F1"] = np.std(f1s)
        if self.verbose:
            print(save_dict)
        np.savez("./tmp/test_stats_" + str(self.id) + ".npz", **save_dict)

    def reload_data(self):
        self.data_files, _, _ = self.load_dataset()
        data = np.load("./tmp/train_test_idxs_{0}.npz".format(self.id))
        self.train_idxs = np.array(data["train_idxs"], copy=True)
        self.test_idxs = np.array(data["test_idxs"], copy=True)

    @abstractmethod
    def load_folds(self, k_fold_dir, train_folds, test_fold):
        pass

    def load_dataset(self):
        train_folds = list(range(self.k_fold))
        test_fold = self.test_step
        if test_fold >= self.k_fold:
            test_fold = 0
        del train_folds[test_fold]

        train_data, test_data = self.load_folds(k_fold_dir=self.k_fold_dir, train_folds=train_folds,
            test_fold=test_fold)
        train_idxs = np.arange(len(train_data), dtype=np.uint32)
        test_idxs = np.arange(len(test_data), dtype=np.uint32) + train_idxs.shape[0]
        train_data.extend(test_data)
        return train_data, train_idxs, test_idxs

    @abstractmethod
    def prediction(self, batch):
        pass

    @abstractmethod
    def compute_losses(self, batch):
        pass

    @abstractmethod
    def on_init_end(self):
        pass

    @abstractmethod
    def load_example(self, dir, files, idx, data_scale):
        pass

    @abstractmethod
    def load_batch(self, i, train_idxs, dir, files, batch_size, data_scale):
        pass


class KFoldTFClient(KFoldClient):
    def __init__(
            self,
            server_ip,
            server_port,
            n_cpus,
            shared_value,
            critical_mem,
            init_buffer=4096,
            data_dir="./tmp",
            dataset_dir="./s3dis/graphs",
            buffer_size=4096):
        super().__init__(
            server_ip=server_ip,
            server_port=server_port,
            n_cpus=n_cpus,
            shared_value=shared_value,
            critical_mem=critical_mem,
            init_buffer=init_buffer,
            data_dir=data_dir,
            dataset_dir=dataset_dir,
            buffer_size=buffer_size)

    def unpack_msg(self, msg, i):
        super().unpack_msg(msg=msg, i=i) 
        self.args_file = msg[i+7]
        print("args_file: {0}".format(self.args_file))

    @abstractmethod
    def load_folds(self, k_fold_dir, train_folds, test_fold):
        pass

    def load_dataset(self):
        train_folds = list(range(self.k_fold))
        test_fold = self.test_step
        if test_fold >= self.k_fold or test_fold < 0:
            test_fold = 0
        del train_folds[test_fold]
        print("client master, test fold:", test_fold)
        train_data, test_data = self.load_folds(k_fold_dir=self.k_fold_dir, train_folds=train_folds,
            test_fold=test_fold)
        train_idxs = np.arange(len(train_data), dtype=np.uint32)
        test_idxs = np.arange(len(test_data), dtype=np.uint32) + train_idxs.shape[0]
        train_data.extend(test_data)
        return train_data, train_idxs, test_idxs

    def get_data(self):
        return self.load_dataset()

    @abstractmethod
    def get_worker(self, conn, id, ready_val, lock, train_idxs, test_idxs):
        pass