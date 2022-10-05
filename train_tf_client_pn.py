import numpy as np
import tensorflow as tf
import argparse
from multiprocessing import Value
import os
import math
import h5py
from optimization.kfold_tf_client import KFoldTFWorker, KFoldTFClient
import utils


class KFoldTFWorkerPN(KFoldTFWorker):
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
            verbose=verbose
            )
        self.batch_size = 1

    def prediction(self, batch):
        blocks = batch[0]
        pred = self.model(obs=blocks, training=False)
        pred = tf.nn.softmax(pred)
        pred = pred.numpy()
        pred = np.argmax(pred, axis=-1)
        return pred

    def compute_losses(self, batch):
        blocks, labels = batch
        #print("train")
        pred = self.model(obs=blocks, training=True)
        #labels_ = tf.cast(labels, tf.float32)
        seg_loss, _, _ = get_loss(seg_pred=pred, seg=labels, check_numerics=self.check_numerics)
        return {
            "loss": seg_loss
            }

    def test(self):
        accs = []
        precs = []
        recs = []
        f1s = []
        ious = []
        #print("Compute test")
        for i in range(self.n_t_batches):
            #print("File:", self.data_files[self.test_idxs[i]])
            t_labels, action = self.test_prediction(i=i)
            #print(np.unique(action, return_counts=True))
            n_acc = t_labels.shape[0] * t_labels.shape[1] * n_t_batches
            acc = 0
            for c in range(n_classes): # calculate performance metrics for each class
                TP = np.sum((t_labels == c) & (pred == c)) # true positives
                FP = np.sum((t_labels != c) & (pred == c)) # false positives
                FN = np.sum((t_labels == c) & (pred != c)) # false negatives
                
                n = TP
                d = float(TP + FP + FN + 1e-12)

                iou = np.divide(n, d) # intersection over union (iou)

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

                ious.append(iou)
                # average the accuracy
                accs.append(TP / n_acc)

                #print("TP: {0}, TN: {1}, FN: {2}, FP: {3}".format(TP, TN, FN, FP))
                accs.append(acc)
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)
        save_dict = {}
        save_dict["Mean_IoU"] = np.mean(ious)
        save_dict["OveralAcc"] = np.sum(accs)
        save_dict["Mean_Prec"] = np.mean(precs)
        save_dict["Mean_Rec"] = np.mean(recs)
        save_dict["Mean_F1"] = np.mean(f1s)
        save_dict["Std_IoU"] = np.std(ious)
        save_dict["Std_Prec"] = np.std(precs)
        save_dict["Std_Rec"] = np.std(recs)
        save_dict["Std_F1"] = np.std(f1s)
        if self.verbose:
            print(save_dict)
        np.savez("./tmp/test_stats_" + str(self.id) + ".npz", **save_dict)

    def on_init_end(self):
        #self.bce = BinaryCrossentropy(from_logits=False)
        return

    def get_n_t_batches(self):
        return self.test_idxs.shape[0]

    def load_example(self, dir, files, idx):
        return utils.load_block(block_dir=dir, name=idx)

    def load_batch(self, i, train_idxs, dir, files, batch_size):
        # note that dir=self.dataset_dir
        return utils.load_batch2(i=i, train_idxs=train_idxs, block_dir=dir, blocks=None,
            labels=None, batch_size=batch_size)

    def load_folds(self, k_fold_dir, train_folds, test_fold):
        return utils.load_folds(dataset_dir=self.dataset_dir, k_fold_dir=k_fold_dir, train_folds=train_folds, test_fold=test_fold)

    def compose_model_args(self, params):
        return utils.compose_model_args(dataset=self.dataset, dataset_dir=self.dataset_dir, params=params)


class KFoldTFClientPN(KFoldTFClient):
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
            buffer_size=4096,
            verbose=False):
        self.verbose = verbose
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

    def load_folds(self, k_fold_dir, train_folds, test_fold):
        return utils.load_folds(dataset_dir=self.dataset_dir, k_fold_dir=k_fold_dir, train_folds=train_folds, test_fold=test_fold)

    def get_worker(self, conn, id, ready_val, lock, train_idxs, test_idxs):
        return KFoldTFWorkerPN(
            conn=conn,
            id=id,
            ready_val=ready_val,
            lock=lock,
            test_interval=self.test_interval,
            k_fold=self.k_fold,
            seed=self.seed,
            dataset=self.dataset,
            dataset_dir=self.dataset_dir,
            args_file=self.args_file,
            p_data=self.p_data,
            train_idxs=train_idxs,
            test_idxs=test_idxs,
            verbose=self.verbose)


def main():
    print("you have to set the ip, port, n_clients, client_id")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",type=str,default="192.168.0.164",help="IP of the server")
    parser.add_argument("--port",type=int,default=5000,help="Port of the server")
    parser.add_argument("--buffer_size",type=int,default=4096,help="Size of the transmission data")
    parser.add_argument("--n_cpus",type=int,default=1,help="Nr of cpus that will be used for the training")
    parser.add_argument("--dataset",type=str,default="S3DIS",help="path to the dataset")
    parser.add_argument("--critical_mem",type=int,default=85,help="Threshold - training will stop if too much memory is used") 
    parser.add_argument("--gpu",type=bool,default=False,help="Should gpu be used")
    parser.add_argument("--verbose",type=bool,default=False,help="Print training progress")
    parser.add_argument("--k_fold",type=bool,default=False,help="Use k fold cross validation")
    args = parser.parse_args()
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    KFoldTFClientPN(
        server_ip=args.ip,
        server_port=args.port,
        n_cpus=args.n_cpus,
        dataset_dir="./Blocks/" + args.dataset,
        buffer_size=args.buffer_size,
        shared_value=Value("i", True),
        critical_mem=args.critical_mem,
        verbose=args.verbose)


if __name__ == "__main__":
    main()