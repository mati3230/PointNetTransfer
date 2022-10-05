import tensorflow as tf
import argparse
import os
import h5py
import random
import math
import numpy as np
from optimization.kfold_tf_server import KFoldTFServer
from optimization.utils import get_type, save_config, load_args_file
from utils import load_block, compose_model_args, mkdir


def store_fold(folds_dir, i, files):
    # e.g. ./s3dis/graphs/Area1_conferenceRoom_1.h5 will be stored as new file
    hf = h5py.File("{0}/{1}.h5".format(folds_dir, i), "w")
    fs = [int(file.split(".")[0]) for file in files]
    hf.create_dataset("files", data=fs)
    hf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",type=str,default="192.168.0.164",help="IP of the server")
    parser.add_argument("--port",type=int,default=5000,help="Port of the server")
    parser.add_argument("--buffer_size",type=int,default=4096,help="Size of the transmission data")
    parser.add_argument("--n_clients",type=int,default=1,help="Number of clients")
    parser.add_argument("--args_file",type=str,default="args_file.json",help="file (relative path) to default configuration")
    parser.add_argument("--recv_timeout", type=int, default=8, help="Timeout to receive data stream.")
    parser.add_argument("--dataset",type=str,default="S3DIS",help="options: scannet, s3dis")
    parser.add_argument("--p_data",type=float,default=1,help="Percentage of the data that should be used")
    parser.add_argument("--gpu",type=bool,default=False,help="Should gpu be used")
    parser.add_argument("--create_folds",type=bool,default=False,help="Should the folds be created?")
    parser.add_argument("--k_fold",type=int,default=5,help="k fold cross validation")
    parser.add_argument("--n_epochs",type=int,default=14,help="Number of epochs for k fold cross validation")
    parser.add_argument("--experiment_name",type=str,default="1",help="Name of the experiment for output file of k fold cross validation")
    args = parser.parse_args()
    if args.create_folds:
        folds_dir = "./Blocks/" + args.dataset + "_Folds"
        mkdir(folds_dir)
        blocks = os.listdir("./Blocks/" + args.dataset)
        random.shuffle(blocks)
        n_files = len(blocks)
        files_per_fold = math.floor(n_files / args.k_fold)
        n_files = args.k_fold * files_per_fold
        blocks = blocks[:n_files]
        for i in range(args.k_fold):
            start = i * files_per_fold
            stop = start  + files_per_fold
            files_fold = blocks[start:stop]
            store_fold(folds_dir=folds_dir, i=i, files=files_fold)
        return
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    params, _ = load_args_file(args_file=args.args_file, types_file="types.json")

    seed = params["seed"]
    model_args = compose_model_args(dataset=args.dataset, dataset_dir="./Blocks/" + args.dataset, params=params)
    tf.random.set_seed(seed)
    model_type = get_type(params["model_path"], params["model_type"])
    print("instantiate neural net")
    model = model_type(**model_args)
    
    print("load example")
    files_dir = "./Blocks/" + args.dataset
    files = os.listdir(files_dir)

    block, b_labels = load_block(block_dir=files_dir, name=0)
    print("prepare example with {0} elements".format(b_labels.shape[0]))
    print("prediction")
    block = np.expand_dims(block, axis=0)
    model(obs=block, training=False)
    print("reset")
    model.reset()
    print("neural net ready")
    #print(model.get_vars()[0])
    #return

    model_dir = "./models/" + args.dataset + "/" + params["model_name"]

    global_norm_t = params["global_norm"]
    learning_rate = params["learning_rate"]

    folds_dir = "./Blocks/" + args.dataset + "_Folds"
    k_fold = len(os.listdir(folds_dir))
    print("k fold with {0} folds".format(k_fold))
    model.save(directory=model_dir, filename="init_net")

    def set_test_interval(dataset, n_epochs, k_fold, fold_nr):
        fold_dir = "./Blocks/" + dataset + "_Folds"

        n_folds = len(os.listdir(fold_dir))
        if fold_nr >= k_fold:
            fold_nr = 0
        fold_file = fold_dir + "/" + str(fold_nr) + ".h5"
        hf = h5py.File(fold_file, "r")
        examples_per_fold = len(list(hf["files"]))
        hf.close()
        test_interval = examples_per_fold * (n_folds - 1) * n_epochs

        print("set test interval to {0} with {1} examples per fold ({2} folds)".format(test_interval, examples_per_fold, n_folds))
        return test_interval, examples_per_fold
    
    tf_server = KFoldTFServer(
        args_file=args.args_file,
        model=model,
        global_norm_t=global_norm_t,
        learning_rate=learning_rate,
        test_interval=1,
        k_fold=k_fold,
        model_dir=model_dir,
        seed=seed,
        p_data=args.p_data,
        dataset=args.dataset,
        ip=args.ip,
        port=args.port,
        buffer_size=args.buffer_size,
        n_nodes=args.n_clients,
        recv_timeout=args.recv_timeout,
        experiment_name=args.experiment_name,
        n_epochs=args.n_epochs,
        set_test_interval=set_test_interval)

    print("tf server initialized")
    save_config(tf_server.log_dir, str(params))
    tf_server.start()

if __name__ == "__main__":
    main()