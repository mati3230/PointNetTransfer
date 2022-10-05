import tensorflow as tf
import numpy as np
import math
import datetime
import argparse
import os
#from scipy.spatial.transform import Rotation as R
from sesa_pointnet import SeSaPointNet
# from utils import render_point_cloud
from tqdm import tqdm
from utils import load_block, load_batch2, get_loss


def update_stats(stat_vec, t_uni):
    unis, counts = np.unique(t_uni, return_counts=True)
    stat_vec[unis] += counts
    return stat_vec


def freeze(vars_, grads, var_idxs, ft_net_vars):
    # filter some weights with the names 'ft_net_vars' from the weight update
    if var_idxs is None:
        var_idxs = []
        for z in range(len(vars_)):
            var_name = vars_[z].name
            if var_name in ft_net_vars:
                var_idxs.append(z)
        # print(var_idxs)
    tmp_vars = []
    tmp_grads = []
    for z in range(len(vars_)):
        if z in var_idxs:
            continue
        tmp_vars.append(vars_[z])
        tmp_grads.append(grads[z])
    vars_ = tmp_vars
    grads = tmp_grads
    return vars_, grads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Set a random seed.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch_size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Training learning rate.")
    parser.add_argument("--global_norm_t", type=float, default=10, help="Training global norm threshold.")
    parser.add_argument("--train_p", type=float, default=0.8, help="Percentage of training data.")
    parser.add_argument("--test_interval", type=int, default=1, help="Interval to test the model.")
    parser.add_argument("--max_epoch", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--n_classes", type=int, default=14, help="Number of classes.")
    parser.add_argument("--initializer", type=str, default="glorot_uniform", help="Initializer of the weights.")
    parser.add_argument("--dataset", type=str, default="S3DIS", help="Name of the dataset.")
    parser.add_argument("--check_numerics", type=bool, default=False, help="Should NaN or Inf values be checked.")
    parser.add_argument("--load", type=bool, default=False, help="Load feature detector.")
    parser.add_argument("--model_file", type=str, help="Name of the feature detector that should be loaded.")
    parser.add_argument("--model_dir", type=str, help="Directory of the feature detector that should be loaded.")
    parser.add_argument("--freeze", type=bool, default=False, help="Freeze weights of the feature detector.")
    parser.add_argument("--transfer_train_p", type=float, default=1.0, help="Use less train examples.")
    parser.add_argument("--clean_ds", type=bool, default=False, help="Faulty blocks will be deleted.")
    parser.add_argument("--k_fold", type=int, default=5, help="Specify k for the k fold cross validation.")
    parser.add_argument("--pre_batched", type=bool, default=False, help="Are the examples stored as batches?")
    args = parser.parse_args()

    p_dim = 6
    seed = args.seed
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    global_norm_t = args.global_norm_t
    test_interval = args.test_interval
    block_dir = "Blocks/" + args.dataset
    n_classes = args.n_classes
    if args.dataset == "S3DIS":
        n_classes = 14
    elif args.dataset == "PCG":
        n_classes = 12
    elif args.dataset == "Scannet":
        n_classes = 524

    np.random.seed(seed)

    # train test split
    tmp_block_dirs = os.listdir(block_dir)
    block_dirs = tmp_block_dirs.copy()
    if args.clean_ds:
        tmp_block, tmp_b_labels = load_block(block_dir=block_dir, name=1)
        points_per_block = tmp_block.shape[0]
        print("We have {0} points per block".format(points_per_block))
        deleted = 0
        for i in tqdm(range(len(tmp_block_dirs)), desc="Filter Blocks"):
            bfile = tmp_block_dirs[i]
            nr = int(bfile.split(".")[0])
            try:
                tmp_block, tmp_b_labels = load_block(block_dir=block_dir, name=nr)
            except Exception as e:
                continue
            if tmp_block.shape[0] != points_per_block or tmp_b_labels.shape[0] != points_per_block:
                #print("Remove block '{0}' (nr of points in block: {1}, nr of points in block: {2})".format(
                #    bfile, tmp_block.shape[0], tmp_b_labels.shape[0]))
                os.remove(block_dir + "/" + bfile)
                block_dirs.remove(bfile)
                deleted += 1
        print("{0} files deleted".format(deleted))
        for i in tqdm(range(len(block_dirs)), desc="Rename Blocks"):
            bfile = block_dirs[i]
            os.rename(block_dir + "/" + bfile, block_dir + "/n" + str(i) + ".npz")
        block_dirs = os.listdir(block_dir)
        for i in tqdm(range(len(block_dirs)), desc="Rename Blocks"):
            bfile = block_dirs[i]
            os.rename(block_dir + "/" + bfile, block_dir + "/" + str(i) + ".npz")
        block_dirs = os.listdir(block_dir)
    #return
    if args.transfer_train_p < 1.0:
        size = math.floor(args.transfer_train_p * len(block_dirs))
        all_idxs = np.arange(size).astype(np.int32)
    else:
        all_idxs = np.arange(len(block_dirs)).astype(np.int32)
    if all_idxs.shape[0] < batch_size:
        raise Exception("Batch size ({0}) is greater than the number of training examples ({1}).".format(batch_size, all_idxs.shape[0]))
    np.random.shuffle(all_idxs)
    if args.k_fold >= 2:
        np.random.shuffle(all_idxs)
        train_p = args.train_p
        examples_n = math.floor(train_p * len(all_idxs))
        examples_per_fold = math.floor(examples_n / args.k_fold)
        examples_n = examples_per_fold * args.k_fold
        all_idxs = all_idxs[:examples_n]
        train_n = (args.k_fold - 1) * examples_per_fold
        test_n = examples_per_fold
        print("Use {0} blocks for training and {1} blocks for testing".format(train_n, test_n))
        train_idxs = all_idxs[:train_n]
        test_idxs = all_idxs[train_n:train_n+test_n]
    else:
        train_p = args.train_p
        train_n = math.floor(train_p * len(all_idxs))
        test_n = len(all_idxs) - train_n
        print("Use {0} blocks for training and {1} blocks for testing".format(train_n, test_n))
        train_idxs = np.random.choice(all_idxs, size=train_n, replace=False)
        test_idxs = np.delete(all_idxs, train_idxs)
        np.random.shuffle(train_idxs)

    if args.pre_batched:
        n_batches = train_n
        n_t_batches = test_n
    else:
        # determine the number of batches
        n_batches = math.floor(train_n / batch_size)
        n_t_batches = math.floor(test_n / batch_size)

    n_epoch = 0
    max_epoch = args.max_epoch

    # prepare containers to store the batches
    b, l = load_block(block_dir, 0, spatial_only=False)
    blocks = np.zeros((batch_size, ) + b.shape, np.float32)
    labels = np.zeros((batch_size, ) + (l.shape[0], ), np.uint8)
    t_blocks = np.zeros((batch_size, ) + b.shape, np.float32)
    t_labels = np.zeros((batch_size, ) + (l.shape[0], ), np.uint8)

    if args.k_fold <= 1:
        if args.load: # load a pretrained network
            net = SeSaPointNet(
                name="SeSaPN",
                n_classes=n_classes,
                seed=seed,
                trainable=True,
                check_numerics=args.check_numerics,
                initializer=args.initializer,
                trainable_net=False,
                n_points=b.shape[0],
                p_dim=p_dim)
            tmp_b = np.array(b, copy=True)
            tmp_b = np.expand_dims(b, axis=0)
            net(tmp_b, training=False)
            net.reset()
            net.load(directory=args.model_dir, filename=args.model_file, net_only=True)
            print("model loaded")
            # get vars from PointNet which is a part of SeSaPointNet
            ft_net_vars_tmp = net.net.get_vars()
            ft_net_vars = []
            var_idxs = None
            for z in range(len(ft_net_vars_tmp)):
                ft_net_vars.append(ft_net_vars_tmp[z].name)
            #print(ft_net_vars)
        else: # train a network from scratch
            net = SeSaPointNet(
                name="SeSaPN",
                n_classes=n_classes,
                seed=seed,
                trainable=True,
                check_numerics=args.check_numerics,
                initializer=args.initializer,
                trainable_net=True,
                n_points=b.shape[0],
                p_dim=p_dim)
    # prepare training and logging
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "./logs/" + current_time
    train_summary_writer = tf.summary.create_file_writer(log_dir)
    train_step = 0
    test_step = 0

    load_batch_func = load_batch
    if args.pre_batched:
        load_batch_func = load_batch2

    if args.k_fold >= 2:
        def get_train_test_idxs(all_idxs, examples_per_fold, k_fold, fold_nr):
            if fold_nr >= k_fold:
                fold_nr = 0
            elif fold_nr < 0:
                fold_nr = 0
            start = fold_nr * examples_per_fold
            stop = start + examples_per_fold
            test_idxs = np.arange(start, stop, dtype=np.uint32)
            idxs = np.arange(all_idxs.shape[0], dtype=np.uint32)
            train_idxs = np.delete(idxs, test_idxs)
            return train_idxs, test_idxs

        fold_stats = {}
        fold_stats["overall_acc"] = []
        fold_stats["mIoU"] = []
        # n batches should match the number of training examples
        for fold_nr in range(args.k_fold):
            train_idxs, test_idxs = get_train_test_idxs(
                all_idxs=all_idxs, examples_per_fold=examples_per_fold, k_fold=args.k_fold, fold_nr=fold_nr)

            if args.load: # load a pretrained network
                net = SeSaPointNet(
                    name="SeSaPN",
                    n_classes=n_classes,
                    seed=seed,
                    trainable=True,
                    check_numerics=args.check_numerics,
                    initializer=args.initializer,
                    trainable_net=False,
                    n_points=b.shape[0],
                    p_dim=p_dim)
                tmp_b = np.array(b, copy=True)
                tmp_b = np.expand_dims(b, axis=0)
                net(tmp_b, training=False)
                net.reset()
                net.load(directory=args.model_dir, filename=args.model_file, net_only=True)
                print("model loaded")
                # get vars from PointNet which is a part of SeSaPointNet
                ft_net_vars_tmp = net.net.get_vars()
                ft_net_vars = []
                var_idxs = None
                for z in range(len(ft_net_vars_tmp)):
                    ft_net_vars.append(ft_net_vars_tmp[z].name)
                #print(ft_net_vars)
            else: # train a network from scratch
                net = SeSaPointNet(
                    name="SeSaPN",
                    n_classes=n_classes,
                    seed=seed,
                    trainable=True,
                    check_numerics=args.check_numerics,
                    initializer=args.initializer,
                    trainable_net=True,
                    n_points=b.shape[0],
                    p_dim=p_dim)

            n_epoch = 0
            print("Fold {0}/{1}".format(fold_nr+1, args.k_fold))
            for n_epoch in tqdm(range(max_epoch), desc="Epoch"):
                #print("epoch {0}/{1}".format(n_epoch+1, max_epoch))
                for i in range(n_batches): # execute n_batches train steps
                    with tf.GradientTape() as tape:
                        blocks, labels = load_batch_func(i, train_idxs, block_dir, blocks, labels, batch_size, apply_random_rotation=False, spatial_only=False)
                        pred = net(blocks, training=True)
                        loss, seg_loss, _ = get_loss(seg_pred=pred, seg=labels, check_numerics=args.check_numerics)
                        # loss: The overall loss that contains the seg_loss and the mat_diff_loss
                        # seg_loss: Cross entropy loss for the semantic segmentation
                        # mat_diff_loss: Loss of the T-Net which part of the PointNet feature exrtactor
                        # get the variables and gradients
                        vars_ = tape.watched_variables()
                        grads = tape.gradient(loss, vars_)

                        if args.freeze: # skip the weigths of the PointNet feature extractor  
                            vars_, grads = freeze(vars_=vars_, grads=grads, var_idxs=var_idxs, ft_net_vars=ft_net_vars)

                        # Threshold operation on the gradients 
                        global_norm = tf.linalg.global_norm(grads)
                        if global_norm_t > 0:
                            grads, _ = tf.clip_by_global_norm(
                                grads,
                                global_norm_t,
                                use_norm=global_norm)
                        # Weight update
                        optimizer.apply_gradients(zip(grads, vars_))
                    # log tensorboard
                    with train_summary_writer.as_default():
                        tf.summary.scalar("train/loss", loss, step=train_step)
                        tf.summary.scalar("train/seg_loss", seg_loss, step=train_step)
                        #tf.summary.scalar("train/mat_diff_loss", mat_diff_loss, step=train_step)
                        tf.summary.scalar("train/global_norm", global_norm, step=train_step)
                    train_summary_writer.flush()
                    train_step += 1
            
            # test step
            accs = []
            # number of accuracy calculations in the test phase
            n_acc = t_labels.shape[0] * t_labels.shape[1] * n_t_batches
            # intersection over unions (ious)
            ious = []

            for i in range(n_t_batches): # execute n_t_batches test steps
                t_blocks, t_labels = load_batch_func(i, test_idxs, block_dir, t_blocks, t_labels, batch_size, spatial_only=False)
                pred = net(t_blocks, training=False)
                pred = tf.nn.softmax(pred)
                pred = pred.numpy()
                pred = np.argmax(pred, axis=-1)
                acc = 0
                for c in range(n_classes): # calculate performance metrics for each class
                    TP = np.sum((t_labels == c) & (pred == c)) # true positives
                    FP = np.sum((t_labels != c) & (pred == c)) # false positives
                    FN = np.sum((t_labels == c) & (pred != c)) # false negatives

                    n = TP
                    d = float(TP + FP + FN + 1e-12)

                    iou = np.divide(n, d) # intersection over union (iou)
                    ious.append(iou)
                    # average the accuracy
                    accs.append(TP / n_acc)
            # log tensorboard
            with train_summary_writer.as_default():
                # log the average statistics
                overall_acc = np.sum(accs)
                mIoU = np.mean(ious)
                tf.summary.scalar("test/overall_acc", overall_acc, step=test_step)
                tf.summary.scalar("test/mIoU", mIoU, step=test_step)
                fold_stats["overall_acc"].append(overall_acc)
                fold_stats["mIoU"].append(mIoU)
            train_summary_writer.flush()
            test_step += 1
            #print("test", test_step)
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            net.save(directory="./models/" + args.dataset, filename="pointnet_" + current_time, net_only=False)
            del net
        # compute the average stats
        avg_stats = {}
        for k, v in fold_stats.items():
            avg_stats[k] = np.mean(v)
        # save fold stats
        results = "Avg Stats\n"
        for k, v in avg_stats.items():
            results += str(k) + ": " + str(v) + "\n"
        results += "Raw Stats\n"
        for k, v in fold_stats.items():
            results += str(k) + ": "
            for i in range(len(v)):
                results += str(v[i]) + ", "
            results += "\n"
        with open("fold_stats.txt", "w") as f:
            f.write(results)
    else:
        # training and testing
        while n_epoch < max_epoch:
            if n_epoch % test_interval == 0: # test step
                accs = []
                # number of accuracy calculations in the test phase
                n_acc = t_labels.shape[0] * t_labels.shape[1] * n_t_batches
                # intersection over unions (ious)
                ious = []

                for i in range(n_t_batches): # execute n_t_batches test steps
                    t_blocks, t_labels = load_batch_func(i, test_idxs, block_dir, t_blocks, t_labels, batch_size, spatial_only=False)
                    pred = net(t_blocks, training=False)
                    pred = tf.nn.softmax(pred)
                    pred = pred.numpy()
                    pred = np.argmax(pred, axis=-1)
                    acc = 0
                    for c in range(n_classes): # calculate performance metrics for each class
                        TP = np.sum((t_labels == c) & (pred == c)) # true positives
                        FP = np.sum((t_labels != c) & (pred == c)) # false positives
                        FN = np.sum((t_labels == c) & (pred != c)) # false negatives

                        n = TP
                        d = float(TP + FP + FN + 1e-12)

                        iou = np.divide(n, d) # intersection over union (iou)
                        ious.append(iou)
                        # average the accuracy
                        accs.append(TP / n_acc)
                # log tensorboard
                with train_summary_writer.as_default():
                    # log the average statistics
                    tf.summary.scalar("test/overall_acc", np.sum(accs), step=test_step)
                    tf.summary.scalar("test/mIoU", np.mean(ious), step=test_step)
                train_summary_writer.flush()
                test_step += 1
                
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                net.save(directory="./models/" + args.dataset, filename="pointnet_" + current_time, net_only=False)
            for i in range(n_batches): # execute n_batches train steps
                with tf.GradientTape() as tape:
                    blocks, labels = load_batch_func(i, train_idxs, block_dir, blocks, labels, batch_size, apply_random_rotation=False, spatial_only=False)
                    pred = net(blocks, training=True)
                    loss, seg_loss, _ = get_loss(seg_pred=pred, seg=labels, check_numerics=args.check_numerics)
                    # loss: The overall loss that contains the seg_loss and the mat_diff_loss
                    # seg_loss: Cross entropy loss for the semantic segmentation
                    # mat_diff_loss: Loss of the T-Net which part of the PointNet feature exrtactor

                    # get the variables and gradients
                    vars_ = tape.watched_variables()
                    grads = tape.gradient(loss, vars_)

                    if args.freeze: # skip the weigths of the PointNet feature extractor  
                        vars_, grads = freeze(vars_=vars_, grads=grads, var_idxs=var_idxs, ft_net_vars=ft_net_vars)

                    # Threshold operation on the gradients 
                    global_norm = tf.linalg.global_norm(grads)
                    if global_norm_t > 0:
                        grads, _ = tf.clip_by_global_norm(
                            grads,
                            global_norm_t,
                            use_norm=global_norm)
                    # Weight update
                    optimizer.apply_gradients(zip(grads, vars_))
                # log tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar("train/loss", loss, step=train_step)
                    tf.summary.scalar("train/seg_loss", seg_loss, step=train_step)
                    #tf.summary.scalar("train/mat_diff_loss", mat_diff_loss, step=train_step)
                    tf.summary.scalar("train/global_norm", global_norm, step=train_step)
                train_summary_writer.flush()
                train_step += 1
            n_epoch += 1



if __name__ == "__main__":
    main()
