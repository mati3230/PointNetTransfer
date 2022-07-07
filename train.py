import tensorflow as tf
import numpy as np
import math
import datetime
import argparse
import os
from scipy.spatial.transform import Rotation as R
from sesa_pointnet import SeSaPointNet
# from utils import render_point_cloud


def get_loss(seg_pred, seg, t, reg_f=1e-3, check_numerics=True):
    # loss calculation
    seg = seg.astype(np.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=seg, logits=seg_pred)
    if check_numerics:
        ce = tf.debugging.check_numerics(ce, "ce")

    per_instance_seg_loss = tf.reduce_mean(ce, axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)

    K = tf.shape(t)[1]
    mul = tf.matmul(t, tf.transpose(t, perm=[0,2,1]))
    mat_diff = mul - tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    if check_numerics:
        mat_diff_loss = tf.debugging.check_numerics(mat_diff_loss, "mat_diff_loss")

    total = seg_loss + mat_diff_loss * reg_f
    return total, seg_loss, mat_diff_loss


def load_block(block_dir, name, spatial_only=True):
    # load a block of the point cloud
    filename = block_dir + "/" + str(name) + ".npz"
    data = np.load(filename)
    block = data["block"]
    b_labels = data["labels"]

    # translate into the origin
    mean_block = np.mean(block[:, :3], axis=0)
    block[:, :3] -= mean_block

    # uniformly scale to [-1, 1]
    max_block = np.max(np.abs(block[:, :3]))
    block[:, :3] /= max_block
    
    # scale point colors to [-0.5, 0.5]
    block[:, 3:] -= 0.5
    # scale point colors to [-1, 1]
    block[:, 3:] *= 2

    # render_point_cloud(block)
    if spatial_only:
        block = block[:, :3]

    return block, b_labels


def load_batch(i, train_idxs, block_dir, blocks, labels, batch_size, apply_random_rotation=False, spatial_only=False):
    # load a batch of blocks and their corresponding labels
    j = i * batch_size
    idxs = train_idxs[j:j+batch_size]
    for k in range(idxs.shape[0]):
        name = idxs[k]
        block, b_labels = load_block(block_dir, name, spatial_only=spatial_only)
        if len(b_labels.shape) == 2:
            b_labels = np.squeeze(b_labels, -1)
        elif len(b_labels.shape) > 2:
            raise Exception("Unexpected shape of labels" + str(b_labels.shape))
        if apply_random_rotation:
            rot = R.random().as_matrix()
            block = np.matmul(block, rot)
        blocks[k] = block
        labels[k] = b_labels
    return blocks, labels


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
    parser.add_argument("--with_color", type=bool, default=False, help="Use color in training.")
    args = parser.parse_args()

    p_dim = 3
    if args.with_color:
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
    block_dirs = os.listdir(block_dir)
    if args.transfer_train_p < 1.0:
        size = math.floor(args.transfer_train_p * len(block_dirs))
        all_idxs = np.arange(size).astype(np.int32)
    else:
        all_idxs = np.arange(len(block_dirs)).astype(np.int32)
    if all_idxs.shape[0] < batch_size:
        raise Exception("Batch size ({0}) is greater than the number of training examples ({1}).".format(batch_size, all_idxs.shape[0]))
    train_p = args.train_p
    train_n = math.floor(train_p * len(all_idxs))
    test_n = len(all_idxs) - train_n
    print("Use {0} blocks for training and {1} blocks for testing".format(train_n, test_n))
    train_idxs = np.random.choice(all_idxs, size=train_n, replace=False)
    test_idxs = np.delete(all_idxs, train_idxs)
    np.random.shuffle(train_idxs)

    # determine the number of batches
    n_batches = math.floor(train_n / batch_size)
    n_t_batches = math.floor(test_n / batch_size)

    n_epoch = 0
    max_epoch = args.max_epoch

    # prepare containers to store the batches
    b, l = load_block(block_dir, 0, spatial_only=not args.with_color)
    blocks = np.zeros((batch_size, ) + b.shape, np.float32)
    labels = np.zeros((batch_size, ) + (l.shape[0], ), np.uint8)
    t_blocks = np.zeros((batch_size, ) + b.shape, np.float32)
    t_labels = np.zeros((batch_size, ) + (l.shape[0], ), np.uint8)

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

    # training and testing
    while n_epoch < max_epoch:
        if n_epoch % test_interval == 0: # test step
            accs = []
            # number of accuracy calculations in the test phase
            n_acc = t_labels.shape[0] * t_labels.shape[1] * n_t_batches
            # intersection over unions (ious)
            ious = []

            for i in range(n_t_batches): # execute n_t_batches test steps
                t_blocks, t_labels = load_batch(i, test_idxs, block_dir, t_blocks, t_labels, batch_size, spatial_only=not args.with_color)
                t, pred = net(t_blocks, training=False)
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
                blocks, labels = load_batch(i, train_idxs, block_dir, blocks, labels, batch_size, apply_random_rotation=True, spatial_only=not args.with_color)
                t, pred = net(blocks, training=True)
                loss, seg_loss, mat_diff_loss = get_loss(seg_pred=pred, seg=labels, t=t)
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
                tf.summary.scalar("train/mat_diff_loss", mat_diff_loss, step=train_step)
                tf.summary.scalar("train/global_norm", global_norm, step=train_step)
            train_summary_writer.flush()
            train_step += 1
        n_epoch += 1



if __name__ == "__main__":
    main()
