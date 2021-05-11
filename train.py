import tensorflow as tf
import numpy as np
import math
import datetime
import argparse
import os
from sesa_pointnet import SeSaPointNet


def get_loss(seg_pred, seg, t, reg_f=1e-3, check_numerics=True):
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


def load_block(block_dir, name):
    filename = block_dir + "/" + str(name) + ".npz"
    data = np.load(filename)
    block = data["block"]
    b_labels = data["labels"]

    max_block = np.max(np.abs(block[:, :3]))
    block[:, :3] /= max_block

    return block, b_labels


def load_batch(i, train_idxs, block_dir, blocks, labels, batch_size):
    j = i * batch_size
    idxs = train_idxs[j:j+batch_size]
    for k in range(idxs.shape[0]):
        name = idxs[k]
        block, b_labels = load_block(block_dir, name)
        if len(b_labels.shape) == 2:
            b_labels = np.squeeze(b_labels, -1)
        elif len(b_labels.shape) > 2:
            raise Exception("Unexpected shape of labels" + str(b_labels.shape))
        blocks[k] = block
        labels[k] = b_labels
    return blocks, labels


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
    args = parser.parse_args()

    seed = args.seed
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    global_norm_t = args.global_norm_t
    test_interval = args.test_interval
    block_dir = "Blocks_" + args.dataset
    n_classes = args.n_classes
    np.random.seed(seed)

    block_dirs = os.listdir(block_dir)
    all_idxs = np.arange(len(block_dirs)).astype(np.int32)
    train_p = args.train_p
    train_n = math.floor(train_p * len(all_idxs))
    test_n = len(all_idxs) - train_n
    print("Use {0} blocks for training and {1} blocks for testing".format(train_n, test_n))
    train_idxs = np.random.choice(all_idxs, size=train_n, replace=False)
    test_idxs = np.delete(all_idxs, train_idxs)
    np.random.shuffle(train_idxs)

    n_batches = math.floor(train_n / batch_size)
    n_t_batches = math.floor(test_n / batch_size)

    n_epoch = 0
    max_epoch = args.max_epoch

    b, l = load_block(block_dir, 0)
    blocks = np.zeros((batch_size, ) + b.shape, np.float32)
    labels = np.zeros((batch_size, ) + (l.shape[0], ), np.uint8)
    t_blocks = np.zeros((batch_size, ) + b.shape, np.float32)
    t_labels = np.zeros((batch_size, ) + (l.shape[0], ), np.uint8)

    if args.load:
        net = SeSaPointNet(
            name="SeSaPN",
            n_classes=n_classes,
            seed=seed,
            trainable=True,
            check_numerics=args.check_numerics,
            initializer=args.initializer,
            trainable_net=False)
        tmp_b = np.array(b, copy=True)
        tmp_b = np.expand_dims(b, axis=0)
        net(tmp_b, training=False)
        net.load(directory=args.model_dir, filename=args.model_file, net_only=True)
    else:
        net = SeSaPointNet(
            name="SeSaPN",
            n_classes=n_classes,
            seed=seed,
            trainable=True,
            check_numerics=args.check_numerics,
            initializer=args.initializer,
            trainable_net=True)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "./logs/" + current_time
    train_summary_writer = tf.summary.create_file_writer(log_dir)
    train_step = 0
    test_step = 0

    while n_epoch < max_epoch:
        for i in range(n_batches):
            with tf.GradientTape() as tape:
                blocks, labels = load_batch(i, train_idxs, block_dir, blocks, labels, batch_size)
                t, pred = net(blocks, training=True)
                loss, seg_loss, mat_diff_loss = get_loss(seg_pred=pred, seg=labels, t=t)
                vars_ = tape.watched_variables()
                grads = tape.gradient(loss, vars_)
                global_norm = tf.linalg.global_norm(grads)
                if global_norm_t > 0:
                    grads, _ = tf.clip_by_global_norm(
                        grads,
                        global_norm_t,
                        use_norm=global_norm)
                optimizer.apply_gradients(zip(grads, vars_))
            with train_summary_writer.as_default():
                tf.summary.scalar("train/loss", loss, step=train_step)
                tf.summary.scalar("train/seg_loss", seg_loss, step=train_step)
                tf.summary.scalar("train/mat_diff_loss", mat_diff_loss, step=train_step)
                tf.summary.scalar("train/global_norm", global_norm, step=train_step)
            train_summary_writer.flush()
            train_step += 1
        if n_epoch % test_interval == 0:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            net.save(directory="./models/" + args.dataset, filename="pointnet_" + current_time, net_only=False)
            acc = 0
            for i in range(n_t_batches):
                t_blocks, t_labels = load_batch(i, test_idxs, block_dir, t_blocks, t_labels, batch_size)
                t, pred = net(t_blocks, training=False)
                pred = tf.nn.softmax(pred)
                pred = pred.numpy()
                classes = np.argmax(pred, axis=-1)
                tp = np.where(classes == t_labels)
                n_points = t_labels.shape[0]*t_labels.shape[1]
                acc += tp[0].shape[0] / n_points
            acc /= n_t_batches
            with train_summary_writer.as_default():
                tf.summary.scalar("test/mean_acc", acc, step=test_step)
            train_summary_writer.flush()
            test_step += 1
        n_epoch += 1



if __name__ == "__main__":
    main()
