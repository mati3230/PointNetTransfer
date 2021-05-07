import tensorflow as tf
import numpy as np
import math
import datetime
import argparse
from sesa_pointnet import SeSaPointNet


def loss(seg_pred, seg, t, reg_f=1e-3):
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=seg, logits=seg_pred)

    per_instance_seg_loss = tf.reduce_mean(ce, axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)

    K = tf.shape(t)[1]
    mat_diff = tf.matmul(t, tf.transpose(t, perm=[0,2,1])) - tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)

    total = seg_loss + mat_diff_loss * reg_f 
    return total, seg_loss, mat_diff_loss


def load_block(block_dir, name):
    filename = block_dir + str(name) + ".npz"
    data = np.load(filename)
    block = data["block"]
    b_labels = data["labels"]
    return block, b_labels


def load_batch(i, train_idxs, block_dir, blocks, labels):
    j = i * batch_size
    idxs = train_idxs[j:j+batch_size]
    for k in range(idxs.shape[0]):
        name = idxs[k]
        block, b_labels = load_block(block_dir, name)
        blocks[k] = block
        labels[k] = b_labels
    return blocks, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Set a random seed.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch_size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Training learning rate.")
    parser.add_argument("--global_norm_t", type=float, default=1, help="Training global norm threshold.")
    parser.add_argument("--test_interval", type=int, default=10, help="Interval to test the model.")
    parser.add_argument("--max_epoch", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--initializer", type=str, default="glorot_uniform", help="Initializer of the weights.")
    parser.add_argument("--check_numerics", type=bool, default=False, help="Should NaN or Inf values be checked.")
    args = parser.parse_args()

    seed = args.seed
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    global_norm_t = args.global_norm_t
    test_interval = args.test_interval
    block_dir = "blocks"
    np.random.seed(seed)
    
    block_dirs = os.listdir(block_dir)
    all_idxs = np.arange(block_dirs, np.int32)
    train_p = 0.8
    train_n = math.floor(train_p * len(all_idxs))
    test_n = len(all_idxs) - train_n
    train_idxs = np.random.choice(all_idxs, size=train_n, replace=False)
    test_idxs = np.delete(all_idxs, train_idxs)
    np.random.shuffle(train_idxs)
    
    n_batches = math.floor(train_n / batch_size)
    
    n_epoch = 0
    max_epoch = args.max_epoch

    b, l = load_block(block_dir, 0)
    blocks = np.zeros((batch_size, ) + b.shape, np.float32)
    labels = np.zeros((batch_size, ), np.uint8)

    net = SeSaPointNet(
        name="SeSaPN",
        n_classes=n_classes,
        seed=seed,
        trainable=True,
        check_numerics=args.check_numerics,
        initializer=args.initializer)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "./logs/" + current_time
    train_summary_writer = tf.summary.create_file_writer(log_dir)
    train_step = 0
    test_step = 0
    
    while n_epoch < max_epoch:
        for i in range(n_batches):
            with tf.GradientTape() as tape:
                blocks, labels = load_batch(i, train_idxs, block_dir, blocks, labels)
                t, pred = net(blocks)
                loss, seg_loss, mat_diff_loss = loss(seg_pred=pred, seg=labels, t=t)
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
            net.save(net_only=True)
            accs = []
            for i in range(test_idxs.shape[0]):
                idx = test_idxs[i]
                b, l = load_block(block_dir, idx)
                pred = net(b)
                tp = np.where(pred == l)[0]
                acc = tp / l.shape[0]
                accs.append(acc)
            with train_summary_writer.as_default():
                tf.summary.scalar("test/mean_acc", np.mean(acc), step=test_step)
                tf.summary.scalar("test/std_acc", np.std(acc), step=test_step)
            test_step += 1



if __name__ == "__main__":
    main()