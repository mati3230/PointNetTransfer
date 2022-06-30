import os
import numpy as np
import tensorflow as tf

from sesa_pointnet import SeSaPointNet
from utils import get_labelled_blocks


def load_net(block, model_dir=".", filename="pointnet", n_classes=14, seed=42,
        check_numerics=True, initializer="glorot_uniform"):
    net = SeSaPointNet(
            name="SeSaPN",
            n_classes=n_classes,
            seed=seed,
            trainable=True,
            check_numerics=check_numerics,
            initializer=initializer,
            trainable_net=False)
    net(block, training=False)
    net.reset()
    net.load(directory=model_dir, filename=filename, net_only=False)
    print("model loaded")
    return net
    

def main():
    dataset = "S3DIS"
    n_classes = -1
    if dataset == "S3DIS":
        n_classes = 14
    scenes = os.listdir("./Scenes/" + dataset)
    scene = scenes[0]
    blocks, b_labels, P, _, sample_indices = get_labelled_blocks(scene=scene, dataset=dataset, num_points=4096)
    block_ = np.expand_dims(blocks[0], axis=0)
    net = load_net(block=block_, n_classes=n_classes)
    p_vec = -np.ones((P.shape[0], ), dtype=np.int32)
    for i in range(blocks.shape[0]):
        block = blocks[i]
        block_ = np.expand_dims(block, axis=0)
        indices = sample_indices[i]
        _, pred = net(block_, training=False)
        pred = tf.nn.softmax(pred)
        pred = pred.numpy()
        pred = np.argmax(pred, axis=-1)
        p_vec[indices] = pred
    # TODO visualize p_vec

if __name__ == "__main__":
    main()