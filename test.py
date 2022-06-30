import os
import numpy as np
import tensorflow as tf
import open3d as o3d
from tqdm import tqdm

from sesa_pointnet import SeSaPointNet
from utils import get_blocks
import visu_utils


def p_vec2sp_idxs(p_vec):
    uni_p_vec = np.unique(p_vec)
    sp_idxs = []
    for i in range(uni_p_vec.shape[0]):
        val = uni_p_vec[i]
        sp_idx = np.where(p_vec == val)[0]
        if sp_idx.shape[0] == 0:
            continue
        sp_idxs.append(sp_idx)
    return sp_idxs


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
    mesh = o3d.io.read_triangle_mesh("./sn000000.ply")
    xyz = np.asarray(mesh.vertices)
    rgb = np.asarray(mesh.vertex_colors)
    P = np.hstack((xyz, rgb))
    P = P.astype(np.float32)
    n_classes = 14

    blocks, sample_indices = get_blocks(P=P, num_points=4096)
    block_ = np.expand_dims(blocks[0], axis=0)
    net = load_net(block=block_, n_classes=n_classes)
    p_vec = -np.ones((P.shape[0], ), dtype=np.int32)
    for i in tqdm(range(blocks.shape[0]), desc="Classify Blocks", disable=True):
        block = blocks[i]
        block_ = np.expand_dims(block, axis=0)
        indices = sample_indices[i]
        #"""
        _, pred = net(block_, training=False)
        pred = tf.nn.softmax(pred)
        pred = pred.numpy()
        pred = np.argmax(pred, axis=-1)
        p_vec[indices] = pred[0, :indices.shape[0]]
        #"""
    colors = visu_utils.load_colors(cpath="./colors.npz")
    colors = colors/255.
    sp_idxs = p_vec2sp_idxs(p_vec=p_vec)
    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=sp_idxs, colors=colors, w_co=False)

if __name__ == "__main__":
    main()