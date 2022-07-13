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
        check_numerics=True, initializer="glorot_uniform", n_points=1024):
    p_dim=9
    net = SeSaPointNet(
            name="SeSaPN",
            n_classes=n_classes,
            seed=seed,
            trainable=True,
            check_numerics=check_numerics,
            initializer=initializer,
            trainable_net=False,
            n_points=n_points,
            p_dim=p_dim)
    net(block, training=False)
    net.reset()
    net.load(directory=model_dir, filename=filename, net_only=False)
    print("model loaded")
    return net
    

def preprocess_blocks(blocks):
    n_blocks = blocks.shape[0]
    for i in range(n_blocks):
        block = blocks[i]
        # translate into the origin
        mean_block = np.mean(block[:, :3], axis=0)
        block[:, :3] -= mean_block

        # uniformly scale to [-1, 1]
        max_block = np.max(np.abs(block[:, :3]))
        print(max_block)
        block[:, :3] /= max_block
        
        # scale point colors to [-0.5, 0.5]
        block[:, 3:] -= 0.5
        # scale point colors to [-1, 1]
        block[:, 3:] *= 2
        blocks[i] = block
    return blocks


def main():
    data = np.load("./s3dis_a1cr1.npz")
    P = data["P"]
    print("Range of rgb colors: [{0}, {1}]".format(np.min(P[:, 3:]), np.max(P[:, 3:])))
    """
    mesh = o3d.io.read_triangle_mesh("./sn000000.ply")
    xyz = np.asarray(mesh.vertices)
    xyz_min = np.min(xyz, axis=0)
    xyz = xyz - xyz_min
    rgb = np.asarray(mesh.vertex_colors)
    print("Range of rgb colors: [{0}, {1}]".format(np.min(rgb), np.max(rgb)))
    P = np.hstack((xyz, rgb))
    """
    P = P.astype(np.float32)
    n_classes = 14
    n_points = 4096
    colors = visu_utils.load_colors(cpath="./colors.npz")
    colors = colors/255.

    blocks, sample_indices = get_blocks(P=P, num_points=n_points)
    # blocks: n_blocks x n_points x p_dim
    #blocks = preprocess_blocks(blocks=blocks)

    print("Have {0} blocks with {1} points per block".format(blocks.shape[0], n_points))
    print("Point cloud has {0} points and all blocks have {1} points".format(P.shape[0], blocks.shape[0] * n_points))
    p_vec = -np.ones((P.shape[0], ), dtype=np.int32)
    for i in range(blocks.shape[0]):
        #block = blocks[i]
        indices = sample_indices[i]
        p_vec[indices] = i
    sp_idxs = p_vec2sp_idxs(p_vec=p_vec)
    #visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=sp_idxs, colors=colors, w_co=True)
    cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(P[:, :3]))
    cloud.colors = o3d.utility.Vector3dVector(P[:, 3:])
    visu_utils.render_partition_o3d_cloud(cloud=cloud, sp_idxs=sp_idxs, colors=colors, w_co=True)
    return

    block_ = np.expand_dims(blocks[0], axis=0)
    net = load_net(block=block_, n_classes=n_classes, n_points=n_points)
    p_vec = -np.ones((P.shape[0], ), dtype=np.int32)
    for i in tqdm(range(blocks.shape[0]), desc="Classify Blocks", disable=True):
        block = blocks[i]
        # TODO add preprocessing of block -- see train.py
        block_ = np.expand_dims(block, axis=0)
        indices = sample_indices[i]
        #"""
        _, pred = net(block_, training=False)
        pred = tf.nn.softmax(pred)
        pred = pred.numpy()
        pred = np.argmax(pred, axis=-1)
        p_vec[indices] = pred[0, :indices.shape[0]]
        #"""
    sp_idxs = p_vec2sp_idxs(p_vec=p_vec)
    visu_utils.render_partition_o3d(mesh=mesh, sp_idxs=sp_idxs, colors=colors, w_co=False)

if __name__ == "__main__":
    main()