import argparse
import os
import numpy as np
from utils import mkdir, file_exists, render_point_cloud, create_blocks, load_scene


def prepare_scenes():
    mkdir("./PCG_Scenes")
    pcg_dir = os.environ["PCG_DIR"] + "/data"

    max_label = 0
    i = 1
    for filename in os.listdir(pcg_dir):
        scene_file = pcg_dir + "/" + filename
        if len(scene_file) <= 4:
            continue
        if not scene_file.endswith(".csv"):
            continue

        n_scene_dir = "./PCG_Scenes/scene" + str(i)
        data = np.loadtxt(scene_file, delimiter=";", skiprows=1)
        label_vec = data[:, 6].astype(np.uint8)
        if not file_exists(n_scene_dir + "/P.npz"):
            P = data[:, :6]
            
            xyz_min = np.min(P[:, :3], axis=0)
            P[:, :3] = P[:, :3] - xyz_min
            

            mkdir(n_scene_dir)
            np.savez(n_scene_dir + "/P.npz", P=P, labels=label_vec)
        uni = np.unique(label_vec)
        ml = int(np.max(uni))
        max_label = max(max_label, ml)
        i += 1
    print("got max_label", max_label)


def main():
    """Program entry point. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="precalc",
        help="options: precalc, visualize_all, visualize_single")
    parser.add_argument(
        "--scene",
        type=str,
        default="scene1",
        help="scene from the scannet dataset")
    parser.add_argument(
        "--animate",
        type=bool,
        default=False,
        help="If True, the point cloud will be animated")
    parser.add_argument(
        "--num_points",
        type=int,
        default=4096,
        help="Number of points per block")
    args = parser.parse_args()
    print(args)
    print("mode:", args.mode)
    if args.mode == "visualize_single":
        P, labels = load_scene(dataset="PCG", scene=args.scene)
        print(args.scene, P.shape, labels.shape)
        P[:, 3:] *= 255
        render_point_cloud(P=P, animate=args.animate)
        render_point_cloud(
            P=P, partition_vec=labels, animate=args.animate)
    elif args.mode == "blocks":
        create_blocks(dataset="PCG", num_points=args.num_points)
    else:
        prepare_scenes()
        if args.mode != "visualize_all":
            return
        scenes = os.listdir("./PCG_Scenes")
        for i in range(len(scenes)):
            scene = scenes[i]
            P, labels = load_scene(dataset="PCG", scene=scene)
            print(scene, P.shape, labels.shape, "progress:", i, "/", len(scenes))
            render_point_cloud(P=P, animate=args.animate)
            render_point_cloud(
                P=P, partition_vec=partition_vec, animate=args.animate)


if __name__ == "__main__":
    main()
