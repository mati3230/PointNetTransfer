import argparse
import os
import numpy as np
from utils import mkdir, file_exists, render_point_cloud, create_blocks


def prepare_scenes():
    mkdir("./PCG_Scenes")
    pcg_dir = os.environ["PCG_DIR"] + "/data"
    special_objects = get_special_objects()

    i = 1
    for filename in os.listdir(pcg_dir):
        scene_file = pcg_dir + "/" + filename
        if len(scene_file) <= 4:
            continue
        if not scene_file.endswith(".csv"):
            continue
        data = np.loadtxt(scene_file, delimiter=";", skiprows=1)
        P = data[:, :6]
        
        xyz_min = np.min(P[:, :3], axis=0)
        P[:, :3] = P[:, :3] - xyz_min
        
        label_vec = data[:, 6]

        n_scene_dir = "./PCG_Scenes/scene" + str(i)

        np.savez(n_scene_dir + "/P.npz", P=P, labels=label_vec)
        i += 1


def load_scene(scene):
    filename = "./PCG_Scenes/" + scene + "/P.npz"
    data = np.load(filename)
    P = data["P"]
    labels = data["labels"]
    return P, labels


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
        default="Area1_office_30",
        help="scene from the scannet dataset")
    parser.add_argument(
        "--use_scene",
        type=bool,
        default=False,
        help="used if all scenes are visualized")
    parser.add_argument(
        "--render_segs",
        type=bool,
        default=False,
        help="flag to render every superpoint of scene")
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="flag to print output")
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
        P, labels = load_scene(args.scene)
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
            P, labels = load_scene(scene)
            print(scene, P.shape, labels.shape, "progress:", i, "/", len(scenes))
            render_point_cloud(P=P, animate=args.animate)
            render_point_cloud(
                P=P, partition_vec=partition_vec, animate=args.animate)


if __name__ == "__main__":
    main()
