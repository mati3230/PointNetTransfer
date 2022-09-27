import argparse
import os
import numpy as np
from tqdm import tqdm
from utils import mkdir, file_exists, render_point_cloud, create_blocks, load_scene


def get_special_objects():
    """Create a dictionary where explicit object names assigned to reseverd numbers.

    Returns
    -------
    boolean
        True if the file exists.
    """
    special_objects = {}
    sobjs = ["wall", "floor", "ceiling"]
    for i in range(len(sobjs)):
        obj = sobjs[i]
        special_objects[obj] = i + 1
    special_objects["floor"] = special_objects["wall"]
    return special_objects


def prepare_scenes(dataset_name, random_sample=1):
    """Method to prepare the scenes.

    Parameters
    ----------
    dataset_name : str
        Name of the s3dis data set to create a blacklist.
    """
    mkdir("./Scenes/S3DIS")
    s3dis_dir = os.environ["S3DIS_DIR"] + "/data"
    special_objects = get_special_objects()

    obj_labels = {}
    obj_label_nr = 0

    for dir in os.listdir(s3dis_dir):
        if dir == ".DS_Store":
            continue
        area_dir = s3dis_dir + "/" + dir
        for scene in os.listdir(area_dir):
            scene_dir = area_dir + "/" + scene + "/Annotations"
            if not os.path.isdir(scene_dir):
                continue
            for obj_file in os.listdir(scene_dir):
                if len(obj_file) <= 4:
                    continue
                if not obj_file.endswith(".txt"):
                    continue
                obj_name = obj_file.split("_")[0]
                if obj_name not in obj_labels:
                    obj_labels[obj_name] = obj_label_nr
                    obj_label_nr += 1
    print(obj_label_nr-1, "classes found")
    print(obj_labels)

    for dir in os.listdir(s3dis_dir):
        if dir == ".DS_Store":
            continue
        area_dir = s3dis_dir + "/" + dir
        #for scene in os.listdir(area_dir):
        scenes = os.listdir(area_dir)
        for i in tqdm(range(len(scenes)), desc="Point Cloud {0}".format(dir)):
            scene = scenes[i]
            scene_name = "Area" + dir[-1] + "_" + scene
            # print(scene_name)
            n_scene_dir = "./Scenes/S3DIS/" + scene_name
            if file_exists(n_scene_dir + "/P.npz"):
                continue
            scene_dir = area_dir + "/" + scene + "/Annotations"
            if not os.path.isdir(scene_dir):
                continue
            O = len(special_objects) + 1
            P = np.zeros((0, 6), np.float32)
            partition_vec = np.zeros((0, 1), np.int32)
            label_vec = np.zeros((0, 1), np.uint8)
            mkdir(n_scene_dir)
            ok = True
            for obj_file in os.listdir(scene_dir):
                if len(obj_file) <= 4:
                    continue
                if not obj_file.endswith(".txt"):
                    continue
                obj_name = obj_file.split("_")[0]
                obj_dir = scene_dir + "/" + obj_file
                P_O = np.loadtxt(obj_dir, delimiter=" ")
                try:
                    P = np.vstack((P, P_O))
                except Exception as e:
                    print("Error in scene", n_scene_dir, ": ", e)
                    blacklist = open(dataset_name + "_blacklist.txt", "a")
                    blacklist.write("\n")
                    blacklist.write(n_scene_dir)
                    blacklist.close()
                    ok = False
                    break
                p_vec = np.ones((P_O.shape[0], 1), np.int32)
                l_vec = np.ones((P_O.shape[0], 1), np.uint8)
                l_vec *= obj_labels[obj_name]
                label = obj_file.split("_")[0]
                if label in special_objects:
                    p_vec *= special_objects[label]
                else:
                    p_vec *= O
                    O += 1
                partition_vec = np.vstack((partition_vec, p_vec))
                label_vec = np.vstack((label_vec, l_vec))
            if ok:
                xyz_min = np.min(P[:, :3], axis=0)
                P[:, :3] = P[:, :3] - xyz_min

                partition_vec = partition_vec.reshape(partition_vec.shape[0], )
                #print(partition_vec.shape)
                sortation = np.argsort(partition_vec)
                #print(sortation.shape)
                P = P[sortation, :]
                P[:, 3:] /= 255
                #print(P.shape)
                label_vec = label_vec[sortation]
                partition_vec = partition_vec[sortation]

                if random_sample > 0 and random_sample < 1:
                    print(P.shape[0])
                    size = random_sample * P.shape[0]
                    idxs = np.arange(P.shape[0])
                    idxs = np.choice(a=idxs, size=size, replace=False)
                    P = P[idxs]
                    print(P.shape[0])
                    print("")
                    label_vec = label_vec[idxs]
                    partition_vec = partition_vec[idxs]
                #print(partition_vec.shape)
                partition_uni, partition_idxs, partition_counts = np.unique(partition_vec, return_index=True, return_counts=True)
                #print(partition_uni)

                np.savez(n_scene_dir + "/P.npz", P=P, labels=label_vec, partition_vec=partition_vec, partition_uni=partition_uni, partition_idxs=partition_idxs, partition_counts=partition_counts)


def main():
    """Program entry point. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="precalc",
        help="options: precalc, blocks, visualize_single, visualize_all")
    parser.add_argument(
        "--scene",
        type=str,
        default="Area1_office_30",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Number of blocks per batch")
    parser.add_argument(
        "--random_sample",
        type=float,
        default=1,
        help="Choose a number between 0 and 1 in order to apply a random sampling of the point clouds")
    args = parser.parse_args()
    print(args)
    print("mode:", args.mode)
    if args.mode == "visualize_single":
        P, labels = load_scene(dataset="S3DIS", scene=args.scene)
        print(args.scene, P.shape, labels.shape)
        P[:, 3:] *= 255
        render_point_cloud(P=P, animate=args.animate)
        render_point_cloud(
            P=P, partition_vec=labels, animate=args.animate)
    elif args.mode == "blocks":
        create_blocks(dataset="S3DIS", num_points=args.num_points, batch_size=args.batch_size)
    else:
        prepare_scenes("s3dis", random_sample=args.random_sample)
        if args.mode != "visualize_all":
            return
        scenes = os.listdir("./Scenes/S3DIS")
        for i in range(len(scenes)):
            scene = scenes[i]
            P, labels = load_scene(dataset="S3DIS", scene=scene)
            print(scene, P.shape, labels.shape, "progress:", i, "/", len(scenes))
            render_point_cloud(P=P, animate=args.animate)
            render_point_cloud(
                P=P, partition_vec=partition_vec, animate=args.animate)


if __name__ == "__main__":
    main()
