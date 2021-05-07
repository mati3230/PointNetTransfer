import argparse
import os
import numpy as np
from utils import mkdir, file_exists, render_point_cloud, room2blocks


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


def create_blocks(num_points=4096):
    mkdir("./Blocks")
    scenes = os.listdir("./S3DIS_Scenes")
    block_n = 0
    for i in range(len(scenes)):
        scene = scenes[i]
        P, labels = load_scene(scene)
        blocks, b_labels = room2blocks(data=P, label=labels, num_point=num_points)
        for k in range(blocks.shape[0]):
            block = blocks[k]
            b_label = b_labels[k]
            np.savez("./Blocks/" + str(block_n) + ".npz", block=block, labels=b_label)
            block_n += 1
    print(block_n, "Blocks saved.")


def prepare_scenes(dataset_name):
    """Method to prepare the scenes.

    Parameters
    ----------
    dataset_name : str
        Name of the s3dis data set to create a blacklist.
    """
    mkdir("./S3DIS_Scenes")
    s3dis_dir = os.environ["S3DIS_DIR"] + "/data"
    special_objects = get_special_objects()

    obj_labels = {}
    obj_label_nr = 0

    for dir in os.listdir(s3dis_dir):
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

    for dir in os.listdir(s3dis_dir):
        area_dir = s3dis_dir + "/" + dir
        for scene in os.listdir(area_dir):
            scene_name = "Area" + dir[-1] + "_" + scene
            # print(scene_name)
            n_scene_dir = "./S3DIS_Scenes/" + scene_name
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
                #print(partition_vec.shape)
                partition_uni, partition_idxs, partition_counts = np.unique(partition_vec, return_index=True, return_counts=True)
                #print(partition_uni)

                np.savez(n_scene_dir + "/P.npz", P=P, labels=label_vec, partition_vec=partition_vec, partition_uni=partition_uni, partition_idxs=partition_idxs, partition_counts=partition_counts)


def load_scene(scene):
    filename = "./S3DIS_Scenes/" + scene + "/P.npz"
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
        create_blocks(num_points=args.num_points)
    else:
        prepare_scenes("s3dis")
        if args.mode != "visualize_all":
            return
        scenes = os.listdir("./S3DIS_Scenes")
        for i in range(len(scenes)):
            scene = scenes[i]
            P, labels = load_scene(scene)
            print(scene, P.shape, labels.shape, "progress:", i, "/", len(scenes))
            render_point_cloud(P=P, animate=args.animate)
            render_point_cloud(
                P=P, partition_vec=partition_vec, animate=args.animate)


if __name__ == "__main__":
    main()