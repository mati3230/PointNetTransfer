from train_test_provider import TrainTestProvider
import os
import numpy as np
#from s3dis_prepare import get_special_objects


class DataProvider(TrainTestProvider):
    """This class loads the S3DIS dataset data. It returns the point clouds
    and their ground truth partitions.

    Parameters
    ----------
    max_scenes : int
        Number of scenes/point clouds that should be used.
    verbose : boolean
        If True, log internal values of this class in the terminal.
    train_mode : TrainMode
        Determine which scenes should be used.
    train_p : float
        Percentage of scenes that will be used for training. Only used in
        TrainMode.Train or TrainMode.Test.
    n_cpus : int
        Number of cpus that will be used for training.
    batch_id : int
        Only use a certain batch with batch_id. The batch size is equal to
        the number of cpus.
    max_P_size : int
        Maximum Size of a point cloud.
    transform : bool
        Should a point cloud be transformed in the origin and scaled by the
        distance to the farthest point.
    filter_special_objects : bool
        Filter objects with special semantics such as walls or the ceiling. 

    Attributes
    ----------
    train_mode : TrainMode
        Determine which scenes should be used.
    max_scenes : int
        Number of scenes/point clouds that should be used.
    scenes : list(str)
        A list with all available scenes as strings.
    train_scenes : list(str)
        Scenes that will be used for training.
    train_idxs : np.ndarray
        Shuffled array so that the scene can be querried randomly.
    train_idx : int
        Current index of the training scene.
    test_scenes : list(str)
        Scenes that will be used for testing.
    test_idx : int
        Current index of the test scene.
    verbose : boolean
        If True, log internal values of this class in the terminal.
    P : np.ndarray
        The current point cloud scene.
    partition_vec : np.ndarray
        The current ground truth object values.
    id : int
        Current scene index that is independend of the training mode.
    current_scene_idx : int
        Current scene index that will be used if data provider is not in
        training mode.
    max_P_size : int
        Maximum Size of a point cloud.
    transform : bool
        Should a point cloud be transformed in the origin and scaled by the
        distance to the farthest point.
    filter_s_objs : bool
        Filter objects with special semantics such as walls or the ceiling. 
    """
    def __init__(
            self,
            max_scenes=2000,
            verbose=False,
            train_mode="All",
            train_p=0.8,
            n_cpus=1,
            batch_id=-1,
            max_P_size=10e7,
            transform=True,
            filter_special_objects=True):
        """Constructor.

        Parameters
        ----------
        max_scenes : int
            Number of scenes/point clouds that should be used.
        verbose : type
            If True, log internal values of this class in the terminal.
        train_mode : str
            Determine which scenes should be used.
        train_p : float
            Percentage of scenes that will be used for training. Only used in
            TrainMode.Train or TrainMode.Test.
        n_cpus : int
            Number of cpus that will be used for training.
        batch_id : int
            Only use a certain batch with batch_id. The batch size is equal to
            the number of cpus.
        max_P_size : int
            Maximum Size of a point cloud.
        transform : bool
            Should a point cloud be transformed in the origin and scaled by the
            distance to the farthest point.
        filter_special_objects : bool
            Filter objects with special semantics such as walls or the ceiling. 
        """
        super().__init__(
            max_scenes=max_scenes,
            verbose=verbose,
            train_mode=train_mode,
            train_p=train_p,
            n_cpus=n_cpus,
            batch_id=batch_id,
            max_P_size=max_P_size,
            transform=transform,
            filter_special_objects=filter_special_objects)

    def list_scenes(self):
        """List all the scenes in the data directory where
        the P.npz are located.
        """
        self.scenes = os.listdir("./S3DIS_Scenes")

    def assign_single_scene(self):
        """assign scenes[0] with a certain scene in case that only
        one scene will be used. 
        """
        scene_0 = os.path.isdir("./S3DIS_Scenes/Area1_conferenceRoom_1")
        if self.max_scenes == 1 and scene_0:
            self.scenes[0] = "Area1_conferenceRoom_1"

    def remove_blacklist(self, black_files):
        """Remove in scenes that are listed in the blacklist. """
        for bf in black_files:
            if len(bf) < 3:
                continue
            bf = bf[:-1]
            if bf in self.scenes:
                self.scenes.remove(bf)

    def get_cloud_and_partition(self):
        """Get the point cloud and the ground truth partition.

        Returns
        -------
        tuple(np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray)
            The point cloud, the partition vector, the scene id, the unique values of the
            partition vector, the start indices of the unique values, the counts of the
            unique values.

        """
        # see scannet_provider for an example
        filepath = "./S3DIS_Scenes/" + self.id + "/P.npz"
        data = np.load(filepath, allow_pickle=True)
        P = data["P"]
        partition_vec = data["partition_vec"]
        partition_uni = data["partition_uni"]
        partition_idxs = data["partition_idxs"]
        partition_counts = data["partition_counts"]

        P, partition_vec, partition_uni, partition_idxs, partition_counts = self.filter_special_objects(
            so_func=get_special_objects, P=P, partition_vec=partition_vec, partition_uni=partition_uni, partition_idxs=partition_idxs, partition_counts=partition_counts)
        if self.verbose:
            print("point cloud and partition loaded")
        P, partition_vec, partition_uni, partition_idxs, partition_counts = self.filter_big_cloud(
            P, partition_vec, partition_uni, partition_idxs, partition_counts)
        P = self.transform_scene(P)

        self.P = P
        self.partition_vec = partition_vec
        return self.P, self.partition_vec, self.id, partition_uni, partition_idxs, partition_counts

    def assign_dataset_name(self):
        """Get the name of the data set.

        Parameters
        ----------

        Returns
        -------
        str
            Name of the data set

        """
        return "s3dis"
