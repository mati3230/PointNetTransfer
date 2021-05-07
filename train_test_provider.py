from base_data_provider import BaseDataProvider
from abc import abstractmethod
import os
import math
import numpy as np


class TrainTestProvider(BaseDataProvider):
    """This class loads the dataset data. It returns the point clouds
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
        """ Constructor.

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
        """
        if max_scenes <= 0:
            raise ValueError("max_scenes should be >0.")
        if train_p <= 0 or train_p >= 1:
            raise ValueError("train_p should be between 0 and 1.")
        if n_cpus <= 0:
            raise ValueError("n_cpus should be >0.")
        super().__init__()
        self.current_scene_idx = 0
        self.max_scenes = max_scenes
        self.verbose = verbose
        self.batch_id = batch_id
        self.train_mode = train_mode
        self.max_P_size = max_P_size
        self.transform = transform
        self.filter_s_objs = filter_special_objects
        self.list_scenes()
        self.id = self.scenes[0]
        if os.path.isfile("blacklist.txt"):
            blacklist = open("blacklist.txt", "r")
            black_files = blacklist.readlines()
            self.remove_blacklist(black_files)
            blacklist.close()
        if self.verbose:
            print(self.scenes)
        self.max_scenes = min(self.max_scenes, len(self.scenes))
        print("Use:", self.max_scenes, "scenes - batch_id:", batch_id, "train_mode:", train_mode)
        if train_mode == "Train" or train_mode == "Test":
            self.train_scenes, self.train_idxs = self.get_scene_and_idxs("Train", train_p, batch_id, n_cpus)
            self.test_scenes, self.test_idxs = self.get_scene_and_idxs("Test", train_p, batch_id, n_cpus)
            np.random.shuffle(self.train_idxs)
            self.train_idx = 0
            self.test_idx = 0
            # print("train scenes:", self.train_scenes)
            print("train scenes:", len(self.train_scenes), self.train_scenes)
            print("test scenes:", len(self.test_scenes), self.test_scenes)
        if train_mode == "All":
            self.scenes = self.scenes[:self.max_scenes]
            self.assign_single_scene()
            print("scenes:", self.scenes)

    def add_id_to_blacklist(self):
        """Add a certain scene to the blacklist. """
        blacklist = open("blacklist.txt", "a")
        blacklist.write("\n" + self.id)
        blacklist.close()
        print("write '", self.id, "' to blacklist")

    def get_scene_and_idxs(self, mode, train_p, batch_id, n_cpus):
        """Calculate the train or test scenes from the set of all scenes.

        Parameters
        ----------
        mode : str
            Train or test mode.
        train_p : float
            Percentage of training data.
        batch_id : int
            Number of worker/cpu.
        n_cpus : int
            Number of workers.

        Returns
        -------
        list(str), np.ndarray
            List of scenes and their corresponding indices.

        """
        n_samples = 0
        offset = 0
        if mode == "Train":
            n_samples = math.floor(self.max_scenes * train_p)
        elif mode == "Test":
            offset = math.floor(self.max_scenes * train_p)
            n_samples = self.max_scenes - offset
        if batch_id != -1:
            frac = n_samples / n_cpus
            frac = math.floor(frac)
            if batch_id == n_cpus - 1:
                n_samples = n_samples - ((n_cpus - 1) * frac)
            else:
                n_samples = frac
            if self.verbose:
                print("-------------batch_id", batch_id, n_samples)
            start = frac * batch_id + offset
            stop = start + n_samples
            if self.verbose:
                print("-------------batch_id", batch_id, start, stop)
            scenes = self.scenes[start:stop]
            idxs = np.arange(n_samples)
        else:
            scenes = self.scenes[:n_samples]
            idxs = np.arange(n_samples)
        return scenes, idxs

    def next_id(self, scenes, idx, idxs=None):
        """Returns the next scene id.

        Parameters
        ----------
        scenes : list(Scene)
            List with point cloud scenes.
        idx : int
            The current scene index.
        idxs : list(int)
            Special indexes from which a scene should be chosen.

        Returns
        -------
        int
            next scene id

        """
        if idxs is not None:
            _idx = idxs[idx]
            self.id = scenes[_idx]
            idx += 1
            if idx == idxs.shape[0]:
                #np.random.shuffle(idxs)
                idx = 0
        else:
            self.id = scenes[idx]
            idx += 1
            if idx == len(scenes):
                idx = 0
        return idx

    def select_id(self):
        """Select the next scene id.

        Parameters
        ----------
        train : boolean
            If True, the scenes will be splitted sampled train scenes. If
            False, from the test scenes. This is only in train_mode relevant.

        Returns
        -------
        int
            ID of the current scene.

        """
        if self.train_mode == "Train":
            self.train_idx = self.next_id(
                self.train_scenes, self.train_idx, idxs=self.train_idxs)
        elif self.train_mode == "Test":
            self.test_idx = self.next_id(
                self.test_scenes, self.test_idx, idxs=self.test_idxs)
        else: # All
            self.current_scene_idx = self.next_id(self.scenes,
                self.current_scene_idx)
        return self.id

    def filter_big_cloud(self, P, partition_vec, partition_uni, partition_idxs, partition_counts):
        """Filter point clouds that are too big.

        Parameters
        ----------
        P : np.ndarray
            Point cloud
        partition_vec : np.ndarray
            Partition vector.
        partition_uni : np.ndarray
            The unique values of the partition vector.
        partition_idxs : np.ndarray
            The start indices of the unique values.
        partition_counts : np.ndarray
            The counts of the unique values.

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            The point cloud, the partition vector, the unique values of the
            partition vector, the start indices of the unique values, the counts of the
            unique values.

        """
        if P.shape[0] > self.max_P_size:
            idxs = np.arange(P.shape[0])
            idxs = np.random.choice(idxs, size=self.max_P_size)
            P = P[idxs, :]
            partition_vec = partition_vec[idxs]
            partition_uni, partition_idxs, partition_counts = np.unique(partition_vec, return_index=True, return_counts=True)
        return P, partition_vec, partition_uni, partition_idxs, partition_counts

    def filter_special_objects(self, so_func, P, partition_vec, partition_uni, partition_idxs, partition_counts):  
        """Filter point clouds that are too big.

        Parameters
        ----------
        so_func : function
            Function that returns a dictionary where an object labels are assigned to
            partition values. 
        P : np.ndarray
            Point cloud
        partition_vec : np.ndarray
            Partition vector.
        partition_uni : np.ndarray
            The unique values of the partition vector.
        partition_idxs : np.ndarray
            The start indices of the unique values.
        partition_counts : np.ndarray
            The counts of the unique values.

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            The point cloud, the partition vector, the unique values of the
            partition vector, the start indices of the unique values, the counts of the
            unique values.

        """  
        if self.filter_s_objs:    
            special_objects = so_func()
            for k, v in special_objects.items():
                #print(k, v)
                idxs_to_del = np.where(partition_uni == v)[0]
                if idxs_to_del.shape[0] == 0:
                    continue
                if idxs_to_del.shape[0] > 1:
                    raise Exception("Too many indices")
                idxs_to_del = np.where(partition_vec == v)[0]
                partition_vec = np.delete(partition_vec, idxs_to_del)
                P = np.delete(P, idxs_to_del, axis=0)
                partition_uni, partition_idxs, partition_counts = np.unique(partition_vec, return_index=True, return_counts=True)
        return P, partition_vec, partition_uni, partition_idxs, partition_counts

    def transform_scene(self, P):
        """Filter point clouds that are too big.

        Parameters
        ----------
        P : np.ndarray
            A point cloud.

        Returns
        -------
        np.ndarray
            The scaled point cloud. 

        """
        if self.transform:
            max_P = np.max(np.abs(P[:, :3]))
            P[:, :3] /= max_P
        return P

    @abstractmethod
    def get_cloud_and_partition(self):
        """Get the point cloud and the ground truth partition.

        Parameters
        ----------

        Returns
        -------
        tuple(np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray)
            The point cloud, the partition vector, the scene id, the unique values of the
            partition vector, the start indices of the unique values, the counts of the
            unique values.

        """ 
        pass

    @abstractmethod
    def list_scenes(self):
        """List all the scenes in the data directory where
        the P.npz are located.
        """
        pass

    @abstractmethod
    def assign_single_scene(self):
        """assign scenes[0] with a certain scene in case that only
        one scene will be used. 
        """
        pass

    @abstractmethod
    def remove_blacklist(self, black_files):
        """Remove in scenes that are listed in the blacklist. """
        pass

    @abstractmethod
    def assign_dataset_name(self):
        """Get the name of the data set.

        Parameters
        ----------

        Returns
        -------
        str
            Name of the data set

        """
        pass
