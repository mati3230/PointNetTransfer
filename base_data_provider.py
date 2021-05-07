from abc import ABC, abstractmethod


class BaseDataProvider(ABC):
    """Abstract class for an environment that uses data from a storage.
    For example, a realization of this class is used to load and preprocess
    point cloud data (see '../scannet_provider.py')"""
    def __init__(self):
        super().__init__()
        self.dataset_name = self.assign_dataset_name()

    @abstractmethod
    def get_cloud_and_partition(self):
        """The method should return a point cloud and the corresponding
        superpoints.

        Returns
        -------
        np.ndarray
            Point Cloud
        np.ndarray
            Vector with superpoint numbers for each point
        """
        pass

    @abstractmethod
    def assign_dataset_name(self):
        pass
