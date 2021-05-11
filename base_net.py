from abc import ABC, abstractmethod


class BaseNet(ABC):
    """Abstract class from which a feature detector should inherit. See
    '../policies/README.md' for more information.

    Parameters
    ----------
    name : str
        Name of the neural net.
    trainable : boolean
        If True the value of the neurons can be changed.
    seed : int
        Random seed that should be used.
    check_numerics : boolean
        If True numeric values will be checked in tensorflow calculation to
        detect, e.g., NaN values.
    stateful : boolean
        Set to True if the base net consists a LSTM and if this net is used
        for optimization.
    initializer : str
        Tensorflow initializer for the weights.
    discrete : bool
        Will the output of the head net be discrete?

    Attributes
    ----------
    name : str
        Name of the neural net.
    seed : int
        Random seed that should be used.
    trainable : boolean
        If True the value of the neurons can be changed.
    check_numerics : boolean
        If True numeric values will be checked in tensorflow calculation to
        detect, e.g., NaN values.
    initializer : str
        Tensorflow initializer for the weights.
    stateful : boolean
        Set to False if the base net is used for optimization. Only relevant
        for net that contains an LSTM.
    initializer : str
        Tensorflow initializer for the weights.
    batch_size : int
        Batch size that is used during the weight optimization
    discrete : bool
        Will the output of the head net be discrete?

    """
    def __init__(
            self,
            name,
            trainable=True,
            seed=None,
            check_numerics=False,
            initializer="glorot_uniform"):
        """Short summary.

        Parameters
        ----------
        name : str
            Name of the neural net.
        trainable : boolean
            If True the value of the neurons can be changed.
        seed : int
            Random seed that should be used.
        check_numerics : boolean
            If True numeric values will be checked in tensorflow calculation to
            detect, e.g., NaN values.
        stateful : boolean
            Set to True if the base net consists a LSTM and if this net is used
            for optimization.
        initializer : str
            Tensorflow initializer for the weights.
        discrete : bool
            Will the output of the head net be discrete?

        """
        super().__init__()
        self.name = name
        self.seed = seed
        self.trainable = trainable
        self.check_numerics = check_numerics
        self.initializer = initializer

    @abstractmethod
    def get_vars(self):
        """This method should return the neurons of the neural net.

        Returns
        -------
        tf.Tensor
            Neurons as variable.
        """
        pass

    @abstractmethod
    def __call__(self, obs, training):
        """Compute a feature vector from the observation.

        Parameters
        ----------
        obs : tf.Tensor
            Observation from which the network will calculate features as
            vector.

        Returns
        -------
        tf.Tensor
            Feature vector.

        """
        pass
