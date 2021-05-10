from abc import ABC, abstractmethod
from utils import mkdir, file_exists
import numpy as np


class BaseClassifier(ABC):
    def __init__(
            self,
            name,
            n_classes,
            seed=None,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform",
            trainable_net=True):
        super().__init__()
        self.name = name
        self.seed = seed
        self.check_numerics = check_numerics
        self.n_classes = n_classes
        self.trainable = trainable
        self.trainable_net = trainable_net
        self.init_net(
            name=name,
            seed=seed,
            trainable=trainable_net,
            check_numerics=check_numerics,
            initializer=initializer)
        self.init_variables(
            name=name,
            n_classes=n_classes,
            trainable=trainable,
            seed=seed,
            initializer=initializer)

    @abstractmethod
    def __call__(self, obs):
        pass

    @abstractmethod
    def init_variables(
            self,
            name,
            n_classes,
            trainable=True,
            seed=None,
            initializer="glorot_uniform"):
        pass

    @abstractmethod
    def init_net(
            self,
            name,
            seed=None,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform"):
        pass

    @abstractmethod
    def get_vars(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def save(self, directory, filename, net_only=False):
        mkdir(directory)
        vars_ = self.get_vars(net_only=net_only)
        if len(vars_) == 0:
            raise Exception("At least one variable is expected")
        var_dict = {}
        for var_ in vars_:
            #print(str(var_.name))
            var_dict[str(var_.name)] = np.array(var_.value())
        np.savez(directory + "/" + filename + ".npz", **var_dict)

    def load(self, directory, filename, net_only=False):
        filepath = directory + "/" + filename + ".npz"
        if not file_exists(filepath):
            raise Exception("File path '" + filepath + "' does not exist")
        model_data = np.load(filepath, allow_pickle=True)
        vars_ = self.get_vars(net_only=net_only)
        if net_only:
            for i in range(len(vars_)):
                var_name = vars_[i].name
                if var_name not in model_data:
                    raise Exception("Got no variable with the name " + var_name)
                model_var = model_data[var_name]
                vars_[i].assign(model_var)
        else:
            if len(vars_) != len(model_data):
                keys = list(model_data.keys())
                print("Expected:", len(vars_), "layer; Got:", len(model_data), "layer, file:", filepath)
                if len(vars_) == 0 or len(model_data) == 0:
                    raise Exception("You have to apply a prediction with, e.g., random data to initialize the weights of the network.")
                for i in range(min(len(vars_), len(model_data))):
                    print(vars_[i].name, "\t", keys[i])
                print("Expected:")
                for i in range(len(vars_)):
                    print(vars_[i].name)
                raise Exception("data mismatch")
            i = 0
            for key, value in model_data.items():
                varname = str(vars_[i].name)
                if np.isnan(value).any():
                    raise Exception("loaded value is NaN")
                if key != varname:
                    raise Exception(
                        "Variable names mismatch: " + key + ", " + varname)
                vars_[i].assign(value)
                i += 1
