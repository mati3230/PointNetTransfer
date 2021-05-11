from base_classifier import BaseClassifier
from base_pointnet import PointNet
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization

class SeSaPointNet(BaseClassifier):
    def __init__(
            self,
            name,
            n_classes,
            seed=None,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform",
            trainable_net=True):
        super().__init__(name=name,
                n_classes=n_classes,
                seed=seed,
                trainable=trainable,
                check_numerics=check_numerics,
                initializer=initializer,
                trainable_net=trainable_net)

    def init_variables(
            self,
            name,
            n_classes,
            trainable=True,
            seed=None,
            initializer="glorot_uniform"):
        self.bn1 = BatchNormalization(name=name+"/bn1")
        self.bn2 = BatchNormalization(name=name+"/bn2")
        self.c1 = Conv1D(
            filters=512,
            kernel_size=1,
            activation="relu",
            name=name+"/c1",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c2 = Conv1D(
            filters=256,
            kernel_size=1,
            activation="relu",
            name=name+"/c2",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c3 = Conv1D(
            filters=n_classes,
            kernel_size=1,
            activation="linear",
            name=name+"/c3",
            trainable=trainable,
            kernel_initializer=initializer)

    def init_net(
            self,
            name,
            seed=None,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform"):
        self.net = PointNet(
            name=name,
            trainable=trainable,
            seed=seed,
            check_numerics=check_numerics,
            initializer=initializer)

    def get_vars(self, net_only=False):
        vars_ = self.net.get_vars()
        if net_only:
            return vars_
        if self.trainable:
            vars_.extend(self.c1.trainable_weights)
            vars_.extend(self.c2.trainable_weights)
            vars_.extend(self.c3.trainable_weights)
            vars_.extend(self.bn1.trainable_weights)
            vars_.extend(self.bn2.trainable_weights)
        else:
            vars_.extend(self.c1.non_trainable_weights)
            vars_.extend(self.c2.non_trainable_weights)
            vars_.extend(self.c3.non_trainable_weights)
            vars_.extend(self.bn1.non_trainable_weights)
            vars_.extend(self.bn2.non_trainable_weights)
        return vars_

    def reset(self):
        self.net.reset()

    def __call__(self, obs, training):
        t, f, g = self.net(obs, training=training)
        g = tf.tile(g, multiples=[1, tf.shape(f)[1], 1])
        x = tf.concat((f, g), axis=-1)
        x = self.c1(x)
        x = self.bn1(x, training=training)
        x = self.c2(x)
        x = self.bn2(x, training=training)
        x = self.c3(x)
        return t, x
