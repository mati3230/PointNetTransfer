from base_classifier import BaseClassifier
from base_pointnet import SemSegPointNet
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout

class SeSaPointNet(BaseClassifier):
    def __init__(
            self,
            name,
            n_classes,
            n_points,
            seed=None,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform",
            trainable_net=True,
            p_dim=3):
        self.n_points = n_points
        self.p_dim = p_dim
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
        self.bn3 = BatchNormalization(name=name+"/bn3")
        self.bn4 = BatchNormalization(name=name+"/bn4")
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
        self.d1sesa = Dense(256, activation="relu", name=name+"/d1sesa", trainable=trainable)
        self.d2sesa = Dense(128, activation="relu", name=name+"/d2sesa", trainable=trainable)
        self.dp = Dropout(rate=0.7)

    def init_net(
            self,
            name,
            seed=None,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform"):
        self.net = SemSegPointNet(
            name=name,
            trainable=trainable,
            seed=seed,
            check_numerics=check_numerics,
            initializer=initializer,
            n_points=self.n_points,
            p_dim=self.p_dim)

    def get_vars(self, net_only=False, head_only=False):
        if not head_only:
            vars_ = self.net.get_vars()
            if net_only:
                return vars_
        else:
            vars_ = []
        vars_.extend(self.bn1.trainable_weights)
        vars_.extend(self.bn2.trainable_weights)
        vars_.extend(self.bn3.trainable_weights)
        vars_.extend(self.bn4.trainable_weights)
        vars_.extend(self.bn1.non_trainable_weights)
        vars_.extend(self.bn2.non_trainable_weights)
        vars_.extend(self.bn3.non_trainable_weights)
        vars_.extend(self.bn4.non_trainable_weights)
        if self.trainable:
            vars_.extend(self.c1.trainable_weights)
            vars_.extend(self.c2.trainable_weights)
            vars_.extend(self.c3.trainable_weights)
            vars_.extend(self.d1sesa.trainable_weights)
            vars_.extend(self.d2sesa.trainable_weights)
        else:
            vars_.extend(self.c1.non_trainable_weights)
            vars_.extend(self.c2.non_trainable_weights)
            vars_.extend(self.c3.non_trainable_weights)
            vars_.extend(self.d1sesa.non_trainable_weights)
            vars_.extend(self.d2sesa.non_trainable_weights)
        return vars_

    def reset(self):
        self.net.reset()

    def __call__(self, obs, training):
        batch_size = obs.shape[0]
        num_point = obs.shape[1]
        pf, g = self.net(obs, training=training)
        
        g = tf.reshape(g, [batch_size, -1])
        g = self.d1sesa(g)
        g = self.bn3(g, training=training)
        g = self.d2sesa(g)
        g = self.bn4(g, training=training)
        #print(batch_size, num_point, g.shape, pf.shape)
        #g_expand = tf.tile(tf.reshape(g, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
        g_expand = tf.expand_dims(g, axis=1)
        g_expand = tf.tile(g_expand, multiples=[1, num_point, 1])
        #print(g_expand.shape, pf.shape)
        x = tf.concat([pf, g_expand], axis=2)
        #print(x.shape)
        #x = tf.concat((f, g), axis=-1)
        
        x = self.c1(x)
        x = self.bn1(x, training=training)
        x = self.c2(x)
        x = self.bn2(x, training=training)
        x = self.dp(x, training=training)
        x = self.c3(x)
        return x


if __name__ == "__main__":
    n_classes = 14
    n_points = 1024
    p_dim = 6
    seed = 42
    net = SeSaPointNet(
        name="SeSaPN",
        n_classes=n_classes,
        seed=seed,
        trainable=True,
        check_numerics=True,
        initializer="glorot_uniform",
        trainable_net=True,
        n_points=n_points,
        p_dim=p_dim)
    import numpy as np
    data = np.random.rand(1, n_points, p_dim)
    net(data, training=True)