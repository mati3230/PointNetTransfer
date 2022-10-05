import tensorflow as tf
import numpy as np
from base_net import BaseNet
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, Conv2D

class PointNet(BaseNet):
    def __init__(
            self,
            name,
            trainable=True,
            seed=None,
            check_numerics=False,
            initializer="glorot_uniform",
            n_points=4096,
            p_dim=3
            ):
        super().__init__(
            name=name,
            trainable=trainable,
            seed=seed,
            check_numerics=check_numerics,
            initializer=initializer)
        self.n_points = n_points
        self.p_dim = p_dim

        self.bn1g = BatchNormalization(name=name+"/bn1g")
        self.bn2g = BatchNormalization(name=name+"/bn2g")
        self.bn3g = BatchNormalization(name=name+"/bn3g")
        self.bn4g = BatchNormalization(name=name+"/bn4g")
        self.bn5g = BatchNormalization(name=name+"/bn5g")

        self.bn1itn = BatchNormalization(name=name+"/bn1itn")
        self.bn2itn = BatchNormalization(name=name+"/bn2itn")
        self.bn3itn = BatchNormalization(name=name+"/bn3itn")
        self.bn4itn = BatchNormalization(name=name+"/bn4itn")
        self.bn5itn = BatchNormalization(name=name+"/bn5itn")

        self.bn1ftn = BatchNormalization(name=name+"/bn1ftn")
        self.bn2ftn = BatchNormalization(name=name+"/bn2ftn")
        self.bn3ftn = BatchNormalization(name=name+"/bn3ftn")
        self.bn4ftn = BatchNormalization(name=name+"/bn4ftn")
        self.bn5ftn = BatchNormalization(name=name+"/bn5ftn")
        self.bn6ftn = BatchNormalization(name=name+"/bn6ftn")

        self.c1itn = Conv2D(
            filters=64,
            kernel_size=[1, self.p_dim],
            activation="relu",
            name=name+"/c1itn",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c2itn = Conv1D(
            filters=128,
            kernel_size=1,
            activation="relu",
            name=name+"/c2itn",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c3itn = Conv1D(
            filters=1024,
            kernel_size=1,
            activation="relu",
            name=name+"/c3itn",
            trainable=trainable,
            kernel_initializer=initializer)
        self.mp1itn = MaxPool1D(name=name + "/mp1itn", pool_size=self.n_points)
        self.d1itn = Dense(512, activation="relu", name=name+"/d1itn", trainable=trainable)
        self.d2itn = Dense(256, activation="relu", name=name+"/d2itn", trainable=trainable)
        bias3 = np.eye(self.p_dim).flatten().astype(np.float32)
        self.d3itn = Dense(self.p_dim**2, weights=[np.zeros([256, self.p_dim**2]), bias3], name=name+"/d3itn", trainable=trainable)

        self.c1g = Conv1D(
            filters=64,
            kernel_size=1,
            activation="relu",
            name=name+"/c1g",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c2g = Conv1D(
            filters=64,
            kernel_size=1,
            activation="relu",
            name=name+"/c2g",
            trainable=trainable,
            kernel_initializer=initializer)

        self.c1ftn = Conv1D(
            filters=64,
            kernel_size=1,
            activation="relu",
            name=name+"/c1ftn",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c2ftn = Conv1D(
            filters=128,
            kernel_size=1,
            activation="relu",
            name=name+"/c2ftn",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c3ftn = Conv1D(
            filters=1024,
            kernel_size=1,
            activation="relu",
            name=name+"/c3ftn",
            trainable=trainable,
            kernel_initializer=initializer)
        self.mp1ftn = MaxPool1D(name=name + "/mp1ftn", pool_size=self.n_points)
        self.d1ftn = Dense(512, activation="relu", name=name+"/d1ftn", trainable=trainable)
        self.d2ftn = Dense(256, activation="relu", name=name+"/d2ftn", trainable=trainable)
        bias4 = np.eye(64).flatten().astype(np.float32)
        self.d3ftn = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), bias4], name=name + "/d3ftn", trainable=trainable)

        self.c3g = Conv1D(
            filters=64,
            kernel_size=1,
            activation="relu",
            name=name+"/c3g",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c4g = Conv1D(
            filters=128,
            kernel_size=1,
            activation="relu",
            name=name+"/c4g",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c5g = Conv1D(
            filters=1024,
            kernel_size=1,
            activation="relu",
            name=name+"/c5g",
            trainable=trainable,
            kernel_initializer=initializer)
        self.mp1g = MaxPool1D(name=name + "/mp1g", pool_size=self.n_points)

        self.r1itn = Reshape((self.p_dim, self.p_dim), name=name+"/r1itn")
        self.r1ftn = Reshape((64, 64), name=name+"/r2itn")

    def feature_t_net(self, g, training):
        # feature transform net
        f = self.c1ftn(g)
        f = self.bn1ftn(f, training=training)
        #print(f.shape)
        f = self.c2ftn(f)
        f = self.bn2ftn(f, training=training)
        #print(f.shape)
        f = self.c3ftn(f)
        f = self.bn3ftn(f, training=training)
        #print(f.shape)
        f = self.mp1ftn(f)
        #print(f.shape)
        f = self.d1ftn(f)
        f = self.bn4ftn(f, training=training)
        #print(x.shape)
        f = self.d2ftn(f)
        f = self.bn5ftn(f, training=training)
        #print(x.shape)
        f = self.d3ftn(f)
        #print(f.shape)
        feature_T = self.r1ftn(f)
        return feature_T

    def input_t_net(self, input_points, training):
        # input_Transformation_net
        #print(input_points.shape)
        x = tf.expand_dims(input_points, -1)
        #print(x.shape)
        x = self.c1itn(x)
        #print(x.shape)
        x=tf.squeeze(x, axis=2)
        #print(x.shape)
        x = self.bn1itn(x, training=training)
        #print(x.shape)
        x = self.c2itn(x)
        x = self.bn2itn(x, training=training)
        #print(x.shape)
        x = self.c3itn(x)
        x = self.bn3itn(x, training=training)
        #print(x.shape)
        x = self.mp1itn(x)
        #print(x.shape)
        x = self.d1itn(x)
        x = self.bn4itn(x, training=training)
        #print(x.shape)
        x = self.d2itn(x)
        x = self.bn5itn(x, training=training)
        #print(x.shape)
        x = self.d3itn(x)
        #print(x.shape)
        input_T = self.r1itn(x)
        #print(input_points.shape)
        #print(input_T.shape)
        return input_T

    # @tf.function
    def __call__(self, obs, training):
        input_points = tf.dtypes.cast(obs, tf.float32)
        #print(input_points.shape)
        input_T = self.input_t_net(input_points, training=training)
        #print(input_T.shape)

        # forward net
        g = tf.matmul(input_points, input_T)
        #print("G", g.shape)
        g = self.c1g(g)
        g = self.bn1g(g, training=training)
        #print(g.shape)
        g = self.c2g(g)
        g = self.bn2g(g, training=training)
        #print("G", g.shape)

        #print(g.shape)
        feature_T = self.feature_t_net(g, training=training)
        #print(feature_T.shape)

        # forward net
        feature_T = tf.matmul(g, feature_T)
        # print("G", g.shape)
        g = self.c3g(feature_T)
        g = self.bn3g(g, training=training)
        #print(g.shape)
        g = self.c4g(g)
        g = self.bn4g(g, training=training)
        #print(g.shape)
        g = self.c5g(g)
        g = self.bn5g(g, training=training)
        #print(g.shape)

        # global_feature
        global_feature = self.mp1g(g)
        #print("D", global_feature.shape)
        #global_feature = self.d1g(global_feature)
        #global_feature = self.bn6g(global_feature)
        #global_feature = tf.squeeze(global_feature, axis=1)
        if self.check_numerics:
            global_feature = tf.debugging.check_numerics(global_feature, "global_feature")
        return input_T, feature_T, global_feature

    def reset(self):
        pass

    def get_vars(self, with_non_trainable=False):
        vars_ = []
        vars_.extend(self.bn1itn.trainable_weights)
        vars_.extend(self.bn2itn.trainable_weights)
        vars_.extend(self.bn3itn.trainable_weights)
        vars_.extend(self.bn4itn.trainable_weights)
        vars_.extend(self.bn5itn.trainable_weights)
        vars_.extend(self.bn1ftn.trainable_weights)
        vars_.extend(self.bn2ftn.trainable_weights)
        vars_.extend(self.bn3ftn.trainable_weights)
        vars_.extend(self.bn4ftn.trainable_weights)
        vars_.extend(self.bn5ftn.trainable_weights)
        vars_.extend(self.bn6ftn.trainable_weights)
        vars_.extend(self.bn1g.trainable_weights)
        vars_.extend(self.bn2g.trainable_weights)
        vars_.extend(self.bn3g.trainable_weights)
        vars_.extend(self.bn4g.trainable_weights)
        vars_.extend(self.bn5g.trainable_weights)
        if with_non_trainable:
            vars_.extend(self.bn1itn.non_trainable_weights)
            vars_.extend(self.bn2itn.non_trainable_weights)
            vars_.extend(self.bn3itn.non_trainable_weights)
            vars_.extend(self.bn4itn.non_trainable_weights)
            vars_.extend(self.bn5itn.non_trainable_weights)
            vars_.extend(self.bn1ftn.non_trainable_weights)
            vars_.extend(self.bn2ftn.non_trainable_weights)
            vars_.extend(self.bn3ftn.non_trainable_weights)
            vars_.extend(self.bn4ftn.non_trainable_weights)
            vars_.extend(self.bn5ftn.non_trainable_weights)
            vars_.extend(self.bn6ftn.non_trainable_weights)
            vars_.extend(self.bn1g.non_trainable_weights)
            vars_.extend(self.bn2g.non_trainable_weights)
            vars_.extend(self.bn3g.non_trainable_weights)
            vars_.extend(self.bn4g.non_trainable_weights)
            vars_.extend(self.bn5g.non_trainable_weights)
        if self.trainable:
            vars_.extend(self.c1itn.trainable_weights)
            vars_.extend(self.c2itn.trainable_weights)
            vars_.extend(self.c3itn.trainable_weights)
            vars_.extend(self.d1itn.trainable_weights)
            vars_.extend(self.d2itn.trainable_weights)
            vars_.extend(self.d3itn.trainable_weights)
            vars_.extend(self.c1ftn.trainable_weights)
            vars_.extend(self.c2ftn.trainable_weights)
            vars_.extend(self.c3ftn.trainable_weights)
            vars_.extend(self.d1ftn.trainable_weights)
            vars_.extend(self.d2ftn.trainable_weights)
            vars_.extend(self.d3ftn.trainable_weights)
            vars_.extend(self.c1g.trainable_weights)
            vars_.extend(self.c3g.trainable_weights)
            vars_.extend(self.c2g.trainable_weights)
            vars_.extend(self.c4g.trainable_weights)
            vars_.extend(self.c5g.trainable_weights)
        else:
            vars_.extend(self.c1itn.non_trainable_weights)
            vars_.extend(self.c2itn.non_trainable_weights)
            vars_.extend(self.c3itn.non_trainable_weights)
            vars_.extend(self.d1itn.non_trainable_weights)
            vars_.extend(self.d2itn.non_trainable_weights)
            vars_.extend(self.d3itn.non_trainable_weights)
            vars_.extend(self.c1ftn.non_trainable_weights)
            vars_.extend(self.c2ftn.non_trainable_weights)
            vars_.extend(self.c3ftn.non_trainable_weights)
            vars_.extend(self.d1ftn.non_trainable_weights)
            vars_.extend(self.d2ftn.non_trainable_weights)
            vars_.extend(self.d3ftn.non_trainable_weights)
            vars_.extend(self.c1g.non_trainable_weights)
            vars_.extend(self.c3g.non_trainable_weights)
            vars_.extend(self.c2g.non_trainable_weights)
            vars_.extend(self.c4g.non_trainable_weights)
            vars_.extend(self.c5g.non_trainable_weights)
        return vars_


class SemSegPointNet(BaseNet):
    def __init__(
            self,
            name,
            trainable=True,
            seed=None,
            check_numerics=False,
            initializer="glorot_uniform",
            n_points=4096,
            p_dim=3
            ):
        super().__init__(
            name=name,
            trainable=trainable,
            seed=seed,
            check_numerics=check_numerics,
            initializer=initializer)
        self.n_points = n_points
        self.p_dim = p_dim

        self.bn1g = BatchNormalization(name=name+"/bn1g")
        self.bn2g = BatchNormalization(name=name+"/bn2g")
        self.bn3g = BatchNormalization(name=name+"/bn3g")
        self.bn4g = BatchNormalization(name=name+"/bn4g")
        self.bn5g = BatchNormalization(name=name+"/bn5g")

        self.c1g = Conv2D(
            filters=64,
            kernel_size=[1, self.p_dim],
            activation="relu",
            name=name+"/c1g",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c2g = Conv1D(
            filters=64,
            kernel_size=1,
            activation="relu",
            name=name+"/c2g",
            trainable=trainable,
            kernel_initializer=initializer)

        self.c3g = Conv1D(
            filters=64,
            kernel_size=1,
            activation="relu",
            name=name+"/c3g",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c4g = Conv1D(
            filters=128,
            kernel_size=1,
            activation="relu",
            name=name+"/c4g",
            trainable=trainable,
            kernel_initializer=initializer)
        self.c5g = Conv1D(
            filters=1024,
            kernel_size=1,
            activation="relu",
            name=name+"/c5g",
            trainable=trainable,
            kernel_initializer=initializer)
        self.mp1g = MaxPool1D(name=name + "/mp1g", pool_size=self.n_points)


    # @tf.function
    def __call__(self, obs, training):
        input_points = tf.dtypes.cast(obs, tf.float32)
        #print(input_points.shape)
        x = tf.expand_dims(input_points, -1)
        g = self.c1g(x)
        g = tf.squeeze(g, axis=2)
        #print(g.shape)
        g = self.bn1g(g, training=training)
        g = self.c2g(g)
        g = self.bn2g(g, training=training)
        #print("G", g.shape)

        # print("G", g.shape)
        g = self.c3g(g)
        g = self.bn3g(g, training=training)
        #print(g.shape)
        g = self.c4g(g)
        g = self.bn4g(g, training=training)
        #print(g.shape)
        g = self.c5g(g)
        points_feat = self.bn5g(g, training=training)
        #print(g.shape)

        # global_feature
        global_feature = self.mp1g(g)
        #print("D", global_feature.shape)
        #global_feature = self.d1g(global_feature)
        #global_feature = self.bn6g(global_feature)
        #global_feature = tf.squeeze(global_feature, axis=1)
        if self.check_numerics:
            global_feature = tf.debugging.check_numerics(global_feature, "global_feature")
        return points_feat, global_feature

    def reset(self):
        pass

    def get_vars(self, with_non_trainable=False):
        vars_ = []
        vars_.extend(self.bn1g.trainable_weights)
        vars_.extend(self.bn2g.trainable_weights)
        vars_.extend(self.bn3g.trainable_weights)
        vars_.extend(self.bn4g.trainable_weights)
        vars_.extend(self.bn5g.trainable_weights)
        if with_non_trainable:
            vars_.extend(self.bn1g.non_trainable_weights)
            vars_.extend(self.bn2g.non_trainable_weights)
            vars_.extend(self.bn3g.non_trainable_weights)
            vars_.extend(self.bn4g.non_trainable_weights)
            vars_.extend(self.bn5g.non_trainable_weights)
        if self.trainable:
            vars_.extend(self.c1g.trainable_weights)
            vars_.extend(self.c3g.trainable_weights)
            vars_.extend(self.c2g.trainable_weights)
            vars_.extend(self.c4g.trainable_weights)
            vars_.extend(self.c5g.trainable_weights)
        else:
            vars_.extend(self.c1g.non_trainable_weights)
            vars_.extend(self.c3g.non_trainable_weights)
            vars_.extend(self.c2g.non_trainable_weights)
            vars_.extend(self.c4g.non_trainable_weights)
            vars_.extend(self.c5g.non_trainable_weights)
        return vars_