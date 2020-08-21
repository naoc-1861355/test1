import tensorflow as tf

import pickle

class DenseConv(tf.keras.layers.Layer):
    def __init__(self,channel):
        super(DenseConv,self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(channel,(3,3),padding='same')
        self.relu = tf.keras.layers.ReLU()
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        x = self.bn(inputs)
        x = self.relu(x)
        x = self.conv2d(x)
        return tf.keras.layers.concatenate([x,inputs],axis=-1)


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, channel):
        super(DenseBlock,self).__init__()
        self.listlayer = []
        for _ in range(num_convs):
            self.listlayer.append(DenseConv(channel))

    def call(self, x, **kwargs):
        for layer in self.listlayer:
            x = layer(x)
        return x

blk = DenseBlock(2, 10)
X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
print(Y.shape)
