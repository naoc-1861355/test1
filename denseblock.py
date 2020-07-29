import tensorflow as tf



class DenseConv(tf.keras.layers.Layer):
    def __init__(self,channel):
        super(DenseConv,self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(channel,(3,3),padding='same')
        self.relu = tf.keras.layers.ReLU()
        self.bn = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        x = self.conv2d(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return tf.keras.layers.concatenate([x,inputs])


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self,channel,num_convs):
        super(DenseBlock,self).__init__()
        self.listlayer = []
        for _ in num_convs:
            self.listlayer.append(DenseConv(channel))

    def call(self, x, **kwargs):
        for layer in self.listlayer:
            x = layer(x)
        return x
