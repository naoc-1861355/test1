import tensorflow as tf
import numpy as np
new_model_1 = tf.keras.models.load_model('lenet-5.h5')
new_model_1.summary()
new_model_2 = tf.keras.models.load_model('lenet-5-2.h5')
new_model_2.summary()
new_model_3 = tf.keras.models.load_model('lenet-5-dropout.h5')
new_model_3.summary()
y = new_model_3(np.zeros((1,28,28,1)).astype('float32'))
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    y = sess.run(y)
    print(y)
for i in range(4):
    new_model_3.layers[i].trainable = False
my_model = tf.keras.models.Sequential([new_model_3.layers[i] for i in range(4)],name='my_model')
my_model.add(tf.keras.layers.Flatten())
my_model.add(tf.keras.layers.Dense(10))
my_model.summary()
nnew_model = tf.keras.models.load_model('3NN_50_mnist_fashion_4.h5')
nnew_model.summary()
'''
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same', input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10)])

model.load_weights('lenet-5.h5')
print("load params success")
print(model.layers[2].weights[0])
print(model.get_weights())
'''
