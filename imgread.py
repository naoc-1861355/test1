import numpy as np
import tensorflow as tf
import sklearn.datasets
import matplotlib.pyplot as plt
# import tensorflow_datasets as tfds
import scipy.linalg

from PIL import Image
import glob
'''
tfds.disable_progress_bar()
tf.enable_v2_behavior()
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

w = tf.Variable(0,dtype=tf.float32)
cost = tf.add(tf.add(w**2, tf.multiply(-10.,w)),25)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))
session.run(train)
print(session.run(w))
for i in range(1000):
    session.run(train)
print(session.run(w))'''
def load_dataset(is_plot=True):
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    if is_plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    plt.show()
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset

def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    (x_train, y_train), (x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train,x_test = x_train/255, x_test/255
    print(y_train.shape)
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset,test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = load_data_fashion_mnist(64)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(160, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)
    ])
    # from_logits is important to training error rate and test rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['sparse_categorical_accuracy'])
    model.fit(train_dataset, epochs=50)
    model.save('3NN_50_mnist_fashion.h5')
    print("save success")
    print("start test:")
    model.evaluate(test_dataset)
#train_iter, test_iter = load_data_fashion_mnist(32)
#from tensorflow.keras.datasets import mnist
#(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
'''
batch_size = 10

initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
loss = tf.keras.losses.MeanSquaredError()
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)

y_hat = tf.constant(36,name="y_hat")            #定义y_hat为固定值36
y = tf.constant(39,name="y")                    #定义y为固定值39

loss1 = tf.Variable((y-y_hat)**2,name="loss" )   #为损失函数创建一个变量

init = tf.global_variables_initializer()        #运行之后的初始化(ession.run(init))
with tf.Session() as session:                   #创建一个session并打印输出
    session.run(init)                           #初始化变量
    print(session.run(loss1))
    x = tf.placeholder(tf.int64,name="x")
    print(session.run(2 * x,feed_dict={x:3}))
    session.close()

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
'''
# x,y,a,b = load_dataset()
# print(x,y)