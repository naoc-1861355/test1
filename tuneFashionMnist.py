import tensorflow as tf
from imgread import load_data_fashion_mnist
import tensorflow_datasets as tfds
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
train,test = load_data_fashion_mnist(64)
print("start test:")
new_model = tf.keras.models.load_model('models/easy_model_mnist_fashion.h5')
new_model.summary()
new_model.evaluate(test)
new_model.evaluate(train)

nnew_model = tf.keras.models.load_model('models/3NN_50_mnist_fashion.h5')
nnew_model.summary()
nnew_model.evaluate(test)
nnew_model.evaluate(train)

