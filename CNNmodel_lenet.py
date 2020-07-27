import tensorflow as tf
import time
from imgprocess import load_data_from_path

def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255, x_test / 255
    x_train, x_test = x_train.reshape((-1, 28, 28, 1)), x_test.reshape((-1, 28, 28, 1))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_train.shape)
    print(y_train.shape)
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset


def decay(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * (0.1 ** (epoch // 10))


if __name__ == '__main__':
    #train_dataset, test_dataset = load_data_fashion_mnist(64)
    batch_size = 32
    train_dataset, test_dataset = load_data_from_path()
    input_shape = train_dataset[0].shape[1:]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(6000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).batch(batch_size)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu',
                               padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=5,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5)], name='lenet-new.h5')
    learning_rate_decay = tf.keras.callbacks.LearningRateScheduler(decay, verbose=1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    model.summary()
    start = time.time()
    model.fit(train_dataset,validation_data=test_dataset, epochs=50, callbacks=[learning_rate_decay])
    #model.save('nett.h5')
    model.evaluate(test_dataset)
    end = time.time()
    print(end-start)
