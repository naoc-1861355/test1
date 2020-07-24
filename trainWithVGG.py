import tensorflow as tf
import time
from imgprocess import load_data_from_path
from CNNmodel_lenet import decay
import matplotlib.pyplot as plt
def train(model):
    start1 = time.time()
    model.fit(train_dataset, validation_data=test_dataset, epochs=30, callbacks=[learning_rate_decay])
    # model.save('lenet-5-dropout.h5')
    model.evaluate(test_dataset)
    end1 = time.time()
    print(end1 - start1)

if __name__ == '__main__':
    batch_size = 32
    train_dataset, test_dataset = load_data_from_path()
    input_shape = train_dataset[0].shape[1:]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(6000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).batch(batch_size)
    vgg = tf.keras.applications.VGG16(input_shape=(64,64,3), include_top=False, weights='imagenet')
    vgg.trainable = False
    # vgg.summary()
    model= tf.keras.models.Sequential([vgg,
                                       tf.keras.layers.GlobalAveragePooling2D(),
                                       tf.keras.layers.Dense(100,activation='relu',kernel_regularizer=tf.keras.regularizers.l2()),
                                       tf.keras.layers.Dense(5)])
    learning_rate_decay = tf.keras.callbacks.LearningRateScheduler(decay, verbose=1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    model.summary()
    #train(model)
    inspect = tf.keras.models.Model(inputs=vgg.input,outputs=vgg.layers[-2].output)
    inspect.summary()
