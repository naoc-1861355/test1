import glob as glob
from PIL import Image
import numpy as np
import tensorflow as tf


def load_data_from_path():
    train = glob.glob("/Users/ezxr.sxxianchengze/Desktop/Linnaeus 5 64X64/train/*/*.jpg")
    x_train = np.array([np.array(Image.open(img)) for img in train])
    y_train = np.array([np.repeat(i,1200) for i in range(5)]).flatten()
    test = glob.glob("/Users/ezxr.sxxianchengze/Desktop/Linnaeus 5 64X64/test/*/*.jpg")
    x_test = np.array([np.array(Image.open(img)) for img in test])
    y_test = np.array([np.repeat(i, 400) for i in range(5)]).flatten()
    return (x_train,y_train),(x_test,y_test)

if __name__ == '__main__':
    # load_data_from_path()
    mobilenet = tf.keras.applications.MobileNet(input_shape=(224,224,3), include_top=False, weights='imagenet')
    mobilenet.summary()
    op = mobilenet.optimizer
    print(op,op.__class__)
    x = mobilenet.layers[-2].output
    customize = tf.keras.models.Model(inputs=mobilenet.input, outputs=x,name='???')
    customize.summary()
    #tf.keras.utils.plot_model(resnet50,'resnet50.png',show_shapes=True)