import glob as glob
from PIL import Image
import numpy as np
import tensorflow as tf


'''
    Returns: (train_x,train_y),(test_x,test_y) in np.ndarray, shape(num,resolution,resolution, rbg_channel)
'''
def load_data_from_path():
    train = glob.glob("/Users/ezxr.sxxianchengze/Desktop/Linnaeus 5 64X64/train/*/*.jpg")
    x_train = np.array([np.array(Image.open(img)) for img in train])/255
    y_train = np.array([np.repeat(i,1200) for i in range(5)]).flatten()
    print(x_train[0,0,0])
    test = glob.glob("/Users/ezxr.sxxianchengze/Desktop/Linnaeus 5 64X64/test/*/*.jpg")
    x_test = np.array([np.array(Image.open(img)) for img in test])/255
    y_test = np.array([np.repeat(i, 400) for i in range(5)]).flatten()
    return (x_train,y_train),(x_test,y_test)

if __name__ == '__main__':
    # load_data_from_path()
    mobilenet = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    mobilenet.summary()
    x = mobilenet.layers[-2].output
    customize = tf.keras.models.Model(inputs=mobilenet.input, outputs=x,name='???')
    customize.summary()
    tf.keras.utils.plot_model(mobilenet,'mobilev2.png',show_shapes=True)
    inception = tf.keras.applications.InceptionV3(input_shape = (299,299,3), include_top=False, weights='imagenet')
    #tf.keras.utils.plot_model(inception,'inceptionv3.png',show_shapes=True)