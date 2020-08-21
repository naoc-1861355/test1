import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import tensorflow.keras.backend as K

def proprocess(path):
    image_util = tf.keras.preprocessing.image
    img = image_util.load_img(path, target_size=(224, 224))
    img_tensor = image_util.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255
    return img_tensor


def setModel():
    model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=True, weights='imagenet')
    model.summary()
    return model


def viz_out(model, num, img_input):
    layers_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layers_outputs, name='activation')
    activations = activation_model.predict(img_input)
    activation = activations[3]
    for j in range(activation.shape[3]):
        plt.matshow(activation[0, :, :, j], cmap='viridis')
        plt.show()


def generate_pattern(model, layer_name, input=np.random.random((1, 225, 225, 3))):
    layer = model.get_layer(layer_name)
    output = layer(input)
    plt.matshow(output[0, :, :, 30], cmap='viridis')
    plt.show()


def heat_map(path, model):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))

    # `x` is a float32 Numpy array of shape (224, 224, 3)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=5))


    african_elephant_output = model.output[:, 386]
    last_conv_layer = model.get_layer('block5_conv3')
    iterate = K.function([model.input], [last_conv_layer.output[0]])
    conv_layer_output_value = iterate([x])

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #heatmap = np.expand_dims(heatmap, axis=-1)
    plt.matshow(heatmap[0])
    plt.show()
    return heatmap[0]


def show_heatmap(heatmap, path):
    import cv2
    img = cv2.imread(path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # 0.3 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.3 + img
    print(superimposed_img)
    # Save the image to disk
    cv2.imwrite('23.jpg', superimposed_img)
    result = cv2.imread('23.jpg')
    cv2.imshow('hmap', result[:,:,::-1])
    cv2.waitKey(0)


def main():
    img_tensor = proprocess("/Users/ezxr.sxxianchengze/Desktop/Linnaeus 5 64X64/train/dog/20_64.jpg")
    model = setModel()
    hmap = heat_map("/Users/ezxr.sxxianchengze/Desktop/0002.jpg", model)
    #how_heatmap(hmap, "/Users/ezxr.sxxianchengze/Desktop/creative_commons_elephant.jpg")
    #generate_pattern(model, 'block1_conv1', img_tensor)
    # vizOutput(mobilenet, 8, img_tensor)


if __name__ == '__main__':
    main()
