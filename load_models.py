import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.layers import  Concatenate, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.applications import VGG16
from lukinoising4GP import lukinoising


def cnn_vgg_model(input_shape):
    vgg16 = VGG16(include_top=False, input_shape=input_shape)
    vgg16.trainable = True
    
    model = Sequential()
    model.add(vgg16)
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))
    
    return model


def encoder_block(inputs, num_filters):
    # Convolution with 3x3 filter followed by ReLU activation
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = Activation('relu')(x)

    # Convolution with 3x3 filter followed by ReLU activation
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = Activation('relu')(x)

    # Max Pooling with 2x2 filter
    p = MaxPool2D(pool_size=(2, 2))(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    # Upsampling with 2x2 filter
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)

    # Ensure the shapes match by cropping or padding
    if x.shape[1] != skip_features.shape[1] or x.shape[2] != skip_features.shape[2]:
        x = ZeroPadding2D(((0, skip_features.shape[1] - x.shape[1]), (0, skip_features.shape[2] - x.shape[2])))(x)

    # Concatenate skip connections
    x = Concatenate()([x, skip_features])

    # Convolution with 3x3 filter followed by ReLU activation
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = Activation('relu')(x)

    # Convolution with 3x3 filter followed by ReLU activation
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = Activation('relu')(x)
    return x


def unet_model(input_shape=(225, 300, 3), num_classes=1):
    inputs = Input(input_shape)

    # Contracting Path
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b1 = Conv2D(1024, 3, padding='same')(p4)
    b1 = Activation('relu')(b1)
    b1 = Conv2D(1024, 3, padding='same')(b1)
    b1 = Activation('relu')(b1)

    # Expansive Path
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output Layer
    outputs = Conv2D(num_classes, 1, activation='sigmoid' if num_classes == 1 else 'softmax')(d4)

    model = Model(inputs, outputs, name='U-Net')
    return model


def load_image(filepath: str, resize_shape: tuple) -> np.ndarray:
    """
    Load image from file
    :param filepath: path to image file
    :return: image as numpy array
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = lukinoising(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, resize_shape)

    return img


def draw_predicted_bounding_box(model: Model, img: np.ndarray, plot: bool = True):
    """
    Side by side plot comparing original bounding box and the predicted one,
    by resizing the model output to original image size and superimposing
    :param model: regression model
    :param img: original image
    :param bounding_box: origina bounding box
    :param plot: whether to plot the image
    :return: mean squared error
    """
    img = img.copy()
    img = img / 255.0
    rescaling_arr = np.array(img.shape[:2] * 2)

    pred = model.predict(np.array([img]))[0]
    pred *= rescaling_arr
    pred = np.int32(pred)

    img = (img * 255).astype(np.uint8)
    img_with_pred_bb = cv2.rectangle(img.copy(), (int(pred[1]), int(pred[0])), (int(pred[3]), int(pred[2])),
                                     (0, 255, 0), 2)

    if plot:
        plt.imshow(img_with_pred_bb)
        plt.axis('off')
        plt.title('Predicted bounding box')
        plt.show()

    return img_with_pred_bb


def draw_predicted_mask(model: Model, img: np.ndarray, plot: bool = True) -> np.ndarray:
    """
    Predict mask and superimpose on original image
    :param model: segmentation model
    :param img: example image
    :param plot: indicates whether to plot the image
    :return: image with predicted mask
    """
    img = img.copy()
    img = img / 255.0
    pred = model.predict(np.array([img]))[0]
    pred = np.argmax(pred, axis=-1)

    img = (img * 255).astype(np.uint8)

    print(np.sum(pred > 0.))


    if plot:
        plt.imshow(img)
        plt.imshow(pred, alpha=0.2, cmap='Blues', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('Predicted mask')
        plt.show()

    return pred


def main():
    resize_shape = (300, 225)
    input_shape = (225, 300, 3)
    ex_path = 'dataset/Set1-Training&Validation Sets CNN/Standard/26.png'
    img = load_image(ex_path, resize_shape)

    # Test lukimodel with example image
    lukimodel = cnn_vgg_model(input_shape)
    lukimodel.load_weights("vgg16_bb_fetus.weights.h5")
    draw_predicted_bounding_box(lukimodel, img)

    # Test ivomodel with example image
    ivomodel = unet_model(input_shape)
    ivomodel.load_weights("unet_segNT_fetus.weights.h5")
    draw_predicted_mask(ivomodel, img)


if __name__ == "__main__":
    main()
