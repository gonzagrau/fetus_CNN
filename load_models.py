import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import VGG16


def create_model(input_shape):
    """
    VGG16 based model for bounding box regression
    :param input_shape: image shape
    :return:
    """
    vgg16 = VGG16(include_top=False, input_shape=input_shape)
    vgg16.trainable = False

    model = Sequential()
    model.add(vgg16)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))

    return model

def load_image(filepath: str) -> np.ndarray:
    """
    Load image from file
    :param filepath: path to image file
    :return: image as numpy array
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (600, 450))

    return img

def draw_prediction(model: Model, img: np.ndarray, plot: bool=True):
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


def main():
    input_shape = (450, 600, 3)
    model = create_model(input_shape)
    weights = "model.weights.h5"
    model.load_weights(weights)
    print(model.summary())

    # Test model with example image
    ex_path = 'dataset/Set1-Training&Validation Sets CNN/Standard/25.png'
    img = load_image(ex_path)
    draw_prediction(model, img)

    # Save model
    model.save("model.h5")


if __name__ == "__main__":
    main()


