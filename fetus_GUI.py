import sys
from tensorflow.keras.models import load_model
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QSizePolicy, QGridLayout
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QTimer
import cv2
from PIL import Image, ImageQt
from lukinoising4GP import lukinoising
from load_models import cnn_vgg_model, load_image, draw_predicted_bounding_box, draw_predicted_mask

# Load model
EXAMPLE_IMG = "dataset/Set1-Training&Validation Sets CNN/Standard/25.png"
MODEL_WEIGHTS = "vgg16_bb_fetus.weights.h5"
IVOMODEL_PATH = "model-unet-model.keras"
IMAGE_SHAPE = (225, 300, 3)
RESIZE_SHAPE = (300, 225)


class ImageBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Browser")

        # General layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Browse button
        self.browse_button = QPushButton("Select an image...")
        self.browse_button.clicked.connect(self.browse_file)
        self.layout.addWidget(self.browse_button)

        # Load model
        self.model = cnn_vgg_model(IMAGE_SHAPE)
        self.model.load_weights(MODEL_WEIGHTS)
        self.segmenter = load_model(IVOMODEL_PATH)

        # Image label
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(True)
        self.layout.addWidget(self.image_label)
        self.raw = None
        self.processed_image = None
        self.img_w_bounding_box = None
        self.img_w_segmentation = None
        self.load_image(EXAMPLE_IMG)

        # Add 4 buttons in a 2x2 grid layout: raw, preprocessed, bounding box, segmentation 
        self.button_layout = QGridLayout()
        self.layout.addLayout(self.button_layout)

        self.show_original_button = QPushButton("Original")
        self.show_original_button.clicked.connect(lambda: self.plot_image(self.raw))
        self.show_original_button.setStyleSheet("background-color: red")
        self.button_layout.addWidget(self.show_original_button, 0, 0)

        self.show_preprocessed_button = QPushButton("Preprocessed")
        self.show_preprocessed_button.clicked.connect(lambda: self.plot_image(self.processed_image))
        self.show_preprocessed_button.setStyleSheet("background-color: purple")
        self.button_layout.addWidget(self.show_preprocessed_button, 0, 1)

        self.show_bounding_box_button = QPushButton("Bounding Box")
        self.show_bounding_box_button.clicked.connect(lambda: self.plot_image(self.img_w_bounding_box))
        self.show_bounding_box_button.setStyleSheet("background-color: green")
        self.button_layout.addWidget(self.show_bounding_box_button, 1, 0)

        self.show_segmentation_button = QPushButton("Segmentation")
        self.show_segmentation_button.clicked.connect(lambda: self.plot_image(self.img_w_segmentation))
        self.show_segmentation_button.setStyleSheet("background-color: cyan")
        self.button_layout.addWidget(self.show_segmentation_button, 1, 1)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, caption="Open Image File", dir="./dataset",
                                                   filter="Image Files (*.jpg *.jpeg *.png *.bmp *.tiff)")
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        raw, img = load_image(file_path, resize_shape=RESIZE_SHAPE)
        self.raw = raw.copy()
        self.processed_image = img.copy()
        self.img_w_bounding_box = draw_predicted_bounding_box(self.model, self.processed_image, plot=False)
        self.img_w_segmentation = draw_predicted_mask(self.segmenter, self.processed_image, plot=False)
        self.plot_image(self.raw)

    def plot_image(self, array):
        """
        Given an image as an array, display it in the GUI
        """
        img = Image.fromarray(array)
        qt_img = ImageQt.ImageQt(img)
        pixmap = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pixmap)
        return pixmap


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageBrowser()
    window.show()
    sys.exit(app.exec())
