import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QTimer
import cv2
from PIL import Image, ImageQt
from lukinoising4GP import lukinoising
from load_models import cnn_vgg_model, load_image, draw_predicted_bounding_box

# Load model
EXAMPLE_IMG = "dataset/Set1-Training&Validation Sets CNN/Standard/25.png"
MODEL_PATH = "model.h5"
MODEL_WEIGHTS = "model.weights.h5"
IMAGE_SHAPE = (450, 600, 3)

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
        self.model = cnn_vgg_model((450, 600, 3))
        self.model.load_weights(MODEL_WEIGHTS)

        # Image label
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        self.image = None
        self.image_grayscale = None
        self.image_denoised = None
        self.image_predicted = None
        self.load_image(EXAMPLE_IMG)


    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff)")
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        img = load_image(file_path)
        self.image = img.copy()
        self.image_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.image_denoised = lukinoising(self.image_grayscale.copy())
        self.image_predicted = draw_predicted_bounding_box(self.model, self.image, plot=False)
        self.loop_image_processing()

    def loop_image_processing(self):
        """
        Loop through the image processing steps
        """
        # Show the original image
        self.plot_image(self.image)

        # Set a timer to show the denoised image after 2 seconds
        QTimer.singleShot(2000, lambda: self.plot_image(self.image_denoised))

        # Set a timer to show the predicted bounding box after 4 seconds
        QTimer.singleShot(4000, lambda: self.plot_image(self.image_predicted))

    def plot_image(self, array):
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
