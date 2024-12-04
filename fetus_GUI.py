import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QTimer
import cv2
from PIL import Image, ImageQt
from lukinoising4GP import lukinoising
from create_model import create_model, load_image, draw_prediction

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
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        self.layout.addWidget(self.browse_button)

        # Image label
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        self.image = None
        self.image_grayscale = None
        self.load_image(EXAMPLE_IMG)

        # Load model
        self.model = create_model((450, 600, 3))
        self.model.load_weights(MODEL_WEIGHTS)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff)")
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        img = load_image(file_path)
        self.image = img.copy()
        self.image_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.plot_image(img)

    def loop_image_processing(self):
        # Set a timer to show the denoised image after 2 seconds
        QTimer.singleShot(2000, self.denoise_image)

        # Set a timer to show the predicted bounding box after 4 seconds
        QTimer.singleShot(2000, self.draw_prediction)

    def denoise_image(self):
        img_denoise = lukinoising(self.image_grayscale)
        self.plot_image(img_denoise)

    def draw_prediction(self):
        img_with_bb = draw_prediction(self.model, self.image, plot=False)
        self.plot_image(img_with_bb)

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
