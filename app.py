import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QSlider, QWidget, QGridLayout, QInputDialog
from PyQt5.QtGui import QPixmap, QImage, QImageReader
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QSize

import numpy as np


class ImageSegmentationApp(QMainWindow):
    def __init__(self):
        super(ImageSegmentationApp, self).__init__()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.image_path = None
        self.original_image = None
        self.display_image = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        open_button = QPushButton("Open Image", self)
        open_button.clicked.connect(self.open_image)

        grayscale_button = QPushButton("Convert to Grayscale", self)
        grayscale_button.clicked.connect(self.convert_to_grayscale)

        
        point_detection_button = QPushButton("Point Detection", self)
        point_detection_button.clicked.connect(self.apply_point_detection)


        horizontal_line_detection_button = QPushButton("Horizontal Line Detection", self)
        horizontal_line_detection_button.clicked.connect(self.apply_horizontal_line_detection)

        vertical_line_detection= QPushButton("Vertical Line Detection",self)
        vertical_line_detection.clicked.connect(self.vertical_line_detection)


        a45p_line_detection= QPushButton("+45 Line Detection",self)
        a45p_line_detection.clicked.connect(self.a45p_line_detection1)


        a45n_line_detection= QPushButton("-45 Line Detection",self)
        a45n_line_detection.clicked.connect(self.a45n_line_detection1)
        


        laplacian_log= QPushButton("Laplacian of Gaussian (log)",self)
        laplacian_log.clicked.connect(self.laplacian_log)

        user_defined_filter_button = QPushButton("User-Defined Filter", self)
        user_defined_filter_button.clicked.connect(self.user_defined_filter)

        save_button = QPushButton("Save Image", self)
        save_button.clicked.connect(self.save_image)

        layout.addWidget(open_button)
        layout.addWidget(grayscale_button)
        layout.addWidget(point_detection_button)
        layout.addWidget(horizontal_line_detection_button)
        layout.addWidget(vertical_line_detection)
        layout.addWidget(a45p_line_detection)
        layout.addWidget(a45n_line_detection)
        layout.addWidget(laplacian_log)
        layout.addWidget(user_defined_filter_button)
        layout.addWidget(save_button)
        layout.addWidget(self.image_label)

        central_widget.setLayout(layout)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Image Segmentation App")

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setOptions(options)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]

            if QImageReader(file_path).size() != QSize(0, 0):
                self.image_path = file_path
                self.original_image = cv2.imread(file_path)
                self.display_image = self.original_image.copy()
                self.update_image_label()
            else:
                print("Invalid image file.")

    def convert_to_grayscale(self):
        if self.original_image is not None:
            self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.update_image_label()
    

    def apply_horizontal_line_detection(self):
        if self.original_image is not None:
            kernel = np.array([[-1, -1, -1],
                               [2,  2,  2],
                               [-1, -1, -1]], dtype=np.float32)

            self.display_image = cv2.filter2D(self.original_image, -1, kernel)
            self.update_image_label()


    def vertical_line_detection(self):
        if self.original_image is not None:
            kernel = np.array([[-1, 2, -1],
                               [-1, 2, -1],
                               [-1, 2, -1]], dtype=np.float32)

            self.display_image = cv2.filter2D(self.original_image, -1, kernel)
            self.update_image_label()



    def a45p_line_detection1(self):
        if self.original_image is not None:
            kernel = np.array([[-1, -1, 2],
                               [-1, 2, -1],
                               [2, -1, -1]], dtype=np.float32)

            self.display_image = cv2.filter2D(self.original_image, -1, kernel)
            self.update_image_label()


    def a45n_line_detection1(self):
        if self.original_image is not None:
            kernel = np.array([[2, -1, -1],
                               [-1, 2, -1],
                               [-1, -1, 2]], dtype=np.float32)

            self.display_image = cv2.filter2D(self.original_image, -1, kernel)
            self.update_image_label()  


    def laplacian_log(self):
        if self.original_image is not None:
            kernel = np.array([[0,0, -1,0,0],
                               [0,-1,-2,-1,0],
                               [-1,-2,16,-2,-1],
                               [0,-1,-2,-1,0],
                               [0,0,-1,0,0]], dtype=np.float32)

            self.display_image = cv2.filter2D(self.original_image, -1, kernel)
            self.update_image_label()  
    
    def apply_point_detection(self):
        if self.original_image is not None:
            kernel = np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=np.float32)

            self.display_image = cv2.filter2D(self.original_image, -1, kernel)
            self.update_image_label()

    def user_defined_filter(self):
      if self.original_image is not None:
        size, ok = QInputDialog.getInt(self, "Filter Size", "Enter filter size:", 3, 1, 11, 2)
        if ok:
            coefficients_str, ok = QInputDialog.getText(self, "Filter Coefficients", "Enter filter coefficients (comma-separated):")
            if ok:
                coefficients = coefficients_str.split(',')
                
                try:
                    coefficients = [float(x.strip()) for x in coefficients]
                except ValueError:
                    print("Invalid coefficients. Please enter valid numeric values.")
                    return

                expected_size = size * size
                if len(coefficients) != expected_size:
                    print(f"Invalid number of coefficients. Expected {expected_size}, got {len(coefficients)}")
                    return

                coefficients = np.array(coefficients).reshape((size, size))

                self.display_image = cv2.filter2D(self.original_image, -1, coefficients)
                self.update_image_label()


    def save_image(self):
        if self.display_image is not None:
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.AnyFile)
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)

            if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()[0]
                cv2.imwrite(file_path, self.display_image)
                print(f"Image saved to {file_path}")

    def update_image_label(self):
         if self.display_image is not None:
          if len(self.display_image.shape) == 3:  # Color image
            height, width, channel = self.display_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
          elif len(self.display_image.shape) == 2:  # Grayscale image
            height, width = self.display_image.shape
            bytes_per_line = width
            q_image = QImage(self.display_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

          pixmap = QPixmap.fromImage(q_image)
          self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ImageSegmentationApp()
    main_window.show()
    sys.exit(app.exec_())
