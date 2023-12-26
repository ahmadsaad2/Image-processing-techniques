import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QSlider, QWidget, QGridLayout, QInputDialog
from PyQt5.QtGui import QPixmap, QImage, QImageReader
import cv2
import scipy

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
        open_button.setFixedWidth(250)
        open_button.clicked.connect(self.open_image)
        self.apply_common_style(open_button)



        grayscale_button = QPushButton("Convert to Grayscale", self)
        grayscale_button.clicked.connect(self.convert_to_grayscale)
        grayscale_button.setFixedWidth(250)
        self.apply_common_style(grayscale_button)

        

        
        point_detection_button = QPushButton("Point Detection", self)
        point_detection_button.setFixedWidth(250)
        self.apply_common_style(point_detection_button)
        point_detection_button.clicked.connect(self.apply_point_detection)

        horizontal_edge_detection_button = QPushButton("Horizontal Edge Detection(sobel)", self)
        horizontal_edge_detection_button.clicked.connect(self.apply_horizontal_edge_detection)
        horizontal_edge_detection_button.setFixedWidth(250)
        self.apply_common_style(horizontal_edge_detection_button)

        
        horizontal_line_detection_button = QPushButton("Horizontal Line Detection", self)
        horizontal_line_detection_button.clicked.connect(self.apply_horizontal_line_detection)
        horizontal_line_detection_button.setFixedWidth(250)
        self.apply_common_style(horizontal_line_detection_button)

        vertical_line_detection= QPushButton("Vertical Line Detection",self)
        vertical_line_detection.clicked.connect(self.vertical_line_detection)
        vertical_line_detection.setFixedWidth(250)
        self.apply_common_style(vertical_line_detection)

        vertical_edge_detection_button = QPushButton("Vertical Edge Detection (sobel)", self)
        vertical_edge_detection_button.clicked.connect(self.apply_vertical_edge_detection)
        vertical_edge_detection_button.setFixedWidth(250)
        self.apply_common_style(vertical_edge_detection_button)
        

        a45p_line_detection= QPushButton("+45 Line Detection",self)
        a45p_line_detection.clicked.connect(self.a45p_line_detection1)
        a45p_line_detection.setFixedWidth(250)
        self.apply_common_style(a45p_line_detection)


        a45n_line_detection= QPushButton("-45 Line Detection",self)
        a45n_line_detection.clicked.connect(self.a45n_line_detection1)
        a45n_line_detection.setFixedWidth(250)
        self.apply_common_style(a45n_line_detection)

        laplacian_button = QPushButton("Laplacian Filter", self)
        laplacian_button.clicked.connect(self.apply_laplacian_filter)
        laplacian_button.setFixedWidth(250)
        self.apply_common_style(laplacian_button)

        plus_45_edge_detection_button = QPushButton("+45 Edge Detection (sobel) ", self)
        plus_45_edge_detection_button.clicked.connect(self.apply_plus_45_edge_detection)
        plus_45_edge_detection_button.setFixedWidth(250)
        self.apply_common_style(plus_45_edge_detection_button)


        minus_45_edge_detection_button = QPushButton("-45 Edge Detection (sobel)", self)
        minus_45_edge_detection_button.clicked.connect(self.apply_minus_45_edge_detection)
        minus_45_edge_detection_button.setFixedWidth(250)
        self.apply_common_style(minus_45_edge_detection_button)
       
        laplacian_log= QPushButton("Laplacian of Gaussian (log)",self)
        laplacian_log.clicked.connect(self.laplacian_log)
        laplacian_log.setFixedWidth(250)
        self.apply_common_style(laplacian_log)

        zero_crossing_button = QPushButton("Zero Crossing", self)
        zero_crossing_button.clicked.connect(self.zero_crossing_button_clicked)
        zero_crossing_button.setFixedWidth(250)
        self.apply_common_style(zero_crossing_button)

        threshold_button = QPushButton("Apply Threshold", self)
        threshold_button.clicked.connect(self.apply_threshold)
        threshold_button.setFixedWidth(250)
        self.apply_common_style(threshold_button)

        adaptive_threshold_button = QPushButton("Adaptive Threshold", self)
        adaptive_threshold_button.clicked.connect(self.apply_adaptive_threshold)
        adaptive_threshold_button.setFixedWidth(250)
        self.apply_common_style(adaptive_threshold_button)

        user_defined_filter_button = QPushButton("User-Defined Filter", self)
        user_defined_filter_button.clicked.connect(self.user_defined_filter)
        user_defined_filter_button.setFixedWidth(250)
        self.apply_common_style(user_defined_filter_button)

        save_button = QPushButton("Save Image", self)
        save_button.clicked.connect(self.save_image)
        save_button.setFixedWidth(250)
        self.apply_common_style(save_button)

        layout.addWidget(open_button)
        layout.addWidget(grayscale_button)
        layout.addWidget(point_detection_button)
        layout.addWidget(horizontal_edge_detection_button)
        layout.addWidget(horizontal_line_detection_button)
        layout.addWidget(vertical_edge_detection_button)
        layout.addWidget(vertical_line_detection)
        layout.addWidget(a45p_line_detection)
        layout.addWidget(a45n_line_detection)
        layout.addWidget(plus_45_edge_detection_button)
        layout.addWidget(minus_45_edge_detection_button)
        layout.addWidget(laplacian_button)
        layout.addWidget(laplacian_log)
        layout.addWidget(zero_crossing_button)
        layout.addWidget(threshold_button)
        layout.addWidget(adaptive_threshold_button)
        layout.addWidget(user_defined_filter_button)
        layout.addWidget(save_button)
        layout.addWidget(self.image_label)

        central_widget.setLayout(layout)

        self.setGeometry(300, 300, 800, 3000)
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
    
    def apply_horizontal_edge_detection(self):
        if self.original_image is not None:
            horizontal_edge_filter = np.array([[-1, -2, -1],
                                           [0, 0, 0],
                                           [1, 2, 1]])

        self.display_image = cv2.filter2D(self.original_image, -1, horizontal_edge_filter)
        self.update_image_label()

    def apply_horizontal_line_detection(self):
        if self.original_image is not None:
            kernel = np.array([[-1, -1, -1],
                               [2,  2,  2],
                               [-1, -1, -1]], dtype=np.float32)

            self.display_image = cv2.filter2D(self.original_image, -1, kernel)
            self.update_image_label()

    def apply_adaptive_threshold(self):
        if self.original_image is not None:
            grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        block_size, ok1 = QInputDialog.getInt(self, "Adaptive Threshold", "Enter block size (must be odd):", 11, 3, 101, 2)
        if not ok1:
            return  

        if block_size % 2 == 0:
            print("Block size must be an odd number. Adding 1 to make it odd.")
            block_size += 1

        constant_value, ok2 = QInputDialog.getDouble(self, "Adaptive Threshold", "Enter constant value:", 2, 0, 255, 1)
        if not ok2:
            return  

        
        adaptive_threshold_img = cv2.adaptiveThreshold(
            grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant_value
        )

        self.display_image = adaptive_threshold_img
        self.update_image_label()


        self.display_image = adaptive_threshold_img
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
    
    def apply_vertical_edge_detection(self):
        if self.original_image is not None:
            vertical_edge_filter = np.array([[-1, 0, 1],
                                         [-2, 0, 2],
                                         [-1, 0, 1]])

        self.display_image = cv2.filter2D(self.original_image, -1, vertical_edge_filter)
        self.update_image_label()

    def apply_plus_45_edge_detection(self):
        if self.original_image is not None:
            plus_45_edge_filter = np.array([[-2, -1, 0],
                                        [-1,  0, 1],
                                        [0, 1, 2]])

        self.display_image = cv2.filter2D(self.original_image, -1, plus_45_edge_filter)
        self.update_image_label()

    def apply_minus_45_edge_detection(self):
        if self.original_image is not None:
            minus_45_edge_filter = np.array([[0, 1, 2],
                                         [-1,  0, 1],
                                         [-2, -1,  0]])

        self.display_image = cv2.filter2D(self.original_image, -1, minus_45_edge_filter)
        self.update_image_label()


    def zero_crossing_button_clicked(self):
        if self.original_image is not None:
            grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            log_filtered = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
            log_filtered = cv2.Laplacian(log_filtered, cv2.CV_64F)

            zero_crossing_img = self.zero_crossing_detection(log_filtered)

            self.display_image = zero_crossing_img
            self.update_image_label()

    def zero_crossing_detection(self, image):
        rows, cols = image.shape
        zero_crossing_img = np.zeros((rows, cols), dtype=np.uint8)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                neighbors = [image[i - 1, j], image[i + 1, j], image[i, j - 1], image[i, j + 1],
                             image[i - 1, j - 1], image[i - 1, j + 1], image[i + 1, j - 1], image[i + 1, j + 1]]
                if np.prod(np.sign(neighbors)) < 0:
                    zero_crossing_img[i, j] = 255

        return zero_crossing_img

    def apply_laplacian_filter(self):
        if self.original_image is not None:
            grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            laplacian_filtered = cv2.Laplacian(grayscale_image, cv2.CV_64F)

            # Normalize the result to display as an image
            laplacian_filtered = cv2.normalize(laplacian_filtered, None, 0, 255, cv2.NORM_MINMAX)

            self.display_image = laplacian_filtered.astype(np.uint8)
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
    

    def apply_threshold(self):
        if self.original_image is not None:
            grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            threshold_value, ok = QInputDialog.getInt(self, "Threshold", "Enter the threshold value:", 120, 0, 255, 1)
            if not ok:
                return  

            _, thresholded_img = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)

            self.display_image = thresholded_img
            self.update_image_label()


    def user_defined_filter(self):
        if self.original_image is not None:
            size, ok = QInputDialog.getInt(self, "Filter Size", "Enter filter size:", 3, 1, 11, 2)
        if ok:
            coefficients_str, ok = QInputDialog.getText(self, "Filter Coefficients", "Enter filter coefficients (comma-separated):")
            if ok:
                try:
                    coefficients = [float(x.strip()) for x in coefficients_str.split(',')]
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

    def apply_common_style(self, button):
        button.setObjectName("bn30")  # Set an object name to match the CSS selector
        button.clicked.connect(self.stylish_button_clicked)
        button.setStyleSheet("""
            #bn30 {
                border: 5em;
                outline: none;
                font-size: 13px;
                background-image: linear-gradient(45deg, #4568dc, #b06ab3);
                padding: 0.3em 2em;
                border-radius: 65px;
                color: red;
                background-color:black;
            }
        """)

    def stylish_button_clicked(self):
        print("Stylish button clicked!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ImageSegmentationApp()
    main_window.show()
    sys.exit(app.exec_())
