################################### Modules ######################################
import os                           # for getting system information
import sys                          # for using sys.exit() function
import cv2                          # for using OpenCV4.5 and CUDNN
import copy                         # for using CSI_Camera module
import signal                       # for making handler of SIGINT
import subprocess                   # for using subprocess call
import numpy as np                  # for getting maximum value
import pyzbar.pyzbar as pyzbar
from enum import Enum               # for using Enum type value
from csi_camera import CSI_Camera   # for using pi-camera in Jetson nano
# PyQt5 essential modules
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget, QSlider, QLabel, QPushButton, QFrame, QMessageBox
##################################################################################

####### Static Literals #######
DISPLAY_WIDTH   = 2048       # Display frame's width
DISPLAY_HEIGHT  = 1536       # Display frame's height
LIGHTER_COUNT   = 10
MAX_FRAME_COUNT = 1
LOWER_BOUNDARY_BLUE = np.array([95, 160, 160])      # HSV format
UPPER_BOUNDARY_BLUE = np.array([110, 255, 255])     # Hue, Saturation, Value
THERMAL_PATH = '/sys/devices/virtual/thermal/thermal_zone0/temp'


class State(Enum):
    IDLE    = 1     # Never used.
    SETTING = 2
    SOLVING = 3


class SystemInfo:
    """
    SystemInfo class is used for getting system information which will be indicated in GUI.
    Each information is updated every 3 minutes through PyQt5.QTimer.
    Optimization is necessary because there is a buffering about 0.5 seconds.
        (I think 'get_CPU_info' method is the cause.)
    """
    # Find AO(Always On) sensor's value
    def get_temp_info(self):
        # There is temperature of Jetson Nano in 'path'. Must devide by 1,000
        return int(subprocess.check_output(['cat', THERMAL_PATH]).decode('utf-8').rstrip('\n')) / 1000
    
    # Get current CPU usage percentage
    def get_CPU_info(self):
        # Get 'vmstat' command's 15th value
        cpu_resource_left = int(subprocess.check_output( "echo $(vmstat 1 2|tail -1|awk '{print $15}')", \
            shell = True, universal_newlines = True).rstrip('\n'))
        return 100 - cpu_resource_left
    
    # Get current memory usage percentage
    def get_mem_info(self):
        # Open '/proc/meminfo' and read first three lines
            # 1) Total memory, 2) free memory, 3) avaliable memory
        f = open('/proc/meminfo', 'rt')
        total = f.readline().split(':')[1].strip()[:-3] # 1) Total memory
        f.readline()
        avail = f.readline().split(':')[1].strip()[:-3] # 3) Available memory
        return round((int(avail) / int(total)) * 100)



class Sticker:
    def __init__(self):
        self.camera = CSI_Camera()
        self.net = cv2.dnn_DetectionModel("model.cfg", "model.weights")
        self.gpu_target_img = cv2.cuda_GpuMat()
        ###### Lighter information #######
        self.head_width = 0
        self.body_height = 0
        self.upper_sticker_bound = 0
        self.lower_sticker_bound = 0
        self.sticker_poses = []   # Lighter's sticker position for each lighter
        self.error_sticker_images = []
        # Manual camera setting variables, initialize with default size.
        self.manual_box_x           = DISPLAY_WIDTH // 2
        self.manual_box_y           = DISPLAY_HEIGHT // 2
        self.manual_box_width       = 150
        self.manual_box_height      = 300
        # ROI of image related parameters.
        self.roi_height = 0
        self.roi_width = 0
        self.roi_sy = 0
        self.roi_std_area = 0
        self.roi_min_area = 0
        self.roi_max_area = 0
        
        ###### Image Information ######
        self.display_contrast       = 30    # Default contrast 110%
        self.display_brightness     = 5     # Default brightness 105%
        
        self.initialize_camera()
        self.initialize_yolo()
        
    def initialize_camera(self):
        self.camera.create_gstreamer_pipeline (
            sensor_id       = 0,
            sensor_mode     = 0,
            framerate       = 30,
            flip_method     = 2,
            display_height  = DISPLAY_HEIGHT,
            display_width   = DISPLAY_WIDTH
        )
        self.camera.open(self.camera.gstreamer_pipeline)
        self.camera.start()
    
    def initialize_yolo(self):
        self.net.setInputSize(448, 448)      # It can be (448, 448) either if you need.
        self.net.setInputScale(1.0 / 255)    # Scaled by 1byte [0, 255]
        self.net.setInputSwapRB(True)        # Swap BGR order to RGB
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)    # For using CUDA GPU
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # For using CUDA GPU

    def get_image(self):
        ret, img = self.camera.read()
        a = 1 + round(self.contrast_slider.value() / 100, 2)
        b = self.brightness_slider.value()
        img = cv2.convertScaleAbs(img, alpha = a, beta = b)
        return img if ret is True else None
    
    def show_image(self):
        img = self.get_image()
        if img is None: return
        for sticker in self.sticker_poses:
            cv2.rectangle(img, sticker, (255, 255, 0), 5)
        return img
    
    def show_image_manual_setting(self):
        img = self.get_image()
        if img is None: return
        box = np.array([self.manual_box_x, self.manual_box_y, self.manual_box_width, self.manual_box_height])
        cv2.rectangle(img, box, (0, 125, 255), 5)
        return img
    
    def save_sticker_info(self):
        f = open("old_sticker_info.txt", 'wt')
        f.write(str(self.manual_box_x) + '\n')
        f.write(str(self.manual_box_y) + '\n')
        f.write(str(self.manual_box_width) + '\n')
        f.write(str(self.manual_box_height) + '\n')
        print("<New information successfully saved!>")
        f.close()

    def load_sticker_info(self):
        try: f = open("old_sticker_info.txt", "rt")
        except FileNotFoundError:
            print("Fail to load information file")
            return
        # Read five lines from text file.
        self.manual_box_x          = int(f.readline().rstrip('\n'))    # Previous box's x axis pos
        self.manual_box_y          = int(f.readline().rstrip('\n'))    # Previous box's y axis pos
        self.manual_box_width      = int(f.readline().rstrip('\n'))    # Previous box's width
        self.manual_box_height     = int(f.readline().rstrip('\n'))    # Previous box's height
        print("<Previous information successfully loaded!>")
        f.close()

    def set_roi(self):
        self.roi_height = self.manual_box_height // 3
        self.roi_width = self.manual_box_width
        self.roi_sy = (self.manual_box_y + self.manual_box_height) - self.roi_height

        self.roi_std_area = int((self.roi_width // 10) * self.roi_height)
        self.roi_min_area = self.roi_std_area * 0.3  # -50%
        self.roi_max_area = self.roi_std_area * 1.5  # +50%
        
        print("Setting Min area = ", self.roi_min_area, ", Max area = ", self.roi_max_area)

    def check_angle(self):
        img = self.get_image()
        if img is None: return
        
        cropped_img = img[self.roi_sy : self.roi_sy + self.roi_height, ]
        cropped_gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, cropped_thrs_img = cv2.threshold(cropped_gray_img, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(cropped_thrs_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cropped_thrs_img = cv2.cvtColor(cropped_thrs_img, cv2.COLOR_GRAY2BGR)
        err_cnt = 0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            #print(i, "th trial : ", area)
            if area < self.roi_min_area or area > self.roi_max_area: continue
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Retrieve the key parameters of the rotated bounding box
            width = int(rect[1][0])
            height = int(rect[1][1])
            angle = 90 - abs(round(rect[2], 1))
            if angle > 45 : angle = 90 - angle
            
            if abs(angle) >= 6: err_cnt += 1
            print(i, "th trial - Angle : ", round(angle, 1), ", Area", area)
            cv2.drawContours(cropped_thrs_img,[box], 0, (255, 0, 255), 4)   # Magenta
            # if abs(angle) > 5: cv2.drawContours(cropped_thrs_img,[box], 0, (255, 0, 255), 4)   # Magenta
            # else : cv2.drawContours(cropped_thrs_img,[box], 0, (50, 205, 154), 4)         # Yellowgreen
            
        print("---------------------")
        cv2.imwrite("test.png", cropped_thrs_img)
        self.sys_result_label.setText("불량품 세트"  if err_cnt > 2 else "정상 세트")

    def check_barcode(self):
        for loop_cnt in range(3):
            for sticker_num, sticker_img in self.error_sticker_images:
                if self.lighter_error_flag[sticker_num] is False: continue
                # Step 1. Substract blue color from sticker image.
                sticker_img_hsv = cv2.cvtColor(sticker_img, cv2.COLOR_BGR2HSV)
                mask_img = cv2.inRange(sticker_img_hsv, LOWER_BOUNDARY_BLUE, UPPER_BOUNDARY_BLUE)
                sticker_img = cv2.cvtColor(sticker_img, cv2.COLOR_BGR2GRAY)
                sticker_img = cv2.add(sticker_img, mask_img)
                # Step 2. Detect 1D barcode from sticker image.
                decoded = pyzbar.decode(sticker_img)
                if len(decoded) > 0 : self.lighter_error_flag[sticker_num] = False
            if True not in self.lighter_error_flag:
                self.sys_result_label.setText("정상 세트 [BM1]")
                return
            
        self.sys_result_label.setText("불량품 세트" if self.lighter_error_flag.count(True) > 1 
                                      else "정상 세트 [BM2]")
        
    def quit_program(self):
        self.camera.stop()
        self.camera.release()
        subprocess.call(["sudo -V", "systemctl", "restart", "nvargus-daemon"], shell = True)
        sys.exit(0)

    def sigint_handler(self, sig, frame):
        self.quit_program()


class StickerApp(QWidget, SystemInfo, Sticker):
    def __init__(self):
        signal.signal(signal.SIGINT, self.sigint_handler)    # Allocate Ctrl + C (SIGINT)'s handler.
        QWidget.__init__(self)
        Sticker.__init__(self)
        self.setFixedSize(1024, 600)    # Set windows 1024x600 for 7" display
        # self.setFixedSize(1280, 800)  # Set windows 1280x800 for 10.1" display
        # self.showMaximized()            # Set windows in fullscreen
        self.system_state = State.IDLE
        self.initUI()
        
    def initUI(self):
        grid_layout = QGridLayout()
        self.setLayout(grid_layout) # Set GUI layout to grid.
        self.load_sticker_info()    # Load second sticker info.
        
        ################ Create widgets in grid layout. ################
        ##### Timers #####
        self.sys_info_timer = QTimer(self)
        self.sys_info_timer.setInterval(180000)  # 3 minutes
        self.sys_info_timer.timeout.connect(self.update_system_info)
        
        self.show_timer = QTimer(self)
        self.show_timer.setInterval(100)    # 0.1 seconds
        self.show_timer.timeout.connect(self.update_camera_img)
        
        self.show_manual_setting_timer = QTimer(self)
        self.show_manual_setting_timer.setInterval(100)
        self.show_manual_setting_timer.timeout.connect(self.update_manual_setting_img)
        
        self.sys_info_timer.start()
        self.show_manual_setting_timer.start()
        
        ##### Labels and fonts #####
        self.sys_info_label = QLabel(self)
        self.sys_info_label.setStyleSheet("font-size: 16px; font-weight: bold")
        self.sys_info_label.setAlignment(Qt.AlignCenter)
        self.sys_info_label.setFrameStyle(QFrame.StyledPanel)
        self.update_system_info()
        
        self.camera_img_label = QLabel(self)
        
        self.sys_result_label = QLabel('판단 결과 출력', self)
        self.sys_result_label.setStyleSheet("font-weight: bold; color: red")
        self.sys_result_label.setWordWrap(True)
        self.sys_result_label.setAlignment(Qt.AlignCenter)
        self.sys_result_label.setFrameStyle(QFrame.Panel)
        
        self.manual_setting_label = QLabel('스티커 박스 크기 설정 (위: 높이, 아래: 너비)', self)
        self.manual_setting_label.setStyleSheet("font-weight: bold; color: darkblue")
        self.manual_setting_label.setWordWrap(True)
        self.manual_setting_label.setFrameStyle(QFrame.Panel)
        
        self.contrast_label = QLabel('이미지 대조(contrast) 설정', self)
        self.contrast_label.setStyleSheet("font-weight: bold; color: darkblue")
        self.contrast_label.setWordWrap(True)
        self.contrast_label.setFrameStyle(QFrame.Panel)
        
        self.brightness_label = QLabel('이미지 밝기(brightness) 설정', self)
        self.brightness_label.setStyleSheet("font-weight: bold; color: darkblue")
        self.brightness_label.setWordWrap(True)
        self.brightness_label.setFrameStyle(QFrame.Panel)
        
        self.updown_slider_label = QLabel(self)
        self.updown_slider_label.setLineWidth(2)
        self.updown_slider_label.setAlignment(Qt.AlignCenter)
        self.updown_slider_label.setFrameStyle(QFrame.Box)
        
        self.side_slider_label = QLabel(self)
        self.side_slider_label.setLineWidth(2)
        self.side_slider_label.setAlignment(Qt.AlignCenter)
        self.side_slider_label.setFrameStyle(QFrame.Box)
        
        self.contrast_slider_label = QLabel(self)
        self.contrast_slider_label.setLineWidth(2)
        self.contrast_slider_label.setAlignment(Qt.AlignCenter)
        self.contrast_slider_label.setFrameStyle(QFrame.Box)
        
        self.brightness_slider_label = QLabel(self)
        self.brightness_slider_label.setLineWidth(2)
        self.brightness_slider_label.setAlignment(Qt.AlignCenter)
        self.brightness_slider_label.setFrameStyle(QFrame.Box)
        
        ##### Buttons #####
        # Buttons for moving bounding box for manual setting.
        self.move_up_btn = QPushButton('↑', self)
        self.move_up_btn.clicked.connect(self.move_up)
        self.move_up_btn.setMaximumHeight(60)
        self.move_up_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.move_up_btn.setEnabled(True)
        
        self.move_down_btn = QPushButton('↓', self)
        self.move_down_btn.clicked.connect(self.move_down)
        self.move_down_btn.setMaximumHeight(60)
        self.move_down_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.move_down_btn.setEnabled(True)
        
        self.move_left_btn = QPushButton('←', self)
        self.move_left_btn.clicked.connect(self.move_left)
        self.move_left_btn.setMaximumHeight(60)
        self.move_left_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.move_left_btn.setEnabled(True)
        
        self.move_right_btn = QPushButton('→', self)
        self.move_right_btn.clicked.connect(self.move_right)
        self.move_right_btn.setMaximumHeight(60)
        self.move_right_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.move_right_btn.setEnabled(True)
        
        self.move_up_up_btn = QPushButton('⇈', self)
        self.move_up_up_btn.clicked.connect(self.move_up_up)
        self.move_up_up_btn.setMaximumHeight(60)
        self.move_up_up_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.move_up_up_btn.setEnabled(True)
        
        self.move_down_down_btn = QPushButton('⇊', self)
        self.move_down_down_btn.clicked.connect(self.move_down_down)
        self.move_down_down_btn.setMaximumHeight(60)
        self.move_down_down_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.move_down_down_btn.setEnabled(True)
        
        self.move_left_left_btn = QPushButton('⇇', self)
        self.move_left_left_btn.clicked.connect(self.move_left_left)
        self.move_left_left_btn.setMaximumHeight(60)
        self.move_left_left_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.move_left_left_btn.setEnabled(True)
        
        self.move_right_right_btn = QPushButton('⇉', self)
        self.move_right_right_btn.clicked.connect(self.move_right_right)
        self.move_right_right_btn.setMaximumHeight(60)
        self.move_right_right_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.move_right_right_btn.setEnabled(True)

        # Buttons for modifying bounding box for manual setting.
        self.increase_height_btn = QPushButton('+', self)
        self.increase_height_btn.clicked.connect(self.increase_box_height)
        self.increase_height_btn.setEnabled(True)
        self.decrease_height_btn = QPushButton('-', self)
        self.decrease_height_btn.clicked.connect(self.decrease_box_height)
        self.decrease_height_btn.setEnabled(True)
        
        self.increase_width_btn = QPushButton('+', self)
        self.increase_width_btn.clicked.connect(self.increase_box_width)
        self.increase_width_btn.setEnabled(True)
        self.decrease_width_btn = QPushButton('-', self)
        self.decrease_width_btn.clicked.connect(self.decrease_box_width)
        self.decrease_width_btn.setEnabled(True)
        
        # Buttons for modifying images.
        self.increase_contrast_btn = QPushButton('+', self)
        self.increase_contrast_btn.clicked.connect(self.increase_contrast)
        self.decrease_contrast_btn = QPushButton('-', self)
        self.decrease_contrast_btn.clicked.connect(self.decrease_contrast)
        
        self.increase_brightness_btn = QPushButton('+', self)
        self.increase_brightness_btn.clicked.connect(self.increase_brightness)
        self.decrease_brightness_btn = QPushButton('-', self)
        self.decrease_brightness_btn.clicked.connect(self.decrease_brightness)
        
        # Button for applying manual setting.
        self.apply_btn = QPushButton('설정', self)
        self.apply_btn.setStyleSheet('background-color: yellow; font-size: 16px; font-weight: bold')
        self.apply_btn.clicked.connect(self.apply_manual_setting)
        self.apply_btn.setMaximumHeight(60)
        self.apply_btn.setEnabled(True)

        # Buttons for etc.
        self.sys_start_btn      = QPushButton('솔루션 작동', self)
        self.sys_start_btn.clicked.connect(self.check_angle)
        self.sys_start_btn.setEnabled(False)
        self.sys_quit_btn       = QPushButton('시스템 종료', self)
        self.sys_quit_btn.clicked.connect(self.quit_program)
        
        ##### Slidebars #####
        self.updown_slider = QSlider(Qt.Horizontal, self)
        self.updown_slider.setRange(100, DISPLAY_HEIGHT)
        self.updown_slider.setValue(self.manual_box_height)
        self.updown_slider.valueChanged.connect(self.control_box_height)
        self.updown_slider_label.setText(str(self.manual_box_height))
        
        self.side_slider = QSlider(Qt.Horizontal, self)
        self.side_slider.setRange(300, DISPLAY_WIDTH)
        self.side_slider.setValue(self.manual_box_width)
        self.side_slider.valueChanged.connect(self.control_box_width)
        self.side_slider_label.setText(str(self.manual_box_width))
        
        self.contrast_slider = QSlider(Qt.Horizontal, self)
        self.contrast_slider.setRange(1, 100)   # From 100% to 200% 
        self.contrast_slider.setValue(self.display_contrast)
        self.contrast_slider.valueChanged.connect(self.control_contrast)
        self.contrast_slider_label.setText(str(self.display_contrast))
        
        self.brightness_slider = QSlider(Qt.Horizontal, self)
        self.brightness_slider.setRange(1, 100) # From 100% to 200% 
        self.brightness_slider.setValue(self.display_brightness)
        self.brightness_slider.valueChanged.connect(self.control_brightness)
        self.brightness_slider_label.setText(str(self.display_brightness))
        
        ##### Set widgets in grid layout. #####
        # Last 4 arguments indicate it's position and width, height.
        # [widget 1] :: System information.
        grid_layout.addWidget(self.sys_info_label,		0, 0, 3, 1)

        # [widget 2] :: OpenCV imshow realtime image.
        grid_layout.addWidget(self.camera_img_label,	3, 0, 12, 1)

        # [widget 3] :: Result information.
        grid_layout.addWidget(self.sys_result_label,	15, 0, 3, 1)	# 0~12

        # [widget 4] :: Joystick button for moving box.
        grid_layout.addWidget(self.move_up_up_btn,		0, 3, 2, 1)
        grid_layout.addWidget(self.move_up_btn,			2, 3, 2, 1)
        grid_layout.addWidget(self.move_left_left_btn,	4, 1, 2, 1)
        grid_layout.addWidget(self.move_left_btn,		4, 2, 2, 1)
        grid_layout.addWidget(self.move_right_btn,		4, 4, 2, 1)
        grid_layout.addWidget(self.move_right_right_btn,4, 5, 2, 1)
        grid_layout.addWidget(self.move_down_btn,		6, 3, 2, 1)
        grid_layout.addWidget(self.move_down_down_btn,	8, 3, 2, 1)

        # [widget 5] :: Apply manual setting button.
        grid_layout.addWidget(self.apply_btn,			8, 5, 2, 1)

        # [widget 6] :: Labels and Sliders
        grid_layout.addWidget(self.manual_setting_label,    10,  1,  1,  5)
        grid_layout.addWidget(self.updown_slider,           11,  1,  1,  2)
        grid_layout.addWidget(self.updown_slider_label,     11,  3,  1,  1)
        grid_layout.addWidget(self.increase_height_btn,     11,  4,  1,  1)
        grid_layout.addWidget(self.decrease_height_btn,     11,  5,  1,  1) 

        grid_layout.addWidget(self.side_slider,             12,  1,  1,  2)
        grid_layout.addWidget(self.side_slider_label,       12,  3,  1,  1)
        grid_layout.addWidget(self.increase_width_btn,      12,  4,  1,  1)
        grid_layout.addWidget(self.decrease_width_btn,      12,  5,  1,  1)

        grid_layout.addWidget(self.contrast_label,          13,  1,  1,  5)
        grid_layout.addWidget(self.contrast_slider,         14,  1,  1,  2)
        grid_layout.addWidget(self.contrast_slider_label,   14,  3,  1,  1)
        grid_layout.addWidget(self.increase_contrast_btn,   14,  4,  1,  1)
        grid_layout.addWidget(self.decrease_contrast_btn,   14,  5,  1,  1)

        grid_layout.addWidget(self.brightness_label,        15,  1,  1,  5)
        grid_layout.addWidget(self.brightness_slider,       16,  1,  1,  2)
        grid_layout.addWidget(self.brightness_slider_label, 16,  3,  1,  1)
        grid_layout.addWidget(self.increase_brightness_btn, 16,  4,  1,  1)
        grid_layout.addWidget(self.decrease_brightness_btn, 16,  5,  1,  1)

        # [widget 7] :: User control buttons.
        grid_layout.addWidget(self.sys_start_btn,		17, 1, 1, 2)
        grid_layout.addWidget(self.sys_quit_btn,		17, 4, 1, 2)

        
        self.setWindowTitle('Lighter GUI Program')
        self.show()

    def update_system_info(self):
        sys_info = "CPU 점유율: %d%%  메모리 점유율: %d%%  온도: %d°C\n" \
            % (self.get_CPU_info(), self.get_mem_info(), self.get_temp_info())
        sys_info += "시스템 현재 상태: "
        sys_info += "환경설정 중" if self.system_state is State.SETTING else "작동 중"
        self.sys_info_label.setText(sys_info)
    
    def set_image_label(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR format by default.
        h, w, c = img.shape
        pixmap = QPixmap.fromImage(QImage(img.data, w, h, w * c, QImage.Format_RGB888)).scaledToWidth(512)
        self.camera_img_label.setPixmap(pixmap)
        
    def update_camera_img(self):
        img = self.show_image()
        self.set_image_label(img)
    
    def update_manual_setting_img(self):
        img = self.show_image_manual_setting()
        self.set_image_label(img)

    def control_box_height(self):
        value = self.updown_slider.value()
        self.manual_box_height = value
        self.updown_slider_label.setText(str(value))
        
    def control_box_width(self):
        value = self.side_slider.value()
        self.manual_box_width = value
        self.side_slider_label.setText(str(value))
    
    def control_contrast(self):
        value = self.contrast_slider.value()
        self.display_contrast = value
        self.contrast_slider_label.setText(str(value))
        
    def control_brightness(self):
        value = self.brightness_slider.value()
        self.display_brightness = value
        self.brightness_slider_label.setText(str(value))
    
    def move_up(self):
        self.manual_box_y -= 3
    
    def move_down(self):
        self.manual_box_y += 3
    
    def move_left(self):
        self.manual_box_x -= 3
    
    def move_right(self):
        self.manual_box_x += 3

    def move_up_up(self):
        self.manual_box_y -= 30
    
    def move_down_down(self):
        self.manual_box_y += 30
    
    def move_left_left(self):
        self.manual_box_x -= 30
    
    def move_right_right(self):
        self.manual_box_x += 30
        
    def increase_box_height(self):
        self.manual_box_height += 3
        self.updown_slider.setValue(self.manual_box_height)
        self.updown_slider_label.setText(str(self.manual_box_height))
    
    def decrease_box_height(self):
        self.manual_box_height -= 3
        self.updown_slider.setValue(self.manual_box_height)
        self.updown_slider_label.setText(str(self.manual_box_height))
    
    def increase_box_width(self):
        self.manual_box_width += 3
        self.side_slider.setValue(self.manual_box_width)
        self.side_slider_label.setText(str(self.manual_box_width))

    def decrease_box_width(self):
        self.manual_box_width -= 3
        self.side_slider.setValue(self.manual_box_width)
        self.side_slider_label.setText(str(self.manual_box_width))
    
    def increase_contrast(self):
        self.display_contrast += 1
    
    def decrease_contrast(self):
        self.display_contrast -= 1
    
    def increase_brightness(self):
        self.display_brightness += 1
    
    def decrease_brightness(self):
        self.display_brightness -= 1
        
    def apply_manual_setting(self):
        self.set_roi()
        self.save_sticker_info()    # Save first sticker info.
        # Disable buttons for manual setting.
        self.move_up_btn.setEnabled(False)
        self.move_down_btn.setEnabled(False)
        self.move_left_btn.setEnabled(False)
        self.move_right_btn.setEnabled(False)
        self.move_up_up_btn.setEnabled(False)
        self.move_down_down_btn.setEnabled(False)
        self.move_left_left_btn.setEnabled(False)
        self.move_right_right_btn.setEnabled(False)
        self.increase_height_btn.setEnabled(False)
        self.decrease_height_btn.setEnabled(False)
        self.increase_width_btn.setEnabled(False)
        self.decrease_width_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        # Change system state to idle.
        self.sys_start_btn.setEnabled(True)
        self.system_state = State.IDLE
        self.update_system_info()
        self.show_manual_setting_timer.stop()
        self.show_timer.start()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = StickerApp()
    sys.exit(app.exec_())
