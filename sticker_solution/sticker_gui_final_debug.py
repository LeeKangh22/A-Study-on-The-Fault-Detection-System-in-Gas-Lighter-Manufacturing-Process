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
TEMPLATE_IMAGE_ACE = cv2.imread("./template_img_ace.jpg")
TEMPLATE_IMAGE_BTN = cv2.imread("./template_img_btn.jpg")
THERMAL_PATH = '/sys/devices/virtual/thermal/thermal_zone0/temp'
TEMPLATE_MATCHER = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_SQDIFF_NORMED)

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
        self.gpu_template_img = cv2.cuda_GpuMat(TEMPLATE_IMAGE_ACE)
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
        self.net.setInputSize(416, 416)      # It can be (448, 448) either if you need.
        self.net.setInputScale(1.0 / 255)    # Scaled by 1byte [0, 255]
        self.net.setInputSwapRB(True)        # Swap BGR order to RGB
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)    # For using CUDA GPU
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # For using CUDA GPU
    
    def get_head_ratio(self, up, down) :      # 라이터 헤드를 찾기 위한 좌표 계산 함수
        return int((down-up) * (5/105)) # 헤드 간 간격/몸통 길이
        
    def load_sticker_info(self):
        if self.apply_btn_push_cnt == 0:
            try: f = open("old_ace_sticker_info.txt", 'rt')
            except FileNotFoundError: 
                print("Fail to load information file")
                return
        else:
            try: f = open("old_btn_sticker_info.txt", 'rt')
            except FileNotFoundError: 
                print("Fail to load information file")
                return
        # Read five lines from text file.
        self.manual_box_x          = int(f.readline().rstrip('\n'))    # Previous box's x axis pos
        self.manual_box_y          = int(f.readline().rstrip('\n'))    # Previous box's y axis pos
        self.manual_box_width      = int(f.readline().rstrip('\n'))    # Previous box's width
        self.manual_box_height     = int(f.readline().rstrip('\n'))    # Previous box's height
        # Unnecessary printing for debugging purpose
        print("<Previous information successfully loaded!>")
        f.close()
    
    def save_sticker_info(self):
        if self.apply_btn_push_cnt == 1:
            f = open("old_ace_sticker_info.txt", 'wt')
            f.write(str(self.manual_box_x) + '\n')
            f.write(str(self.manual_box_y) + '\n')
            f.write(str(self.manual_box_width) + '\n')
            f.write(str(self.manual_box_height) + '\n')
        else:
            f = open("old_btn_sticker_info.txt", 'wt')
            f.write(str(self.manual_box_x) + '\n')
            f.write(str(self.manual_box_y) + '\n')
            f.write(str(self.manual_box_width) + '\n')
            f.write(str(self.manual_box_height) + '\n')
        self.save_template_image()
        # Unnecessary printing for debugging purpose
        print("<New information successfully saved!>")
        f.close()

    def save_template_image(self):
        img = self.get_image()
        sx = self.manual_box_x
        ex = self.manual_box_x + self.manual_box_width
        sy = self.manual_box_y
        ey = self.manual_box_y + self.manual_box_height
        template_img = img[sy:ey, sx:ex]
        offset_x =  int(round(self.manual_box_width * 0.1))
        offset_y = int(round(self.manual_box_height) * 0.05)
        template_img = cv2.medianBlur(template_img, 3)
        template_img = template_img[offset_y : -offset_y, offset_x : -offset_x]
        save_location = "./template_img_ace.jpg" if self.apply_btn_push_cnt == 1 else "./template_img_btn.jpg"
        cv2.imwrite(save_location, template_img)

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
        
    def set_camera_auto(self):
        head_widths = []
        head_heights = []
        head_lower_y = []
        
        img = self.get_image()
        self.sticker_poses.clear()
        
        classes, confidences, boxes = self.net.detect(img, confThreshold = 0.7, nmsThreshold = 0.7)
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            x, y, w, h = box
            head_heights.append(h)
            head_widths.append(w)
            head_lower_y.append(y + h)
            self.sticker_poses.append([x])
        self.head_width             = int(round(sum(head_widths) / 10))
        self.upper_sticker_bound    = int(round(sum(head_lower_y) / 10))
        self.body_height            = int(round(self.head_width * 3.8))
        self.lower_sticker_bound    = self.upper_sticker_bound + self.body_height
        self.sticker_poses.sort(key = lambda x : x[0])
        for i in range(LIGHTER_COUNT):
            self.sticker_poses[i].extend([self.upper_sticker_bound, self.head_width, self.body_height])
            self.sticker_poses[i] = np.array(self.sticker_poses[i])

    def do_template_matching(self):
        if len(self.sticker_poses) is not 10: return self.set_camera_auto()
        
        img = self.get_image()
        end_y = self.lower_sticker_bound
        start_y = self.upper_sticker_bound
        self.error_sticker_images.clear()
        self.lighter_error_flag = [True for _ in range(LIGHTER_COUNT)]
        
        for idx, sticker_pos in enumerate(self.sticker_poses):
            start_x = sticker_pos[0]
            end_x = sticker_pos[0] + sticker_pos[2]
            sticker_img = img[start_y : end_y, start_x : end_x]
            sticker_img = cv2.medianBlur(sticker_img, 3)
            self.gpu_target_img.upload(sticker_img)
            result = TEMPLATE_MATCHER.match(self.gpu_target_img, self.gpu_template_img).download()
            score = cv2.minMaxLoc(result)[0]
            if score <= 0.09: self.lighter_error_flag[idx] = False
        
        # Check whether there is an error
        if True not in self.lighter_error_flag:
            self.sys_result_label.setText("정상 세트 [TM]")
            return
        
        for idx, result in enumerate(self.lighter_error_flag):
            if result is True:
                start_x = self.sticker_poses[idx][0]
                end_x = start_x + self.head_width
                error_sticker_image = img[start_y : end_y, start_x : end_x]
                error_sticker_image = cv2.resize(error_sticker_image, (int(self.head_width / 1.5), int(self.upper_sticker_bound / 1.5)))
                self.error_sticker_images.append([idx, error_sticker_image])
        
        self.check_barcode()
            
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
            
        err_string = "불량품 세트"
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
        self.show_timer.start()
        
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
        # Buttons for selecting lighters type which is produced today.
        self.lighter_type_btn = QPushButton('에이스 라이터', self)
        self.lighter_type_btn.setCheckable(True)
        self.lighter_type_btn.setEnabled(True)
        self.lighter_type_btn.toggle()
        self.lighter_type_btn.toggled.connect(self.change_lighter_type)
        self.lighter_type_btn.setStyleSheet(
            "background-color: lightpink;       \
            border-style: outset;               \
            border-width: 2px;                  \
            border-radius: 10px;                \
            border-color: darkgray;             \
            padding: 6px;                       \
            color: black;                       \
            font-size: 16px;                    \
            font-weight: bold;                  \
            "
        )
        # Buttons for moving bounding box for manual setting.
        self.move_up_btn = QPushButton('↑', self)
        self.move_up_btn.clicked.connect(self.move_up)
        self.move_up_btn.setEnabled(False)
        
        self.move_down_btn = QPushButton('↓', self)
        self.move_down_btn.clicked.connect(self.move_down)
        self.move_down_btn.setEnabled(False)
        
        self.move_left_btn = QPushButton('←', self)
        self.move_left_btn.clicked.connect(self.move_left)
        self.move_left_btn.setEnabled(False)
        
        self.move_right_btn = QPushButton('→', self)
        self.move_right_btn.clicked.connect(self.move_right)
        self.move_right_btn.setEnabled(False)
        
        self.increase_height_btn = QPushButton('+', self)
        self.increase_height_btn.clicked.connect(self.increase_box_height)
        self.increase_height_btn.setEnabled(False)
        self.decrease_height_btn = QPushButton('-', self)
        self.decrease_height_btn.clicked.connect(self.decrease_box_height)
        self.decrease_height_btn.setEnabled(False)
        
        self.increase_width_btn = QPushButton('+', self)
        self.increase_width_btn.clicked.connect(self.increase_box_width)
        self.increase_width_btn.setEnabled(False)
        self.decrease_width_btn = QPushButton('-', self)
        self.decrease_width_btn.clicked.connect(self.decrease_box_width)
        self.decrease_width_btn.setEnabled(False)
        
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
        self.apply_btn.setStyleSheet('background-color: yellow; font-weight: bold')
        self.apply_btn.clicked.connect(self.apply_manual_setting)
        self.apply_btn.setEnabled(False)
        self.apply_btn_push_cnt = 0
        # Buttons for etc.
        self.auto_setting_btn   = QPushButton('자동설정', self)
        self.auto_setting_btn.clicked.connect(self.set_camera_auto)
        self.manual_setting_btn = QPushButton('수동설정', self)
        self.manual_setting_btn.clicked.connect(self.enter_manual_setting_mode)
        self.sys_start          = QPushButton('솔루션 작동', self)
        self.sys_start.clicked.connect(self.do_template_matching)
        self.sys_quit_btn       = QPushButton('시스템 종료', self)
        self.sys_quit_btn.clicked.connect(self.quit_program)
        
        ##### Slidebars #####
        self.updown_slider = QSlider(Qt.Horizontal, self)
        self.updown_slider.setRange(100, 1000)
        self.updown_slider.setValue(self.manual_box_height)
        self.updown_slider.valueChanged.connect(self.control_box_height)
        self.updown_slider_label.setText(str(self.manual_box_height))
        
        self.side_slider = QSlider(Qt.Horizontal, self)
        self.side_slider.setRange(50, 300)
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
        grid_layout.addWidget(self.sys_info_label,          0,  0,  2,  1)
        
        # [widget 2] :: OpenCV imshow realtime image.
        grid_layout.addWidget(self.camera_img_label,        2,  0,  10,  1)
        
        # [widget 3] :: Result information.
        grid_layout.addWidget(self.sys_result_label,        12,  0,  4,  1)
        
        # [widget 4] :: Select lighter's type (ACE or BTN)
        grid_layout.addWidget(self.lighter_type_btn,        0,  2,  2,  4)
        
        # [widget 5] :: Joystick button for moving box.
        grid_layout.addWidget(self.move_up_btn,             2,  3,  1,  2)
        grid_layout.addWidget(self.move_left_btn,           3,  1,  1,  2)
        grid_layout.addWidget(self.move_right_btn,          3,  5,  1,  2)
        grid_layout.addWidget(self.move_down_btn,           4,  3,  1,  2)
        
        # [widget 6] :: Apply manual setting button.
        grid_layout.addWidget(self.apply_btn,               2,  6,  1,  1)
        
        # [widget 7] :: Sliders
        grid_layout.addWidget(self.manual_setting_label,    5,  1,  1,  6)
        grid_layout.addWidget(self.updown_slider,           6,  1,  1,  3)
        grid_layout.addWidget(self.updown_slider_label,     6,  4,  1,  1)
        grid_layout.addWidget(self.increase_height_btn,     6,  5,  1,  1)
        grid_layout.addWidget(self.decrease_height_btn,     6,  6,  1,  1) 
        
        grid_layout.addWidget(self.side_slider,             7,  1,  1,  3)
        grid_layout.addWidget(self.side_slider_label,       7,  4,  1,  1)
        grid_layout.addWidget(self.increase_width_btn,      7,  5,  1,  1)
        grid_layout.addWidget(self.decrease_width_btn,      7,  6,  1,  1)
        
        grid_layout.addWidget(self.contrast_label,          8,  1,  1,  6)
        grid_layout.addWidget(self.contrast_slider,         9,  1,  1,  3)
        grid_layout.addWidget(self.contrast_slider_label,   9,  4,  1,  1)
        grid_layout.addWidget(self.increase_contrast_btn,   9,  5,  1,  1)
        grid_layout.addWidget(self.decrease_contrast_btn,   9,  6,  1,  1)
        
        grid_layout.addWidget(self.brightness_label,        10,  1,  1,  6)
        grid_layout.addWidget(self.brightness_slider,       11,  1,  1,  3)
        grid_layout.addWidget(self.brightness_slider_label, 11,  4,  1,  1)
        grid_layout.addWidget(self.increase_brightness_btn, 11,  5,  1,  1)
        grid_layout.addWidget(self.decrease_brightness_btn, 11,  6,  1,  1)
        
        # [widget 8] :: User control buttons.
        grid_layout.addWidget(self.auto_setting_btn,        12,  1,  2,  3)
        grid_layout.addWidget(self.manual_setting_btn,      12,  4,  2,  3)
        grid_layout.addWidget(self.sys_start,               14,  1,  2,  3)
        grid_layout.addWidget(self.sys_quit_btn,            14,  4,  2,  3)
        
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
    
    def change_lighter_type(self):
        if self.lighter_type_btn.isChecked():   # BTN -> ACE
            self.lighter_type_btn.setText('에이스 라이터')
            self.gpu_template_img.upload(TEMPLATE_IMAGE_ACE)
            QMessageBox.warning(
                self,
                '라이터 종류 변경',
                "판단 라이터가 에이스로 변경됩니다!\n에이스 제품이 맞는지 다시 한 번 확인해주세요.",
                QMessageBox.Ok
            )
        else: # ACE -> BTN
            self.lighter_type_btn.setText('불티나 라이터')
            self.gpu_template_img.upload(TEMPLATE_IMAGE_BTN)
            QMessageBox.warning(
                self,
                '라이터 종류 변경',
                "판단 라이터가 불티나로 변경됩니다!\n불티나 제품이 맞는지 다시 한 번 확인해주세요.",
                QMessageBox.Ok
            )

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
        
    def enter_manual_setting_mode(self):
        QMessageBox.warning(
            self,
            '수동설정 안내창 1',
            "수동설정 모드로 진입합니다!\n에이스 제품의 스티커 주위를 감싸도록 박스를 조절해주세요.",
            QMessageBox.Ok,
            QMessageBox.Ok
        )
        # Disable manual setting button.
        self.manual_setting_btn.setEnabled(False)
        # Enable buttons for manual setting.
        self.move_up_btn.setEnabled(True)
        self.move_down_btn.setEnabled(True)
        self.move_left_btn.setEnabled(True)
        self.move_right_btn.setEnabled(True)
        self.increase_height_btn.setEnabled(True)
        self.decrease_height_btn.setEnabled(True)
        self.increase_width_btn.setEnabled(True)
        self.decrease_width_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        
        self.load_sticker_info()        # Load first sitcker info.
        self.system_state = State.SETTING
        self.update_system_info()
        self.show_timer.stop()
        self.show_manual_setting_timer.start()
        
        
    def apply_manual_setting(self):
        self.apply_btn_push_cnt += 1
        if self.apply_btn_push_cnt == 1:
            self.save_sticker_info()    # Save first sticker info.
            self.load_sticker_info()    # Load second sticker info.
            QMessageBox.warning(
                self,
                '수동설정 안내창 2',
                "첫 번째 탬플릿 이미지 저장 성공!\n불티나 제품의 스티커 주위를 감싸도록 박스를 조절해주세요.",
                QMessageBox.Ok,
                QMessageBox.Ok
            )
            return
        self.save_sticker_info()        # Save second sticker info.
        self.apply_btn_push_cnt = 0
        # Enable buttons for manual setting.
        self.move_up_btn.setEnabled(False)
        self.move_down_btn.setEnabled(False)
        self.move_left_btn.setEnabled(False)
        self.move_right_btn.setEnabled(False)
        self.increase_height_btn.setEnabled(False)
        self.decrease_height_btn.setEnabled(False)
        self.increase_width_btn.setEnabled(False)
        self.decrease_width_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        # Change system state to idle.
        self.manual_setting_btn.setEnabled(True)
        if self.lighter_type_btn.isChecked():   # BTN -> ACE:
            self.gpu_template_img.upload(TEMPLATE_IMAGE_ACE)
        else:
            self.gpu_template_img.upload(TEMPLATE_IMAGE_BTN)
        self.system_state = State.IDLE
        self.update_system_info()
        self.show_manual_setting_timer.stop()
        self.show_timer.start()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = StickerApp()
    sys.exit(app.exec_())
