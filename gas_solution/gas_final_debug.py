################# Modules #######################################################
import os               # for getting system information
import re               # for getting system information
import sys              # for using sys.exit() function
import cv2              # for using OpenCV4.5 and CUDNN
import copy             # for using CSI_Camera module
import time             # for using time.sleep function
import numpy as np      # for making various zeros array
import math             # for using math.ceil() function
import signal           # for making handler of SIGINT
import platform         # for getting system information
import subprocess       # for getting system information
from enum import Enum   # for using Enum type value
from csi_camera import CSI_Camera   # for using pi-camera in Jetson nano
from collections import OrderedDict # for using OrderedDict type value
# PyQt5 essential modules
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget, QSlider, QLabel, QPushButton, QFrame
##################################################################################

############## Static Variables ################
class State(Enum):
    IDLE    = 1     # Never used.
    SETTING = 2
    SOLVING = 3

THERMAL_PATH = '/sys/devices/virtual/thermal/thermal_zone0/temp'

############## Classes ################

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



class Yolo:
    """
    Yolo class is used for getting images from camera and judging defect existance.
    Flow is serial process which is consisted of 
        1) Initialize camera (Pi-Camera HQ (IMX477))
        2) Load lighter information which saved previous execution.
        3) Set standard lines and normal lines
        4) Find defects that may be in a single set of lighters.
    """
    def __init__(self):
        # Pi-Camera for jetson nano (CSI_Camera)
        self.camera = CSI_Camera()
        self.gpu_frame = cv2.cuda_GpuMat()

        # 4:3 Resolution (Capture Resolution: 4032 x 3040)
        self.display_width   = 1280     # Display width (not captured width)
        self.display_height  = 960      # Display height (not captured height)
        self.upper_not_roi   = self.display_height // 4         # Default value of ROI
        self.lower_not_roi   = 3 * self.display_height // 4     # Default value of ROI

        # Load YOLOv4
        self.net = cv2.dnn_DetectionModel("yolov4-tiny.cfg", "yolov4-tiny-final.weights")
        self.net.setInputSize(448, 448)      # It can be (416, 416) either
        self.net.setInputScale(1.0 / 255)    # Scaled by 1byte [0, 255]
        self.net.setInputSwapRB(True)        # Swap BGR order to RGB
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)    # For using CUDA GPU
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # For using CUDA GPU

        # Standard Lines
        self.upper_std_line     = 0     # Lighter's standard line which is used for getting height
        self.lower_std_line     = 0     # Lighter's standard line which is used for getting height
        self.upper_normal_line  = 0     # Normal bound line to decide whether lighter has defect or not
        self.lower_normal_line  = 0     # Normal bound line to decide whether lighter has defect or not
        
        # QSlider bars
        # ROI which can be controlled by user
        self.roi_upper   = 0                    
        self.roi_lower   = self.display_height
        self.roi_col     = 0
        # Lighter's standard line which can be controlled by user
        self.std_upper   = 0
        self.std_lower   = self.display_height

        # Lighter Informations
        self.lighter_width      = 0     # Thickness of one lighter
        self.lighter_height     = 0     # Height of lighter

        # Image filters
        # Sobel filter is used to extract the outline of an image.
        self.sobel_filter        = cv2.cuda.createSobelFilter(cv2.CV_8U, cv2.CV_8U, 0, 1, ksize=5, scale=0.25)
        # Hough segment filter is used to extract standard line of lighter.
            # You can decrease first two arguments if you want higher precision.
            # 3rd arg: minimum length of line
            # 4th arg: minimum gap of each lines
        self.hough_seg_filter    = cv2.cuda.createHoughSegmentDetector(0.1, np.pi / 180, 80, 1)
        
        signal.signal(signal.SIGINT, self.sigint_handler)    # Allocate Ctrl + C (SIGINT)'s handler.
        self.initialize_camera()
        self.load_lighter_info()
    
    def initialize_camera(self):
        """
        Args:
        
        Returns:
        
        Raises:
            camera.video_capture.isOpened():
                Camera module cannot capture images.
                Something wrong in camera module(HW) or nvargus-daemon(SW)
                'sudo systemctl restart nvargus-daemon' can be help.
                self.quit_program() function performs that role.
                
        Note:
            CIS Image sensor (IMX477) provides maximum 29FPS in sensor mode 0.
            You do not need to modify 'sensor_id' and 'sensor_mode' unless you use only one camera module.
            'flip_method' is set to '2' for flipping 180 degree image clockwise.
        """
        
        self.camera.create_gstreamer_pipeline(
            sensor_id       = 0,        # Camera sensor ID      (Do not need modify)
            sensor_mode     = 0,        # Camera sensor mode    (IMX477 driver provides 4 option)
            framerate       = 10,       # Camera framerate      (10FPS used for this application) 
            flip_method     = 2,        # Flip image 90 degree (90 x 2 = 180)
            display_height  = self.display_height,   
            display_width   = self.display_width
        )
        self.camera.open(self.camera.gstreamer_pipeline)
        # [Exception handling] :: Camera can't start.
        if not self.camera.video_capture.isOpened():
            print("[Exception] :: Unable to open camera")
            self.quit_program()
        self.camera.start()     # Start camera (Make threads and execute).
        
    def make_ROI_screen(self, img):
        """
        Args:
            img:
                The image for which the Region of Interest (ROI) is to be set.
                
        Returns:
            bool:
                Variable to indicate whether a function is successful.
            img:
                The image with ROI of which pixels in some areas have changed to zero.
                
        Raises:
            Invalid image:
                if img is None, it should be returned.
        
        Note:
            'roi_upper', 'roi_lower' and 'roi_col' is controlled by user.
            Coordinates get larger as they go from top to bottom. 
            It's easy to understand when you think of a two-dimensional array or list.
            Therefore, the coordinates of the lower boundary are bigger than the upper boundary.
        """
        # [Exception handling] :: The image is invalid.
        if img is None:
            print("[Exception] :: There is no image to making ROI")
            return False, img
        
        img[:self.roi_upper, :] = img[self.roi_lower:, :] = 0   # Remove the top and bottom areas.
        img[:, :self.roi_col] = img[:, self.display_width - self.roi_col:] = 0  # Remove the left and right areas.
        return True, img

    def calculate_lighter_info(self, img, roi = 10, loop = 50) :
        """
        Args:
            img:
                The image that we want to know properties(width, hegith, standard line). 
            roi:
                The pixel value of the area where the upper standard line is expected to be found.
                If the 'roi_upper' is properly set, upper standard line must exist within 10px.
            loop:
                The number of iterations of a function to find more accurate return values.
                As a result of the experiment, there was no significant difference from 50 or higher.
                
        Returns:
            bool:
                Variable to indicate whether a function is successful.
            width:
                Thickness of one lighter.
                It is detected as 98 ~ 101px at a distance of approximately 30 ~ 33cm.
            height:
                Height of a set of lighter.
                It is detected as 380 ~ 404px at a distance of approximately 30 ~ 33cm.
                In general, the height is about four times the width.
                
        Raises:
            Invalid image:
                if img is None, it should be returned.
        
        Note:
            'roi_upper', 'roi_lower' and 'roi_col' is controlled by user.
            Coordinates get larger as they go from top to bottom. 
            It's easy to understand when you think of a two-dimensional array or list.
            Therefore, the coordinates of the lower boundary are bigger than the upper boundary.
            
            3 is added to 'width', which is the correction value obtained from the experiment.
            This is because the line obtained by the hough filter can be omitted in both ends about 1, 2 pixels.
            
            3.95 is multiplied to 'width' because the actual length of the fuel tank is measured.
            
        """
        
        widths        = []          # Calulated thicknesses of each lighter. (there are loop elements)
        upper_lines   = []          # Configured lines of a set of lighter. (there are loop elements)
        self.gpu_frame.upload(img)  # Upload image to GPU side for optimization.
        
        for _ in range(loop):
            width_candidates        = []
            upper_line_candidates   = []
            # Apply 'hough segmentation filter' to the image and download image to CPU side from GPU side.
            lines = self.hough_seg_filter.detect(self.gpu_frame).download()
            if lines is None: continue  # [Exception 1] :: No line detected. Something wrong.
            for line in lines:
                for x1, y1, x2, y2 in line: # Each line is consisted of start and end coords.
                    # y-coordinate exists in an area where the upper standard line is likely to exist.
                    if self.roi_upper < y1 < self.roi_upper + roi :
                        upper_line_candidates.append(y1)
                        width_candidates.append(math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2)))
            # [Exception 2] :: No valid line detected.
            if len(upper_line_candidates) == 0 or len(width_candidates) == 0: continue
            # Get upper standard line's y-coordinate and thickness of lighter.
            upper_lines.append(sum(upper_line_candidates) / len(upper_line_candidates))
            widths.append(sum(width_candidates) / len(width_candidates))
        # [Exception 3] :: No valid data detected. Something wrong in image. Retry.
        if len(widths) is 0 or len(upper_lines) is 0: return False, 0, 0, 0
        
        width       = int(round(sum(widths, 0) / len(widths))) + 3          # Get lighter's thickness
        upper_line  = int(round(sum(upper_lines, 0) / len(upper_lines)))    # Get upper_std_line
        height      = int(round(width * 3.95))                              # Get lighter's height
        return True, width, height, upper_line

    def get_camera_image(self):
        """
        Args:
                
        Returns:
            img:
                The image from camera sensor.
            
        Raises:
            Invalid image:
                If return flag of 'make_ROI_screen()' or `camera.read()` function is False, there is something wrong in image.
                Therefore, we need to get new image from camera.
                But we should make proper exception handler since it can make infinite loop in case of cameara malfunction.
        
        Note:
            Get camera image and process image are consisted of below steps.
                1) Get image from pi-camera(IMX477).
                2) Convert image from RGB to gray through `cv2.cvtColor()`.
                3) Upload image to GPU side from CPU side.
                4) Apply sobel filter to image.
                5) Download image from GPU side and apply ROI to image.
                6) Convert image from gray to binary for optimization.
                
            Be cautious to make any `cv2.cuda_GpuMat()` because \
            upload and download image between GPU side and CPU side can be serious overhead.
            
        """
        ret, frame = self.camera.read()                                 # step 1
        if ret == True: 
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)        # step 2
            self.gpu_frame.upload(gray_frame)                           # step 3
            self.gpu_frame = self.sobel_filter.apply(self.gpu_frame)    # step 4
            img = self.gpu_frame.download()                             # step 5
            result, img = self.make_ROI_screen(img)
            if result is False:
                print("[Exception] :: There is something wrong in an image to making ROI")
                return self.get_camera_image()   # TODO :: Make proper exception handling
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # step 6
            return img
        else:
            print("[Exception] :: Read image from CSI_Camera is failed!, retry...")
            return self.get_camera_image()   # TODO :: Make proper exception handling.
        
    def test(self, img, err_rate = [], flag = False):
        """
        Args:
            img:
                Image that requires determining whether it is defective.
            err_rate:
                2D array which indicates error rates of each lighter.
                Each index represents a lighter number.
            flag:
                A flag variable to determine whether a call is for loading a YOLO model or a call for judgement.
                Because the first time this function is called, a long delay (approximately 3 seconds) occurs, 
                we call this function at program execution to pass the delay in advance before full-scale judgment is performed.
                
        Returns:
            img:
                The image which is drawed rectangles and labels.
                But final result doesn't need this image, so it is only for debugging purpose.
            
        Raises:
        
        Note:
            YOLO model detects gas surface and makes boundary NMS box automatically.
            Of course, box and label are used only for debugging purpose.
            So in final release, we should delete every unnecessary part except 'center coords'.
            
        """
        try:
            classes, confidences, boxes = self.net.detect(img, confThreshold = 0.3, nmsThreshold = 0.3)
            # [Exception] :: Invalid image or setting screen phase.
            if len(boxes) == 0 or flag is True: return
            box_cnt = 0
            center_list = []
            # get center position of each box (gas surface).
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%.2f' % confidence
                labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                left, top, width, height = box
                label_width, label_height = labelSize
                
                # [Exception] :: The box is under of ROI.
                if top + height > self.lower_std_line : continue
                
                box_cnt += 1
                # Make rectangles of gas surface and label to image.
                cv2.rectangle(img, box, color = (0, 255, 255), thickness = 2)   
                cv2.rectangle(img, (left, top - label_height), (left + label_width, top), (0, 255, 255), cv2.FILLED)
                cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                # Get center pos of box.
                # center_x = left + (width // 2)
                # center_y = top + (height // 2)
                # center_list.append([center_x, center_y, left, top, width, height])
                center_list.append(box)
            
            center_list.sort(key = lambda x : x[0]) # Sort boxes in ascending order.
            num = 0
            prev = self.roi_col
            for pos in center_list: # For every lighter,
                l, t, w, h = pos
                # Calculation to find the correct order if n-th lighter surface is not recognized.
                # Calculate how many lighters can fit between two adjacent lighters.
                num += ((abs(l - prev)) // self.lighter_width) + 1
                prev = l + w
                # TODO : Proper exception handling.
                if num > 10: break      # A set of lighter must consists of 10-lighter. 
                
                # Get error rate of each lighter
                if t < self.upper_normal_line: hh = self.upper_normal_line - t                # Get 'excessive' error rate
                elif t + h > self.lower_normal_line: hh = self.lower_normal_line - (t + h)    # Get 'under' error rate
                else: hh = 0.0          # If there is no defect, it has 0% error rate.
                err_rate[num - 1].append(hh / h)
            
            # print("\t{} boxes founded".format(box_cnt))
            return img
            
        except Exception as ex:
            print("[Error] : ", ex)
            self.quit_program()
        
    def solve(self, loop = 5):
        """
        Args:
            loop:
                Number of images taken to increase accuracy.
                The images are taken once every 0.1 second when outter signal is recognized.
                The lighter set travels about every 1.8 seconds.
                The first 0.5 seconds are not appropriate because the surface shakes.
                So about 5 to 10 pictures are appropriate.
                
        Returns:
                
        Raises:
        
        Note:
            All print statements in this function are only for debugging purpose.
            
        """
        img_cnt = 0
        self.lighter_error = [0 for _ in range(10)]     # Bool flag list whether n'th lighter has a defect or not
        lighter_error_rate = [[] for _ in range(10)]    # n'th lighter's error rate
        try:
            
            while img_cnt < loop:
                img = self.get_camera_image()
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = self.test(img, lighter_error_rate)
                # Unnecessary drawing for debugging purpose
                cv2.line(img, (0, self.upper_std_line),         (self.display_width, self.upper_std_line),      (255,255,0),    2)
                cv2.line(img, (0, self.upper_normal_line),      (self.display_width, self.upper_normal_line),   (0,0,255),      2)
                cv2.line(img, (0, self.lower_normal_line),      (self.display_width, self.lower_normal_line),   (0,0,255),      2)
                cv2.line(img, (0, self.lower_std_line),         (self.display_width, self.lower_std_line),      (255,0,255),    2)
                if img is not None: cv2.imwrite("/home/nano1/Desktop/final/" + str(img_cnt) + ".png", img)
                img_cnt += 1
                time.sleep(0.1)   # sleep for 0.1 seconds.
            # Decide whether n'th lighter is abnormal.
            for i, rate in enumerate(lighter_error_rate):
                print('[' + str(i + 1) + ']', end="\t")
                if (len(rate) == 0):    # DBZ error prevention.
                    print("-1.00\tAbnormal [인식불가]")
                    self.lighter_error[i] = 1
                    continue
                for ele in rate: print("%5.2f" % ele, end="    ")
                # Get average error rate of each lighter.   
                err_rate = round(sum(rate) / len(rate), 2)
                print(":%5.2f" % err_rate, end="\t")
                # Lighter has defect when it's error rate is over 90%.
                if err_rate >= 0.9:
                    print("\tAbnormal [초과]")
                    self.lighter_error[i] = 2
                elif err_rate <= -0.9 :
                    print("\tAbnormal [미달]")
                    self.lighter_error[i] = 1
                else:
                    print("\tNormal")
                    self.lighter_error[i] = 0
            print("[Capture Done]")
        except Exception as ex:
            print("[Error] : ", ex)
            self.quit_program()
    
    def set_screen_auto(self):
        """
        Args:
            
        Returns:
            
        Raises:
        
        Note:
            Because of `calculate_lighter_info()` function which gets lighter's height, width and standard lines,
            we can get lighter's information automatically.
            
        """
        print("진입")
        result = False
        while result is False : 
            img = self.get_camera_image()
            result, self.lighter_width, self.lighter_height, self.upper_std_line = self.calculate_lighter_info(img)
            print(result)
        print("탈출")
        self.lower_std_line = self.upper_std_line + self.lighter_height  # Lower boundary of lighter (tray)
        self.upper_normal_line = self.upper_std_line + int(self.lighter_height * (21 / 39))  # Upper boundary of normal area.
        self.lower_normal_line = self.upper_std_line + int(self.lighter_height * (29 / 39))  # Lower boundary of normal area.
        self.test(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), flag = True)
        self.save_lighter_info()
        # Unnecessary printing for debugging purpose
        print("[Initialization Done]")
        print("\tUPPER_STANDARD_LINE = {}, self.lighter_width = {}, self.lighter_height = {}".format(self.upper_std_line, self.lighter_width, self.lighter_height))
        print("\tUPPER_NORMAL_LINE = {}, self.lower_normal_line = {}".format(self.upper_normal_line, self.lower_normal_line))

    def set_screen_manual(self):
        """
        Args:
            
        Returns:
            
        Raises:
        
        Note:
            User can control sliders and buttons to set lighter's properties.            
            Because sometimes automatic setting shows lack precision, user might set properties by himself.
            
        """
        self.upper_std_line = self.std_upper    # Upper standard boundary which is set by user.
        self.lower_std_line = self.std_lower    # Lower standard boundary which is set by user.
        self.lighter_height = self.std_lower - self.std_upper   # Get height of lighter.
        self.lighter_width = self.lighter_height // 4           # Get thickness of lighter.
        self.upper_normal_line = self.std_upper + int(self.lighter_height * (21 / 39))  # Upper boundary of normal area.
        self.lower_normal_line = self.std_upper + int(self.lighter_height * (29 / 39))  # Lower boundary of normal area.
        img = self.get_camera_image()
        self.test(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), flag = True)
        self.save_lighter_info()
        # Unnecessary printing for debugging purpose
        print("[Initialization Done]")
        print("\tUPPER_STANDARD_LINE = {}, self.lighter_width = {}, self.lighter_height = {}".format(self.upper_std_line, self.lighter_width, self.lighter_height))
        print("\tUPPER_NORMAL_LINE = {}, self.lower_normal_line = {}".format(self.upper_normal_line, self.lower_normal_line))

    def load_lighter_info(self):
        """
        Args:
            
        Returns:
            
        Raises:
        
        Note:
            Loads several lighter properties saved by a previously run program.
            
        """
        try: f = open("old_lighter_info.txt", 'rt')
        except FileNotFoundError: 
            print("Fail to load information file")
            return
        # Read five lines from text file.
        self.upper_std_line = int(f.readline().rstrip('\n'))    # Previous standard line boundary
        self.lower_std_line = int(f.readline().rstrip('\n'))    # Previous standard line boundary
        self.roi_upper      = int(f.readline().rstrip('\n'))    # Previous ROI value
        self.roi_lower      = int(f.readline().rstrip('\n'))    # Previous ROI value
        self.roi_col        = int(f.readline().rstrip('\n'))    # Previous ROI value

        self.lighter_height = self.lower_std_line - self.upper_std_line    # Get height of lighter.
        self.lighter_width = self.lighter_height // 4                      # Get thickness of lighter.    
        self.upper_normal_line = self.upper_std_line + int(self.lighter_height * (21 / 39))  # Upper boundary of normal area.
        self.lower_normal_line = self.upper_std_line + int(self.lighter_height * (29 / 39))  # Lower boundary of normal area.
        # Unnecessary printing for debugging purpose
        print("<Previous information successfully loaded!>")
        print("[Initialization Done]")
        print("\tUPPER_STANDARD_LINE = {}, self.lighter_width = {}, self.lighter_height = {}".format(self.upper_std_line, self.lighter_width, self.lighter_height))
        print("\tUPPER_NORMAL_LINE = {}, self.lower_normal_line = {}".format(self.upper_normal_line, self.lower_normal_line))

        f.close()
        
    def save_lighter_info(self):
        """
        Args:
            
        Returns:
            
        Raises:
        
        Note:
            Save several lighter properties for next run program.
            
        """
        f = open("old_lighter_info.txt", 'wt')
        f.write(str(self.upper_std_line) + '\n')
        f.write(str(self.lower_std_line) + '\n')
        f.write(str(self.roi_upper) + '\n')
        f.write(str(self.roi_lower) + '\n')
        f.write(str(self.roi_col) + '\n')
        # Unnecessary printing for debugging purpose
        print("<New information successfully saved!>")
        f.close()
    
    def quit_program(self):
        """
        Args:
            
        Returns:
            
        Raises:
        
        Note:
            Shut down the program with proper resources collecting.
            Because 'camera' is made of multiple threads, proper resources collecting must be done.
            
        """
        self.camera.stop()
        self.camera.release()
        subprocess.call(["sudo -V", "systemctl", "restart", "nvargus-daemon"], shell = True)
        sys.exit(0)
        
    def sigint_handler(self, sig, frame):
        # If Ctrl + C is called.
        self.quit_program()
        
    def yolo_loop(self):
        """
        Args:
            
        Returns:
            
        Raises:
        
        Note:
            This function does not perform any special functions, 
            but only performs to show images from the camera to the GUI in real time.
            And it visually helps user can easily control sliders and buttons.
            
        """
        img = self.get_camera_image()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.line(img, (self.roi_col, 0), (self.roi_col, self.display_height), (200, 200, 200), 1)
        cv2.line(img, (self.display_width - self.roi_col, 0), (self.display_width - self.roi_col, self.display_height), (200, 200, 200), 1)
        cv2.line(img, (0, self.std_upper), (self.display_width, self.std_upper), (200, 200, 200), 1)
        cv2.line(img, (0, self.std_lower), (self.display_width, self.std_lower), (200, 200, 200), 1)
        cv2.line(img, (0, self.upper_std_line), (self.display_width, self.upper_std_line), (255, 0, 255), 2)
        cv2.line(img, (0, self.lower_std_line), (self.display_width, self.lower_std_line), (255, 0, 255), 2)
        cv2.line(img, (0, self.upper_normal_line), (self.display_width, self.upper_normal_line), (0, 0, 255), 2)
        cv2.line(img, (0, self.lower_normal_line), (self.display_width, self.lower_normal_line), (0, 0, 255), 2)
        return img
        

class LighterApp(QWidget, SystemInfo, Yolo):
    """
    LighterApp class is GUI application by using PyQt5.
    GUI has a grid layout (which is consisted of several rows and cols).
    """
    def __init__(self):
        """
            Initialize it's super classes and UI.
            GUI window's size is set by 1024x600 because of display hardware.
            It can be modified if you use bigger display.
        """
        QWidget.__init__(self)
        Yolo.__init__(self)
        self.setFixedSize(1024, 600)    # Set windows 1024x600 for debugging
        # self.showMaximized()            # Set windows in fullscreen
        self.system_state = State.SETTING
        self.initUI()
        
    def initUI(self):
        """
            Initialize UI by matching grid layout.
        """
        grid_layout = QGridLayout()
        self.setLayout(grid_layout) # Set GUI layout to grid.
        
        ################ Create widgets in grid layout. ################
        ##### Timers #####
        sys_info_timer = QTimer(self)
        sys_info_timer.setInterval(180000)  # 3 minutes
        sys_info_timer.timeout.connect(self.update_system_info)
        
        yolo_loop_timer = QTimer(self)
        yolo_loop_timer.setInterval(100)    # 0.1 seconds
        yolo_loop_timer.timeout.connect(self.update_camera_img)
        
        sys_info_timer.start()
        yolo_loop_timer.start()
        
        
        ##### Labels and fonts #####
        self.sys_info_label = QLabel(self)
        self.sys_info_label.setStyleSheet("font-size: 16px; font-weight: bold")
        self.sys_info_label.setAlignment(Qt.AlignCenter)
        self.sys_info_label.setFrameStyle(QFrame.StyledPanel)
        self.update_system_info()
        
        self.camera_img_label = QLabel(self)
        
        self.sys_result_label       = QLabel('판단 결과 출력', self)
        self.sys_result_label.setStyleSheet("font-weight: bold; color: red")
        self.sys_result_label.setWordWrap(True)
        self.sys_result_label.setAlignment(Qt.AlignCenter)
        self.sys_result_label.setFrameStyle(QFrame.Panel)
        
        self.upper_roi_label        = QLabel('상단 ROI 설정', self)
        self.upper_roi_label.setFrameStyle(QFrame.StyledPanel)
        self.upper_roi_label.setAlignment(Qt.AlignCenter)
        self.upper_roi_label.setStyleSheet("font-size: 20px; font-weight: bold")
        
        self.lower_roi_label        = QLabel('하단 ROI 설정', self)
        self.lower_roi_label.setFrameStyle(QFrame.StyledPanel)
        self.lower_roi_label.setAlignment(Qt.AlignCenter)
        self.lower_roi_label.setStyleSheet("font-size: 20px; font-weight: bold")
        
        self.col_roi_label          = QLabel('양옆 ROI 설정', self)
        self.col_roi_label.setFrameStyle(QFrame.StyledPanel)
        self.col_roi_label.setAlignment(Qt.AlignCenter)
        self.col_roi_label.setStyleSheet("font-size: 20px; font-weight: bold")
        
        self.upper_std_label        = QLabel('상단 기준선 설정', self)
        self.upper_std_label.setFrameStyle(QFrame.StyledPanel)
        self.upper_std_label.setAlignment(Qt.AlignCenter)
        self.upper_std_label.setStyleSheet("font-size: 20px; font-weight: bold")
        
        self.lower_std_label        = QLabel('하단 기준선 설정', self)
        self.lower_std_label.setFrameStyle(QFrame.StyledPanel)
        self.lower_std_label.setAlignment(Qt.AlignCenter)
        self.lower_std_label.setStyleSheet("font-size: 20px; font-weight: bold")
        
        self.upper_roi_value_label  = QLabel(self)
        self.upper_roi_value_label.setLineWidth(2)
        self.upper_roi_value_label.setAlignment(Qt.AlignCenter)
        self.upper_roi_value_label.setFrameStyle(QFrame.Box)
        
        self.lower_roi_value_label  = QLabel(self)
        self.lower_roi_value_label.setLineWidth(2)
        self.lower_roi_value_label.setAlignment(Qt.AlignCenter)
        self.lower_roi_value_label.setFrameStyle(QFrame.Box)
        
        self.col_roi_value_label    = QLabel(self)
        self.col_roi_value_label.setLineWidth(2)
        self.col_roi_value_label.setAlignment(Qt.AlignCenter)
        self.col_roi_value_label.setFrameStyle(QFrame.Box)
        
        self.upper_std_value_label  = QLabel(self)
        self.upper_std_value_label.setLineWidth(2)
        self.upper_std_value_label.setAlignment(Qt.AlignCenter)
        self.upper_std_value_label.setFrameStyle(QFrame.Box)
        
        self.lower_std_value_label  = QLabel(self)
        self.lower_std_value_label.setLineWidth(2)
        self.lower_std_value_label.setAlignment(Qt.AlignCenter)
        self.lower_std_value_label.setFrameStyle(QFrame.Box)
        
        
        ##### Slidebars #####
        self.upper_roi_slider = QSlider(Qt.Horizontal, self)
        self.upper_roi_slider.setRange(0, self.display_height)
        self.upper_roi_slider.setValue(self.roi_upper)
        self.upper_roi_slider.valueChanged.connect(self.update_upper_roi)
        self.upper_roi_value_label.setText(str(self.roi_upper))
        
        self.lower_roi_slider = QSlider(Qt.Horizontal, self)
        self.lower_roi_slider.setRange(0, self.display_height)
        self.lower_roi_slider.setValue(self.roi_lower)
        self.lower_roi_slider.valueChanged.connect(self.update_lower_roi)
        self.lower_roi_value_label.setText(str(self.roi_lower))
        
        self.col_roi_slider = QSlider(Qt.Horizontal, self)
        self.col_roi_slider.setRange(0, self.display_width)
        self.col_roi_slider.setValue(self.roi_col)
        self.col_roi_slider.valueChanged.connect(self.update_col_roi)
        self.col_roi_value_label.setText(str(self.roi_col))
        
        self.upper_std_slider = QSlider(Qt.Horizontal, self)
        self.upper_std_slider.setRange(0, self.display_height)
        self.upper_std_slider.setValue(self.upper_std_line)
        self.upper_std_slider.valueChanged.connect(self.update_upper_std)
        self.upper_std_value_label.setText(str(self.upper_std_line))
        
        self.lower_std_slider = QSlider(Qt.Horizontal, self)
        self.lower_std_slider.setRange(0, self.display_height)
        self.lower_std_slider.setValue(self.lower_std_line)
        self.lower_std_slider.valueChanged.connect(self.update_lower_std)
        self.lower_std_value_label.setText(str(self.lower_std_line))
        
        
        ##### Buttons #####
        # Update slider's value when button is clicked by lambda function.
            # (It is not recommended to using lambda function in argument.)
        self.upper_roi_slider_l_btn = QPushButton('-', self)
        self.upper_roi_slider_l_btn.clicked.connect(lambda: self.upper_roi_slider.setValue(self.upper_roi_slider.value() - 1))
        self.upper_roi_slider_r_btn = QPushButton('+', self)
        self.upper_roi_slider_r_btn.clicked.connect(lambda: self.upper_roi_slider.setValue(self.upper_roi_slider.value() + 1))
        self.lower_roi_slider_l_btn = QPushButton('-', self)
        self.lower_roi_slider_l_btn.clicked.connect(lambda: self.lower_roi_slider.setValue(self.lower_roi_slider.value() - 1))
        self.lower_roi_slider_r_btn = QPushButton('+', self)
        self.lower_roi_slider_r_btn.clicked.connect(lambda: self.lower_roi_slider.setValue(self.lower_roi_slider.value() + 1))
        self.col_roi_slider_l_btn   = QPushButton('-', self)
        self.col_roi_slider_l_btn.clicked.connect(lambda: self.col_roi_slider.setValue(self.col_roi_slider.value() - 1))
        self.col_roi_slider_r_btn   = QPushButton('+', self)
        self.col_roi_slider_r_btn.clicked.connect(lambda: self.col_roi_slider.setValue(self.col_roi_slider.value() + 1))
        self.upper_std_slider_l_btn = QPushButton('-', self)
        self.upper_std_slider_l_btn.clicked.connect(lambda: self.upper_std_slider.setValue(self.upper_std_slider.value() - 1))
        self.upper_std_slider_r_btn = QPushButton('+', self)
        self.upper_std_slider_r_btn.clicked.connect(lambda: self.upper_std_slider.setValue(self.upper_std_slider.value() + 1))
        self.lower_std_slider_l_btn = QPushButton('-', self)
        self.lower_std_slider_l_btn.clicked.connect(lambda: self.lower_std_slider.setValue(self.lower_std_slider.value() - 1))
        self.lower_std_slider_r_btn = QPushButton('+', self)
        self.lower_std_slider_r_btn.clicked.connect(lambda: self.lower_std_slider.setValue(self.lower_std_slider.value() + 1))
        
        self.auto_setting_btn   = QPushButton('자동설정', self)
        self.auto_setting_btn.clicked.connect(self.set_screen_auto)
        self.manual_setting_btn = QPushButton('수동설정', self)
        self.manual_setting_btn.clicked.connect(self.set_screen_manual)
        self.sys_start          = QPushButton('솔루션 작동', self)
        self.sys_start.clicked.connect(self.do_solve)
        self.sys_quit_btn       = QPushButton('시스템 종료', self)
        self.sys_quit_btn.clicked.connect(self.quit_program)
        
        
        ##### Set widgets in grid layout. #####
        # Last 4 arguments indicate it's position and width, height.
        # [widget 1] :: System information.
        grid_layout.addWidget(self.sys_info_label,          0,  0,  1,  2)
        # [widget 2] :: OpenCV imshow realtime image.
        grid_layout.addWidget(self.camera_img_label,        1,  0,  9,  2)
        # [widget 3] :: Result information.
        grid_layout.addWidget(self.sys_result_label,        10, 0,  2,  2)
        # [widget 4] :: Upper ROI boundary line.
        grid_layout.addWidget(self.upper_roi_label,         0,  2,  1,  8)
        grid_layout.addWidget(self.upper_roi_slider,        1,  2,  1,  5)
        grid_layout.addWidget(self.upper_roi_slider_l_btn,  1,  7,  1,  1)
        grid_layout.addWidget(self.upper_roi_value_label,   1,  8,  1,  1)
        grid_layout.addWidget(self.upper_roi_slider_r_btn,  1,  9,  1,  1)
        # [widget 5] :: Lower ROI boundary line.
        grid_layout.addWidget(self.lower_roi_label,         2,  2,  1,  8)
        grid_layout.addWidget(self.lower_roi_slider,        3,  2,  1,  5)
        grid_layout.addWidget(self.lower_roi_slider_l_btn,  3,  7,  1,  1)
        grid_layout.addWidget(self.lower_roi_value_label,   3,  8,  1,  1)
        grid_layout.addWidget(self.lower_roi_slider_r_btn,  3,  9,  1,  1)
        # [widget 6] :: Left ROI boundary line.
        grid_layout.addWidget(self.col_roi_label,           4,  2,  1,  8)
        grid_layout.addWidget(self.col_roi_slider,          5,  2,  1,  5)
        grid_layout.addWidget(self.col_roi_slider_l_btn,    5,  7,  1,  1)
        grid_layout.addWidget(self.col_roi_value_label,     5,  8,  1,  1)
        grid_layout.addWidget(self.col_roi_slider_r_btn,    5,  9,  1,  1)
        # [widget 7] :: Upper standard line.
        grid_layout.addWidget(self.upper_std_label,         6,  2,  1,  8)
        grid_layout.addWidget(self.upper_std_slider,        7,  2,  1,  5)
        grid_layout.addWidget(self.upper_std_slider_l_btn,  7,  7,  1,  1)
        grid_layout.addWidget(self.upper_std_value_label,   7,  8,  1,  1)
        grid_layout.addWidget(self.upper_std_slider_r_btn,  7,  9,  1,  1)
        # [widget 8] :: Lower standard line.
        grid_layout.addWidget(self.lower_std_label,         8,  2,  1,  8)
        grid_layout.addWidget(self.lower_std_slider,        9,  2,  1,  5)
        grid_layout.addWidget(self.lower_std_slider_l_btn,  9,  7,  1,  1)
        grid_layout.addWidget(self.lower_std_value_label,   9,  8,  1,  1)
        grid_layout.addWidget(self.lower_std_slider_r_btn,  9,  9,  1,  1)
        # [widget 9] :: User control buttons.
        grid_layout.addWidget(self.auto_setting_btn,        10, 2,  1,  5)
        grid_layout.addWidget(self.manual_setting_btn,      10, 7,  1,  3)
        grid_layout.addWidget(self.sys_start,               11, 2,  1,  5)
        grid_layout.addWidget(self.sys_quit_btn,            11, 7,  1,  3)
        
        self.setWindowTitle('Lighter GUI Program')
        self.show()
    
    def update_system_info(self):
        """
            Update 'system_info' label's text in every 3 minutes.
            Updating this label is fair overhead enough.
            For user-friendliness, 3-step expression may be better, not percentage expression.
        """
        sys_info = "CPU 점유율: %d%%  메모리 점유율: %d%%  온도: %d°C\n" \
            % (self.get_CPU_info(), self.get_mem_info(), self.get_temp_info())
        sys_info += "시스템 현재 상태: "
        sys_info += "환경설정 중" if self.system_state is State.SETTING else "작동 중"
        self.sys_info_label.setText(sys_info)
    
    def change_system_state(self):
        """
            Change 'system_state'.
        """
        self.system_state = State.SOLVING if self.system_state == State.SETTING else State.SETTING
    
    def update_camera_img(self):
        """
            Get camera image and display within GUI properly in every 0.1second (10FPS).
        """
        img = Yolo.yolo_loop(self)
        h, w, c = img.shape
        pixmap = QPixmap.fromImage(QImage(img.data, w, h, w * c, QImage.Format_RGB888)).scaledToWidth(512)
        self.camera_img_label.setPixmap(pixmap)
    
    """
        Update and show slider's value in each label.
    """
    def update_upper_roi(self):
        value = self.upper_roi_slider.value()
        self.roi_upper = value
        self.upper_roi_value_label.setText(str(value))
    
    def update_lower_roi(self):
        value = self.lower_roi_slider.value()
        self.roi_lower = value
        self.lower_roi_value_label.setText(str(value))
        
    def update_col_roi(self):
        value = self.col_roi_slider.value()
        self.roi_col = value
        self.col_roi_value_label.setText(str(value))
    
    def update_upper_std(self):
        value = self.upper_std_slider.value()
        self.std_upper = value
        self.upper_std_value_label.setText(str(value))
    
    def update_lower_std(self):
        value = self.lower_std_slider.value()
        self.std_lower = value
        self.lower_std_value_label.setText(str(value))
    
    def show_result(self):
        """
        Show defect detection result through GUI.
        """
        result = ""
        for num, value in enumerate(self.lighter_error):
            if value == 1: result += (str(num + 1) + "번 라이터 미달 / ")
            if value == 2: result += (str(num + 1) + "번 라이터 초과 / ")
        self.sys_result_label.setText(result)
    
    def do_solve(self):
        """
        Call Yolo.solve() function to judge defection.
        Constant '5' can be modified if you want.
        """
        self.solve(5)
        self.show_result()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LighterApp()
    sys.exit(app.exec_())