import cv2
import numpy as np
import time
import pickle
from csi_camera import CSI_Camera

# 1280 x 960    (4:3)
# 1400 x 1050   (4:3)
DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = 1050
mtx = []
dist = []


def load_calibration():
    mtx = []
    dist = []
    with open("C:\\Users\\cydph\\Desktop\\cali_images\\test.txt", "r") as file:
        for line in file.readlines():
            mtx = list(map(float, line.strip().split(',')))

    with open("C:\\Users\\cydph\\Desktop\\cali_images\\test2.txt", "r") as file:
        for line in file.readlines():
            dist.append(list(map(float, line.strip().split(','))))

    return mtx, dist


def get_capture() :
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
        sensor_id = 0,
        sensor_mode = 0,
        framerate = 30,
        flip_method = 2,
        display_height = DISPLAY_HEIGHT,
        display_width = DISPLAY_WIDTH
    )
    camera.open(camera.gstreamer_pipeline)
    camera.start()
    cv2.namedWindow("DISPLAY", cv2.WINDOW_AUTOSIZE)
    
    if not camera.video_capture.isOpened():
        print("Unable to open camera")
        SystemExit(0)
    
    try:
        camera.start_counting_fps()
        while cv2.getWindowProperty("DISPLAY", 0) >= 0:
            ret, img = camera.read()
            print("ret : ",ret)
            img_undist = cv2.undistort(img, mtx, dist, None, mtx)
            cv2.imshow("DISPLAY", img_undist)
            camera.frames_displayed += 1
            if (cv2.waitKey(5) & 0xFF) == 27: break
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    mtx, dist = load_calibration()
    get_capture()
    