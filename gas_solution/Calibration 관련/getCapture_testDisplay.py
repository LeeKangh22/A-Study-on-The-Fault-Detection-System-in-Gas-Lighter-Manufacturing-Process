import cv2
import numpy as np
import time
from csi_camera import CSI_Camera

# 1280 x 960    (4:3)
# 1400 x 1050   (4:3)
DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = 1050

def getCapture() :   # 실시간으로 화면을 캡쳐 후 로컬저장함
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
        sensor_id = 0,
        sensor_mode = 0,
        framerate = 30,
        flip_method = 0,
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
            _, img = camera.read()
            cv2.imshow("DISPLAY", img)
            camera.frames_displayed += 1
            if (cv2.waitKey(5) & 0xFF) == 27: break
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    getCapture()
    