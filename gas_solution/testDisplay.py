import cv2
import numpy as np
from csi_camera import CSI_Camera

# 1280 x 960    (4:3)
# 1400 x 1050   (4:3)
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

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
    cv2.namedWindow("Gas Solution", cv2.WINDOW_AUTOSIZE)
    
    if not camera.video_capture.isOpened():
        print("Unable to open camera")
        SystemExit(0)
    
    try:
        camera.start_counting_fps()
        while cv2.getWindowProperty("Gas Solution", 0) >= 0:
            _, img_ori = camera.read()

            temp = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)

            img = np.zeros([DISPLAY_HEIGHT, DISPLAY_HEIGHT, 3])
            
            diff_half = (DISPLAY_WIDTH - DISPLAY_HEIGHT) // 2
            for i in range(3) :
               img[:,:,i] = temp[:,diff_half:DISPLAY_WIDTH-diff_half]
            img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5, scale=0.15)
            img = cv2.convertScaleAbs(img)
            
            cv2.imshow("TEST_DISPLAY", img)
            camera.frames_displayed += 1
            if (cv2.waitKey(5) & 0xFF) == 27: break
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    getCapture()
    