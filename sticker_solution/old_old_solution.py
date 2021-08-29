import cv2
import numpy as np
import os
import io
import time
import multiprocessing
import copy
from csi_camera import CSI_Camera

low_threshold = 0
high_threshold = 150
rho = 1                 # distance resolution in pixels of the Hough grid
theta = np.pi / 180     # angular resolution in radians of the Hough grid
threshold = 200         # minimum number of votes (intersections in Hough grid cell)
max_line_gap = 20       # maximum gap in pixels between connectable line segments

def get_interest(img) : # 라이터 위치를 찾기 위한 이미지의 절반을 흑백 처리 함수
    img[0:360, :] = 0
    return img

def checkRawRatio(candidate) :  # 라이터 헤드 아랫부분 기준을 찾기 위한 좌표 추정 함수
    return int(candidate * (274/490))   #헤드 아랫 부분/라이터 고정대 여기 사이즈 바꿔야 함

def checkStickRatio(raw) :
    return int(raw * (490/274))

def checkHeadRatio(raw, stick) :    # 라이터 헤드를 찾기 위한 좌표 추정 함수
    return int((stick-raw) * (5/105))    #헤드 간 간격/몸통 길이

def checkStickerRatio(raw, stick) :
    return int((stick-raw)*(45/216)), int((stick-raw)*(133/216))

def checkStickerRatio2(raw, stick) :
    return int((stick-raw)*(134/216)), int((stick-raw)*(175/216))

def findRaw(img) :  # 라이터 고정대 좌표를 찾기 위한 함수
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = get_interest(gray)
    kernel_size = 5

    for i in range(1) :       # 시작 전 여기 수정
        gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    min_line_length = int(img.shape[0]*0.4)  # minimum number of pixels making up a line

    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                        min_line_length, max_line_gap)

    candidate = []
    if lines is not None :
        for line in lines:
            for x1,y1,x2,y2 in line:
                if y1 > img.shape[0]*0.55  :   #여기 사이즈 수정
                    candidate.append([y1, y2])

    if candidate :
        candidate.sort(reverse=True, key = lambda x : x[0])
        return checkRawRatio(candidate[0][0]), candidate[0][0]
    else :
        return -1, -1

def getCapture(cap) :   # 반복적으로 화면 캡쳐를 얻는 함수
    # 로컬에 화면 캡쳐 이미지를 저장함
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
        sensor_id = 0,
        sensor_mode = 2,
        framerate = 30,
        flip_method = 0,
        display_height = 720,
        display_width = 1280
    )
    camera.open(camera.gstreamer_pipeline)
    camera.start()
    cv2.namedWindow("Sticker Solution", cv2.WINDOW_AUTOSIZE)
    
    if not camera.video_capture.isOpened():
        print("Unable to open camera")
        SystemExit(0)
    
    try:
        camera.start_counting_fps()
        while cv2.getWindowProperty("Sticker Solution", 0) >= 0:
            _, img = camera.read()
            cv2.imwrite("images/"+str(cap)+".jpg", img[:,280:1000,:])
            cv2.imshow("Sticker Solution", img[:,280:1000,:])
            time.sleep(0.25)        #시작 전 여기 수정
            camera.frames_displayed += 1
            cap = cap + 1
            if (cv2.waitKey(5) & 0xFF) == 27: break
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()

def yolo(cap) :     # 로컬에 저장된 화면 캡쳐를 불러와 라이터의 스티커 불량 여부를 확인하는 함수
    # 인식이 완료된 화면 캡쳐는 삭제 됨
    raw = 274
    stick = 490
    net = cv2.dnn.readNet("yolov3-tiny_4000.weights", "yolov3-tiny.cfg")    # 학습 모델을 불러옴
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    classes = ["nomal_head", "shake_head"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    stickers = []
    stickers_error = []
    for i in range(11) :    # 스티커 불량 여부를 판정하기 위한(템플릿 매칭에 사용될) 기준 스티커를 불러옴
        if os.path.isfile('nnum'+str(i)+'.JPG') :
            stickers.append(cv2.imread('nnum'+str(i)+'.JPG'))
    for k in range(4) :
        if os.path.isfile('error'+str(k)+'.JPG') :
            stickers_error.append(cv2.imread('error'+str(k)+'.JPG'))

    prev = time.time()
    while True :
        if os.path.isfile("images/"+str(cap)+".jpg") :      # 로컬에 저장된 화면 캡쳐를 불러옴
            img = cv2.imread("images/"+str(cap)+".jpg")
            sub_img = copy.deepcopy(img)
            try :

                #-----라이터 헤드를 찾고 헤드를 기준으로 바디를 추정-----#

                blob = cv2.dnn.blobFromImage(img, 0.00392, (448, 448), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                confidences = []
                boxes = []
                boxes_low = -1

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5 and class_id == 0 :
                            # Object detected
                            center_x = int(detection[0] * 720)
                            center_y = int(detection[1] * 720)
                            w = int(detection[2] * 720)
                            h = int(detection[3] * 720)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            # 바디를 학습시키지 않는다는 가정 하에
                            #if y+h < raw * 1.05  :         # 시작 전 여기 수정
                            boxes.append([x, y, w, h])
                            boxes_low = max(boxes_low, y+h)
                            confidences.append(float(confidence))

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

                # 인식된 라이터가 다섯개 미만이면 화면 캡쳐가 흔들린 것 혹은 라이터가 아래로 내려간 상태인 것으로 간주
                if len(boxes) < 7 :
                    cv2.imshow("window", sub_img)
                    if (cv2.waitKey(5) & 0xFF) == 27: break
                    os.remove("images/"+str(cap)+".jpg")
                    cap += 1
                    prev = time.time()
                    continue

                temp = boxes_low
                temp2 = checkStickRatio(temp)
                if temp2 > 0 :
                    # if 0.9*stick < temp2 < 1.1*stick :
                    stick = temp2
                if temp > 0 :
                    # if 0.9*raw < temp < 1.1*raw : 
                    raw = temp

                boxes.sort(key=lambda x : x[0])
                #-----라이터 헤드가 가려진 경우 라이터 헤드의 위치와 그에 따른 바디 위치를 임의로 추정-----#
                between = checkHeadRatio(raw, stick)

                error_region = []
                error_minVal = 1
                start = boxes[0][0]-between if boxes[0][0]-between >= 0 else 0
                end = boxes[-1][0]+boxes[-1][2] + between if boxes[-1][0]+boxes[-1][2] + between < 720 else 719

                cut_img = img[raw:stick, start:end]
                for error in stickers_error :
                    error_temp = copy.deepcopy(error)
                    if stick-raw <= error.shape[0] :
                        resize_width, resize_height = checkStickerRatio2(raw, stick)
                        error_temp = cv2.resize(error, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
                    result = cv2.matchTemplate(cut_img, error_temp, cv2.TM_SQDIFF_NORMED)
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
                    x, y = minLoc
                    h, w, c = error_temp.shape

                    if minVal < 0.09 and error_minVal > minVal :
                        if error_region : error_region[0] = [start+x, raw+y, w, h, minVal]
                        else : error_region.append([start+x, raw+y, w, h, minVal])
                if error_region :
                    cv2.rectangle(sub_img, (error_region[0][0], error_region[0][1]), (error_region[0][0]+error_region[0][2], error_region[0][1]+error_region[0][2]), (0, 255, 255), 2, cv2.LINE_8)

                results = []
                for index in boxes :
                    start = index[0]-between if index[0]-between >= 0 else 0
                    end = index[0]+index[2]+between if index[0]+index[2]+between < 720 else 719 #여기 사이즈 바꿔야함
                    resize_width, resize_height = checkStickerRatio(raw, stick)

                    cut_img = img[raw:stick, start:end]
                    # cv2.rectangle(sub_img, (start,raw), (end, stick), (255, 255, 0), 2, cv2.LINE_8)

                    resul = []
                    for sticker in stickers :
                        sticker_temp = copy.deepcopy(sticker)
                        #-----이거 나중에 높이 대비 스티커 길이 비율로 수정하기-----#
                        sticker_temp = cv2.resize(sticker, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
                        result = cv2.matchTemplate(cut_img, sticker_temp, cv2.TM_SQDIFF_NORMED)

                        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
                        x, y = minLoc
                        h, w, c = sticker_temp.shape
                        resul.append([start+x, raw+y, w, h, minVal])
                    resul.sort(key = lambda x : x[4])
                    if resul[0][4] <= 0.10 :
                        if error_region :
                            if error_region[0][0] < resul[0][0]+resul[0][2] < error_region[0][0]+error_region[0][2] or error_region[0][0] < resul[0][0] < error_region[0][0]+error_region[0][2] :
                                continue
                        results.append(resul[0])

                #-----불량이 있을 경우 불량임을 알린다-----#
                if error_region or len(results) < 10 : 
                    cv2.rectangle(sub_img, (4, 4), (716, 716), (0, 0, 255), 8, cv2.LINE_8)
                # for index in boxes :
                #     cv2.rectangle(sub_img, (index[0], index[1]), (index[0]+index[2], index[1]+index[3]), (255, 0, 0), 2, cv2.LINE_8)

                if len(boxes) < 10 :
                    first = boxes[0]
                    last = boxes[-1]
                    i = 0
                    while i < len(boxes)-1 :
                        if boxes[i+1][0] - boxes[i][0] + boxes[i][2] > between :
                            numOfTempLighter = (boxes[i+1][0] - (boxes[i][0] + boxes[i][2] + between)) // (between + boxes[i][2])
                            if numOfTempLighter > 0 :
                                for k in range(numOfTempLighter) :
                                    boxes.append([(boxes[i][0] + boxes[i][2])+(k+1)*between+k*boxes[i][2], boxes[i][1], boxes[i][2], boxes[i][3]])
                                i += numOfTempLighter + 1
                                continue
                        i += 1
                    if len(boxes) < 10 :
                        num = 10-len(boxes)
                        for k in range(0, num) :
                            if first[0] - (k+1)*(between + first[2]) < 0 : break
                            boxes.append([first[0] - (k+1)*(between + first[2]), first[1], first[2], first[3]])
                            # cv2.rectangle(sub_img, (first[0] - (k+1)*(between + first[2]), first[1]), (first[0] - (k+1)*(between + first[2])+first[2], first[1]+first[3]), (100, 0, 200), 2, cv2.LINE_8)
                            cv2.rectangle(sub_img, (first[0] - (k+1)*(between + first[2]),raw), (first[0] - (k+1)*(between + first[2])+between+first[2], stick), (0, 155, 255), 2, cv2.LINE_8)
                        for k in range(0, num) :
                            if (last[0] + last[2])+(k+1)*between+k*last[2] > 719 : break   # 여기 사이즈 수정, width 확인
                            boxes.append([(last[0] + last[2])+(k+1)*between+k*last[2], last[1], last[2], last[3]])
                            # cv2.rectangle(sub_img, ((last[0] + last[2])+(k+1)*between+k*last[2], last[1]), ((last[0] + last[2])+(k+1)*between+k*last[2]+last[2], last[1]+last[3]), (100, 0, 200), 2, cv2.LINE_8)
                            cv2.rectangle(sub_img, ((last[0] + last[2])+(k+1)*between+k*last[2]-between,raw), ((last[0] + last[2])+(k+1)*between+k*last[2]+last[2], stick), (0, 155, 255), 2, cv2.LINE_8)

                #-----확인한 불량 여부 가능성과 스티커 위치에 박스 표시-----#

                for i, index in enumerate(results) :
                    cv2.rectangle(sub_img, (index[0], index[1]), (index[0]+index[2], index[1]+index[3]), (255, 255, 0), 2, cv2.LINE_8)
                cv2.imshow("window", sub_img)
                if (cv2.waitKey(5) & 0xFF) == 27: break

                # 처리가 끝난 이미지는 무조건 삭제
                os.remove("images/"+str(cap)+".jpg")
                cap += 1
                prev = time.time()

            except Exception as e :
                if str(e) == "OpenCV(4.2.0) /home/hyojung/opencv/opencv-4.2.0/modules/imgproc/src/templmatch.cpp:1109: error: (-215:Assertion failed) _img.size().height <= _templ.size().height && _img.size().width <= _templ.size().width in function 'matchTemplate'\n" :
                    cv2.rectangle(sub_img, (4, 4), (716, 716), (0, 255, 255), 8, cv2.LINE_8)
                    os.remove("images/"+str(cap)+".jpg")
                    cap += 1
                if str(e) == "OpenCV(4.2.0) /home/hyojung/opencv/opencv-4.2.0/modules/imgproc/src/resize.cpp:4045: error: (-215:Assertion failed) !ssize.empty() in function 'resize'\n" :
                   time.sleep(0.2)
                print(str(e))
            
        else :      # 10초 이상 화면 캡쳐가 추가되지 않으면 종료
            if time.time() - prev > 10 :
                return
            else :
                pass
        
if __name__ == '__main__' :
    cap = 0
    proc1 = multiprocessing.Process(target=getCapture, args=(cap,))
    proc1.start()
    proc2 = multiprocessing.Process(target=yolo, args=(cap,))
    proc2.start()
    
    proc1.join()
    proc2.join()
