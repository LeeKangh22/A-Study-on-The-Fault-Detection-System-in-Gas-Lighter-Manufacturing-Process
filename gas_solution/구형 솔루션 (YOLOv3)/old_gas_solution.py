import cv2
import numpy as np
import os
import io
import time
import multiprocessing
from csi_camera import CSI_Camera
import copy

low_threshold = 0
high_threshold = 150
rho = 1                 # distance resolution in pixels of the Hough grid
theta = np.pi / 180     # angular resolution in radians of the Hough grid
threshold = 200         # minimum number of votes (intersections in Hough grid cell)
max_line_gap = 20       # maximum gap in pixels between connectable line segments

DISPLAY_HEIGHT  = 1280  # 효정이의 하드코드들 전부 바꾸기 위해 전역변수로 옮김
DISPLAY_WIDTH   = 720

class Yolo_Line() :
    def __init__(self) :
        self.net = cv2.dnn.readNet("yolov3-tiny_3000.weights", "yolov3-tiny.cfg")    # 학습 모델을 불러옴
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = ["back", "front"]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detect_lines(self, img) :
        blob = cv2.dnn.blobFromImage(img, 0.00392, (448, 448), (104.0, 177.0, 123.0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        confidences_back    = []    # Back line에 대한 confidence를 저장할 리스트
        confidences_front   = []    # Front line에 대한 confidence를 저장할 리스트
        boxes_back          = []    # Back line에 칠 NMS 박스를 저장할 리스트
        boxes_front         = []    # Front line에 칠 NMS 박스를 저장할 리스트

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3 :
                    # Object detected
                    center_x = int(detection[0] * 720)
                    center_y = int(detection[1] * 720)
                    w = int(detection[2] * 720)
                    h = int(detection[3] * 720)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    if class_id :   # Class_id == 1, front line의 경우
                        boxes_front.append([x, y, w, h, class_id])
                        confidences_front.append(float(confidence))
                    else :          # Class_id == 0, back line의 경우
                        boxes_back.append([x, y, w, h, class_id])
                        confidences_back.append(float(confidence))
                        
        cv2.dnn.NMSBoxes(boxes_front, confidences_front, 0.3, 0.2)
        cv2.dnn.NMSBoxes(boxes_back, confidences_back, 0.3, 0.2)
        boxes_back.sort(key=lambda x : x[0])    # 수면을 x축 좌표에 대해서 정렬함!
        .
        .sort(key=lambda x : x[0])   # 수면을 x축 좌표에 대해서 정렬함!
        return boxes_back, boxes_front  # 후방 수면, 전방 수면 

def get_interest(img, k) : # 라이터 위치를 찾기 위한 이미지의 절반을 흑백 처리 함수
    if k : img[0:360, :] = 0
    else : img[360:, :] = 0
    return img

def checkHeadRatio(raw, stick) :        # 라이터 헤드 사이 간격을 알기 위한 좌표 추정 함수
    # 아마 205는 라이터 2개의 가로길이, 50은 라이터 1개의 가로길이의 절반을 의미하는듯?
    return int(((stick-raw) * (50/205))/2)

def checkBetweenRatio(raw, stick) :
    return int((stick-raw) * (30/105))

def checkLineRatio(stick, raw) :
    height = stick - raw
    return raw + int(height * (12/40)), raw + int(height * (19.5/40)), raw + int(height * (32/40))
    #return raw + int(height * (12/39)), raw + int(height * (22/39)), raw + int(height * (32/39))

def findRaw(img) :  # 라이터 고정대 좌표를 찾기 위한 함수
    result = []

    for k in range(2) :         # 0 : 상위 기준선 찾기, 1 : 하위 기준선(트레이) 찾기
        gray = get_interest(copy.deepcopy(img), k)  # 절반이 흑백이 된 이미지를 얻는다.
        gray = cv2.GaussianBlur(gray, (5, 5), 0)    # kernel_size = 5
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        min_line_length = 0     # [TODO] minimum number of pixels making up a line -> 설정하는게 좋을듯???

        candidate = []
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        if lines is not None :
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if 355 * (1 + k)> y1 > 365 * k : candidate.append([y1, y2]) # [TODO] 이 부분 맘에 안듦. 수정 예정. 
					# k가 0이면(상위) 0 < y1 < 355
        if candidate :  # 기준선 후보가 하나라도 있다면
            candidate.sort(key = lambda x : x[0])       # y1 좌표에 대해서 정렬함
            if k : result.append(candidate[0][0])       # 상위 기준선일 때는 가장 작은 y1 좌표
            else : result.append(candidate[0][0] + 8)   # 하위 기준선일 때는 가장 작은 y1 좌표 + 8...? 8은 어디서 나온 수???
        else : result.append(-1)
    return result[0], result[1]     # 상단 기준선 y좌표, 하단 기준선 y좌표 반환.


def getCapture(cap) :   # 실시간으로 화면을 캡쳐 후 로컬저장함
    camera = CSI_Camera() 
    camera.create_gstreamer_pipeline(
        sensor_id = 0,
        sensor_mode = 0,        # [TODO] Sensor_mode로 설정하지 말고 해상도로 설정하도록 해야할 것 같은데? 4032x3040 (Ratio: 4:3)
        framerate = 30,
        flip_method = 0,
        display_height = 720,   # [TODO] 이것도 4:3으로 맞춰주는게 가장 best인데...
        display_width = 1280
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

            img = np.zeros([720, 720, 3])   # [TODO] 이런거 다 전역변수로 바꿔야 함

            for i in range(3) :
               img[:,:,i] = temp[:,280:1000] # [TODO] 이런거 다 전역변수로 바꿔야 함
            img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5, scale=0.15) # ksize랑 scale 다시 설정해보기 (카메라 바꼈으니)
            img = cv2.convertScaleAbs(img)
            # Sobel 먹인 뒤에 sharpening도 먹이는 거 잊지 말기

            cv2.imwrite("images/"+str(cap)+".jpg", img)     # [중요!] 이미지를 로컬에 저장함! TODO. 이거 저장하지말고 바로 판단하도록 바꿔야함
            #cv2.imshow("Gas Solution", img)                # [TODO]  이걸 굳이 띄울 필요가 있나?
            time.sleep(0.33)                                # 의도적으로 3프레임으로 만들려고 0.33초 sleep
            camera.frames_displayed += 1
            cap = cap + 1
            if (cv2.waitKey(5) & 0xFF) == 27: break
    finally:
        camera.stop()
        camera.release()
        cv2.destroyAllWindows()


def yolo(cap) :     # 로컬에서 캡쳐 이미지를 불러와 불량 여부를 확인하고 이미지는 삭제함.
    raw = 0
    yolo_line = Yolo_Line()     # [중요!] 이거...한 번 실행되는 거 맞겠지?
    prev = time.time()          # 현재 시간 기록
    
    while True :
        if os.path.isfile("images/"+str(cap)+".jpg") :      
            img = cv2.imread("images/"+str(cap)+".jpg") # 로컬에 저장된 화면 캡쳐를 불러옴
            if img is None :        # 불러올 이미지가 없다면 0.3초 동안 다음 사진 기다림
                time.sleep(0.33)
                continue
            sub_img = copy.deepcopy(img)    # 불러온 이미지를 sub_img로 deepcopy 
            try :                           # [TODO] 제발 temp, raw, stick 이 정신나간 변수 네이밍 센스 바꾸기;;
                temp, stick = findRaw(img)  # temp(=raw): 상단 기준선 y좌표, stick: 하단 기준선 y좌표
                if temp is not None and temp > 0 : raw = temp   # 이제 raw는 temp임. temp 쓴 이유는 아래 logic 구현하기 위해서 인듯?
                    # [TODO] 상단 기준선이 위아래로 10% 이상 변동됐다면 카메라가 물리적으로 움직인 것임.
                    # if 0.9 * raw > temp or temp > 1.1 * raw : print("카메라가 움직임: from 상단 기준선")
                    # if 0.9 * 'prevStick' > stick or stick > 1.1 * 'prevStick' : print("카메라가 움직임: from 하단 기준선")                    

                # 각 영역의 기준선 찾기 ('상' 영역, '중' 영역, '하' 영역의 최하단선)
                h_line, m_line, l_line = checkLineRatio(stick, raw)
                # cv2.line 함수로 해당 좌표에 직선을 그림. (일종의 오버헤드로 작용할 가능성 있음)
                cv2.line(sub_img, (0, raw),     (720, raw),     (255,255,0),    1)
                cv2.line(sub_img, (0, stick),   (720, stick),   (255,255,0),    1)
                cv2.line(sub_img, (0, h_line),  (720, h_line),  (0,255,0),      1)
                cv2.line(sub_img, (0, m_line),  (720, m_line),  (0,255,0),      1)
                cv2.line(sub_img, (0, l_line),  (720, l_line),  (0,255,0),      1)

                boxes_back, boxes_front = yolo_line.detect_lines(img)   # 후방, 전방 수면 인식하고 반환받아온다. (type: list)
                
                # 인식한 수면들 전부 합쳤더니 7개 미만이거나 아예 없거나 temp(상단 기준선 y좌표)가 None이라면
                # [TODO] 이거 경계조건 왜 이따위로 설정함...? 이유가 있을거같은데 되게 지저분하네... 7개로 설정한 근거는?
                if len(boxes_back) + len(boxes_front) < 7 or (boxes_back is None and boxes_front is None) or temp is None:
                    cv2.imshow("window", sub_img)
                    if (cv2.waitKey(5) & 0xFF) == 27: break
                    # [TODO] [예외처리]: 해당 이미지 삭제하고 그냥 무시하네? 적절한 예외처리 새롭게 해줘야 할 것 같다.
                    os.remove("images/"+str(cap)+".jpg") 
                    cap += 1
                    prev = time.time()  # [TODO] 이거 굳이 필요함? continue 하면 어차피 while True문 바로 밑에 똑같은거 있는데?
                    continue
                
                lighter_num     = 0
                between         = checkHeadRatio(raw, stick)        # return int(((stick - raw) * (50 / 205)) / 2)
                head_between    = checkBetweenRatio(raw, stick)     # return int((stick - raw) * (30 / 105))
                last            = -1e9
                back_result     = -1
                is_nomal        = True
                line_boxes      = []
                next_front      = 0
                
                # [주의] 일단 이 for문 정말 잘 이해 안됨. 아예 logic 자체를 뜯어 고쳐야겠음.
                for idx, box_back in enumerate(boxes_back) :
                    is_done = False
                    center_x = (box_back[0] + box_back[2]) // 2     # 무게중심의 x좌표
                    while (boxes_front[next_front][0] + boxes_front[next_front][2]) // 2 < center_x - between :     # ?? (인식된 front line 개수만큼 loop)
                        line_boxes.append(boxes_front[next_front])  # line box list에 front line 정보 넣고
                        next_front += 1                             # 다음 번 front line으로
                        if next_front >= len(boxes_front) :         # front line 전부 append 했다면,
                            line_boxes.extend(boxes_back[idx:])     # 이 부분이 잘 이해가 안감. idx번부터 끝까지 back을 그냥 넣는건가?
                            is_done = True                          # done flag TRUE 해준다.
                            break
                    if is_done : break  # [here]로 이동
                    
                    line_boxes.append(box_back)
                    if center_x - between <= boxes_front[next_front][0] + boxes_front[next_front][2] // 2 <= center_x + between :
                        line_boxes.append(boxes_front[next_front])
                        next_front += 1
                        if next_front >= len(boxes_front) :
                            if idx + 1 < len(boxes_back) : line_boxes.extend(boxes_back[idx + 1:])
                            break
                            
                # [here] is_done : break 됐다면 여기로 이동한다.
                if next_front < len(boxes_front) : line_boxes.extend(boxes_front[next_front:])
                
                for line_box in line_boxes :    # 인식한 모든 수면에 대해 (front가 먼저, back은 그 뒤에 그려진다.)
                    # line_box[4] = class_id, if문 (front_line, cyan box), else문 (back_line, yellow box)
                    # LINE_4와 LINE_8은 계단 현상이 발생하지만, LINE_AA는 계단 현상이 발생하지 않는다. (but overhead 발생하므로 LINE_4 써도 될듯?)
                    if line_box[4] : cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (255, 255, 0), 2, cv2.LINE_8)
                    else : cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (0, 255, 255), 2, cv2.LINE_8)
                    center_x = line_box[0] + line_box[2] // 2   # 무게중심의 x좌표
                    center_y = line_box[1] + line_box[3] // 2   # 무게중심의 y좌표
                    
                    def checkPlace(center_y) :      # return 상 = 0, 중 = 1, 하 = 2
                        if center_y <= m_line : return 0
                        if m_line < center_y < l_line : return 1
                        if l_line <= center_y : return 2
                        return -1
                    
                    # 가장 왼쪽 front line의 무게중심 x좌표가 valid 하다면 (왜 이게 valid 조건인지는 모르겠음)
                    if last-between <= center_x <= last+between :
                        now_result = checkPlace(center_y)   
                        if back_result == 0 and now_result == 0 :       # 상+상 이면 초과인 경우니까
                            print(lighter_num, "번 라이터 가스량 초과")
                            cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (187,67, 246), 2, cv2.LINE_8)
                            is_nomal = False
                        if back_result == 2 and now_result == 2 :       # 하+하 면 미달인 경우니까
                            print(lighter_num, "번 라이터 가스량 미달")
                            cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (187,67, 246), 2, cv2.LINE_8)
                            is_nomal = False
                        back_result = -1    # back_result를 초기화해준다.

                    else :
                        lighter_num += 1

                        if back_result == 0 :   # back이 '상'에 있다면,
                            print(lighter_num - 1, "번 라이터 가스량 초과")
                            #cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (187,67, 246), 2, cv2.LINE_8)

                        if last > 0 and (center_x - last) // head_between >= 2 :
                            for t in range(1, (center_x - last) // head_between) :
                                print(">>>", lighter_num, "번 라이터 가스량 완전 미달<<<")
                                is_nomal = False
                                lighter_num += 1

                        back_result = -1
                        if line_box[4] :    # front만 인식됨
                            #if center_y > m_line + (stick-m_line) * (3/17) :
                            if l_line <= center_y :
                                print(lighter_num, "번 라이터 가스량 미달")
                                cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (187,67,246), 2, cv2.LINE_8)
                                is_nomal = False
                            if h_line <= center_y < m_line :
                                print(lighter_num, "번 라이터 가스량 초과")
                                cv2.rectangle(sub_img, (line_box[0],line_box[1]), (line_box[0]+line_box[2], line_box[1]+line_box[3]), (187,67,246), 2, cv2.LINE_8)
                                is_nomal = False
                        else :
                            back_result = checkPlace(center_y)

                    last = center_x # last = 직전의 center_x

                if back_result == 0 : print(lighter_num, "번 라이터 가스량 초과")
                if not is_nomal : cv2.rectangle(sub_img, (4, 4), (716, 716), (0, 0, 255), 8, cv2.LINE_8)    # 불량품 발견 시 빨간색 테두리 사각형 등장.
                #if lighter_num < 10 : print('가스가 완전히 미달인 라이터 존재')

                # for head_box in head_boxes :
                #     results = []
                    
                #     # 선의 x축 무게중심이 헤더 안에 위치하는지 확인하여 라이터 위치 식별
                #     while head_box[0] <= line_boxes[t][0] + line_boxes[t][2]//2 <= head_box[0]+head_box[2] :
                #         results.append(line_boxes[t])
                #         t += 1
                    
                #     if len(results) == 1 : 
                #         if not results[0][4] : print("인식 오류, back만 인식되었습니다.")
                #         else :
                #             center_y = results[0][1] + results[0][3]//2
                #             if center_y > m_line + (m_line+stick) * (3/17) : print(lighter_num, "번 라이터 가스량 미달")

                #     elif len(results) == 2 :
                #         results.sort(key = lambda x : x[4])

                #         def checkPlace(center_y) :      # 상 = 0, 중 = 1, 하 = 2
                #             if h_line <= center_y < m_line : return 0
                #             if m_line <= center_y < l_line : return 1
                #             if l_line <= center_y <= stick : return 2
                #             return -1

                #         total_results = []
                        
                #         for result in results :
                #             center_y = result[1] + result[3]//2
                #             results.append(checkPlace(center_y))

                #         if total_results[0] == 0 and total_results[1] == 0 : print(lighter_num, "번 라이터 가스량 초과")
                #         if total_results[0] == 2 and total_results[1] == 2 : print(lighter_num, "번 라이터 가스량 미달")

                #     else : continue

                #     lighter_num += 1
                cv2.imshow("window",sub_img)
                if (cv2.waitKey(5) & 0xFF) == 27: break
                os.remove("images/"+str(cap)+".jpg")     # 처리가 끝난 이미지는 무조건 삭제
                cap += 1
                prev = time.time()

            except Exception as e :
                if str(e) == "'NoneType' object does not support item assignment" : time.sleep(0.33)
                print(str(e))
            
        else :      # 10초 이상 화면 캡쳐가 추가되지 않으면 종료
            if time.time() - prev > 10 : return
            else : pass
        
if __name__ == '__main__' :
    cap = 0
    proc1 = multiprocessing.Process(target=getCapture, args=(cap,))
    proc1.start()
    proc2 = multiprocessing.Process(target=yolo, args=(cap,))
    proc2.start()
    proc1.join()
    proc2.join()