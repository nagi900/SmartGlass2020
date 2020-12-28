import cv2
from src.hand_tracker import HandTracker
#オリジナル
import subprocess # アプリを立ち上げたりするモジュール
import numpy as np
import pyautogui as pag # マウスやキーボードの操作
import time
import os # デスクトップ録画用　なくてもいい
#

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite" #手のひら検出専用？調べた方がよさそう
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2


#オリジナル
font = cv2.FONT_HERSHEY_SIMPLEX #fontのところを元のまま書くと、cv2.FONT~ の.に反応してしまう

#カメラ映像を録画　フレームレートが合わないので直さないといけない
fourcc=cv2.VideoWriter_fourcc(*'mp4v') # 書き出すコーデック名
out=cv2.VideoWriter('output.mp4',fourcc, 8.0, (640,480))#書き込むファイル 謎 フレームレート 縦横比？

actwin_width=int(100) # 目に表示するウィンドウの大きさ
actwin_height=int(100)

ACTWIN='actwin'
cv2.namedWindow(ACTWIN) # これで一つwindowが開く　特に変数に代入したりする必要はない
#ここにアクティブウィンドウのスクショの画像を表示するやつを書く
#

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

#テスト
detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)


#↓↓オリジナル↓↓
#

xfir = 0.0
xsec = 0.0
xthir = 0.0
def vector(standard,xfir):
    global xsec #関数内でグローバル変数の変更をするときはglobalをつけないといけない
    global xthir
    v = xfir - (xfir + xsec + xthir)/3  #三つまえの位置の平均と現在位置の差を調べる
    xthir = xsec
    xsec = xfir

    if v<-standard:
        Vnm='li'
    elif v>standard:
        Vnm='ri'
    else :
        Vnm='ke'
    return Vnm #Vdicから移動している方向に対応した文字を返す

#対象が入力されてからの時間経過を返す関数
time_terget_back = 'null'
time_count = 0.0
time_start = 0.0
def time_measu(time_terget):
    global time_terget_back
    global time_count
    global time_start
    #ターゲットが変更されたか、5秒以上経過していたらリセット
    if time_terget != time_terget_back or time_count > 5.0: 
        time_count = 0.0
        time_start = time.time()
        #time_terget_back = ''
    time_count = time.time() - time_start #time.time()で現在時刻
    time_terget_back = time_terget
    return time_count
#
#↑↑オリジナル↑↑


while hasFrame:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)
    if points is not None:
        print(points)
        for point in points: #for文の基本系 文字列の配列(オブジェクト)pointsに最初から順にpointに格納
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            # ↑フレームを画像として円を書き込み
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
        
        #↓↓オリジナル↓↓
        #
        #人差し指と小指の付け根を読み取り
        inpalm_x,inpalm_y = points[5]
        outpalm_x,outpalm_y = points[17]
        #指先と第二間接 x座標は使わないので散らす
        fintip1_x,fintip1_y = points[6]
        _,scdjoint1_y = points[8]
        _,fintip2_y = points[10]
        _,scdjoint2_y = points[12]
        _,fintip3_y = points[14]
        _,scdjoint3_y = points[16]
        _,fintip4_y = points[18]
        _,scdjoint4_y = points[20]
        #手のひらの横幅の1/3を計算
        palm_length = abs(inpalm_x - outpalm_x) #abs()で絶対値 
        palen_onethird = palm_length / 3

        #手のひらの裏表を判別
        ForB = 'overse'
        if inpalm_x < outpalm_x :
            ForB = 'reverse'
        #指の曲げ伸ばしを判別
        finsign1 = np.sign(fintip1_y - scdjoint1_y)
        finsign2 = np.sign(fintip2_y - scdjoint2_y)
        finsign3 = np.sign(fintip3_y - scdjoint3_y)
        finsign4 = np.sign(fintip4_y - scdjoint4_y)
        finsigns=[finsign1,finsign2,finsign3,finsign4]

        
        #frameへの書き込み
        #裏表を記述
        cv2.putText(frame,ForB,(0,20),font,1,(200,255,200),3)
        #手のひらの左右の動きを記述
        cv2.putText(frame,vector(palen_onethird,inpalm_x),(0,40),font,1,(125,0,125),3)
        #指の曲げ伸ばしを記述
        textsign = str (finsigns)
        cv2.putText(frame,textsign,(0,70),font,1,(0,0,255),3)
        
        if ForB=='reverse' and (finsigns==[1,-1,-1,-1] or finsigns==[1,-1,-1,1]): #pythonはorとかandとかで書く
            _=time_measu('choice-mode')# time_measuに送信　返り値は捨てる
            cv2.putText(frame,'choice-mode',(200,20),font,1,(0,0,255),3)
            pag.moveTo(int(fintip1_x)*2,int(fintip1_y)*2)

            if finsigns==[1,-1,-1,1]: #小指を立てたらクリック
                pag.click(int(fintip1_x)*2,int(fintip1_y)*2)
        #手が表向きなら
        if ForB=='overse':
            #指が2ならexcelを開く
            if finsigns==[1,1,-1,-1]:
                cv2.putText(frame,'app2-open',(200,20),font,1,(0,0,255),2)
                time_measu_int = int(time_measu('app2-open')*10)
                cv2.putText(frame,str(time_measu_int),(200,40),font,1,(255,255,255),2)
                cv2.ellipse(frame,(350,200),(80,80),270,0,72/10*time_measu_int,(155,255,255),20)
            
                if time_measu_int > 45:
                    subprocess.Popen(r'"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"')
            #指が3ならwordを開く
            elif finsigns==[1,1,1,-1]:
                cv2.putText(frame,'app3-open',(200,20),font,1,(0,0,255),2)
                time_measu_int = int(time_measu('app3-open')*10)
                cv2.putText(frame,str(time_measu_int),(200,40),font,1,(255,255,255),2)
                cv2.ellipse(frame,(350,200),(80,80),270,0,72/10*time_measu_int,(255,155,255),20)
                if time_measu_int > 45:
                    subprocess.Popen(r'"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"')
            #指が4ならpowerpointを開く
            elif finsigns==[1,1,1,1]:
                cv2.putText(frame,'app4-open',(200,20),font,1,(0,0,255),2)
                time_measu_int = int(time_measu('app4-open')*10)
                cv2.putText(frame,str(time_measu_int),(200,40),font,1,(255,255,255),2)
                cv2.ellipse(frame,(350,200),(80,80),270,0,72/10*time_measu_int,(255,255,155),20)
                if time_measu_int > 45:
                    subprocess.Popen(r'"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE"')
            
        else :
            _=time_measu('reset')#time_measuをリセット
        
    out.write(frame)#録画
    #
    # ↑↑オリジナル↑↑
    
    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

out.release()#オリジナル　録画

capture.release()
cv2.destroyAllWindows()
