import dlib
import cv2 as cv
import numpy as np
import glob
import shutil

#path_video = glob.glob(r'*.mp4')
#path_txt = glob.glob(r'*.txt')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')


cap = cv.VideoCapture('video3.mp4') #비디오 이름 입력해여
check=0
count=0
while True:
    ret, img_frame_o = cap.read()
    # resize the video
    if(ret==False) :
        break
    count=count+1
    img_frame = cv.resize(img_frame_o, dsize=(720, 480), interpolation=cv.INTER_AREA)
    cv.cvtColor(img_frame,cv.COLOR_BGR2YUV)
    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)
    out_face = np.zeros_like(img_frame)
    dets = detector(img_gray, 1)
    if not dets :
        print("frame "+str(count)+" no face sensing")# 얼굴인식이 안될 경우 --> 처리 x
        check=1
        #filename=i
        #dir="/Users/정문주/PycharmProjects/vip/"
        #shutil.copy(dir,dir+'NO')
        break
    print('frmae '+str(count) +' ok')

if check==0:
    print('ok')
    #dir="/Users/정문주/PycharmProjects/vip/"
    #shutil.copy(dir, dir+'OK')
cap.release()
#writer.release()
cap.destroyAllWindows()