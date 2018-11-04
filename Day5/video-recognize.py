#참고한 코드 https://webnautes.tistory.com/577
import cv2 as cv
#카메라 가져오기
cam = cv.VideoCapture(0)

#저장하기
r = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('video.avi', r, 30.0, (640, 480))
while(cam.isOpened()):
    #사진 찍기 (캡쳐)
    ret, frame = cam.read()

    if ret == False:
        continue

    #frame = cv.flip(frame, 0)
    out.write(frame)
    cv.imshow('frame', frame)

    ##종료하려면 ESC키
    if cv.waitKey(1) & 0xFF == 27:
        break
#종료하기
cam.relase()
out.release()
cv.destroyAllWindows()

