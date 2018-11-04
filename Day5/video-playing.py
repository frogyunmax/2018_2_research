#참고한 코드 https://webnautes.tistory.com/577 , https://goo.gl/NNBStM
import cv2
from PIL import Image
cam = cv2.VideoCapture('video.mp4')
cntvid = 0
while(cam.isOpened()):
    ret, image = cam.read()
    image_rotate = cv2.transpose(image)
    image_rotate = cv2.flip(image_rotate, 1)
    fps = cam.get(cv2.CAP_PROP_FPS)
    if (int(cam.get(1)) % 5 == 0):
        cv2.imwrite("frames/frame%d.png" % cntvid, image_rotate)
        cntvid += 1
    if cntvid == 12 : break
#작업 종료시키기
cam.release()
cv2.destroyAllWindows()

#이미지 변환
for i in range(cntvid):
    #print (i)
    path = 'frames/frame' + str(i) + '.png'
    save_path = 're_frames/re_' + str(i) + '.png'
    im_r = Image.open(path)
    size = (28, 28)
    im_r.thumbnail(size)
    im_r.save(save_path)
