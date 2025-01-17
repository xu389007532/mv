from cv2 import cv2
import numpy as np

frame1=cv2.imread(r"./err/20230304/img0000085.png")
ret,frame=cv2.threshold(frame1,100,255,cv2.THRESH_BINARY)
cv2.imshow("",frame)
cv2.waitKey()
np_ker=2
ker=np.ones((np_ker,np_ker),np.uint8)

dic1=cv2.dilate(frame,ker)
frame=cv2.erode(frame,ker)
# cv2.imshow("",dic1)
# cv2.waitKey()
cv2.imshow("",frame)
cv2.waitKey()

kernel=0
if kernel:
    print(0)

kernel=1
if kernel:
    print(1)
