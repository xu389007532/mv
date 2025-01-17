from pylibdmtx.pylibdmtx import decode
import cv2
test=decode(cv2.imread("./img0000001.png"))
list1=[]

list1.append(str(test[0][0]))
b1=b'U7000'
print(b1.decode())
print(test[0][0].decode())

print(str(list1))
