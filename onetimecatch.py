import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
ret,frame = cap.read()
out = cv2.imwrite('capture.jpg',frame)
o = cv2.imread('capture.jpg')
plt.hist(o.ravel(),255)
cv2.imshow('original_grey',o)
plt.show()
cv2.imshow('original_grey',o)
cv2.waitKey()
cv2.destroyAllWindows()