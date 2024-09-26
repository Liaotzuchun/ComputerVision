import cv2

cap = cv2.VideoCapture(0)
while(True):
    ret,frame   = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    im_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    t,binary_img = cv2.threshold(im_gray,150,255,cv2.THRESH_BINARY_INV)

    contours,hierachy = cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    o1 = cv2.drawContours(frame,contours,-1,(0,0,255),5)
    print(contours)
    cv2.imshow('frame',o1)
    if cv2.waitKey(1) == ord('q'):
        out = cv2.imwrite('capture.jpg',frame)
        break
cap.release()
cv2.destroyAllWindows()