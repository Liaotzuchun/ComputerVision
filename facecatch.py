import cv2

cap = cv2.VideoCapture("ning.mp4")
facecas = cv2.CascadeClassifier('face.xml')

while True:
    ref , frame = cap.read()
    if ref:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facerect = facecas.detectMultiScale(gray,1.1,5)
        print(len(facerect))
        for(x,y,w,h) in facerect:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
        frame = cv2.resize(frame,(0,0),fx=0.2,fy=0.2)
        cv2.imshow("img",frame)
        cv2.waitKey(1)
    else:
        break
    if cv2.waitKey(1) == ord('q'):
        break