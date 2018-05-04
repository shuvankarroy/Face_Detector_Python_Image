import cv2
import numpy as np
i=1 #for saving each face
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
left_eyeCascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eyeCascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
img = cv2.imread(input('Enter Full Image Path'))

gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(40, 40),
)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    right_eyes = right_eyeCascade.detectMultiScale(roi_gray)
    for (rex,rey,rew,reh) in right_eyes:
        cv2.rectangle(roi_color,(rex,rey),(rex+rew,rey+reh),(0,255,0),2)
    left_eyes = left_eyeCascade.detectMultiScale(roi_gray)
    for (lex,ley,lew,leh) in left_eyes:
        cv2.rectangle(roi_color,(lex,ley),(lex+lew,ley+leh),(0,255,0),2)
    #saves each face in independent images:
    cropped=img[y:y+h , x:x+w]
    cv2.imwrite("face{0}.jpg".format(i), cropped)
    i+=1
    cv2.imshow('cropped',img)
    
cv2.imshow('img',img)
print("Found {0} faces!".format(len(faces)))
cv2.waitKey(0)
cv2.destroyAllWindows()


# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]
cv2.imwrite('subpixel5.png',img)
cv2.imshow('img',img)

