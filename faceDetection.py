import cv2

# Import the model
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
img = cv2.imread('Resources/ProjectExhibition.JPG')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the faces in our image
faces = faceCascade.detectMultiScale(imgGray,1.1,4)

# Create bounding box around the faces that we have detected. So we need to loop through all the faces that we have detected.
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 4)

cv2.imshow("Result", img)
cv2.imwrite("Resources/ProjectExhibitionOutput.png", img)

cv2.waitKey(0)