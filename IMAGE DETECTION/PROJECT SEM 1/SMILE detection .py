import numpy as np
import cv2


# Haarcascade are old mechine learning libraries
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')


#cap is to read the image using the webcam
#using 0 to access the default cam...we can use 1,2,.. for other webcam 
cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read() #ret is a flag to check the image is stored prooperly
   
    #converting(cvtColor) the gray haarcascade to color as they were predesigned gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    #detect the image that it is looking for 
    faces = faceCascade.detectMultiScale(gray,1.3,5,minSize=(30, 30))
        #scaleFactor=1.3
        #minNeighbors=5      
        #minSize=(30, 30)

   

    for (x,y,w,h) in faces:
    #we are using cv2.rectangle to detect the x and y corrdn and 
    #heigh and width then according to that info the bonding box will be created
    #here we are letting openCV know (x,y) is starting position and (x+w,y+h) is ending position
    #(255,0,0) is the color of the bonding box 
    # 2 is the thickness of the box
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
    #Region of interest(roi) 
    #we are letting opencv know i m only interested in the gray region of the face
    #using [y:y+h, x:x+w]
    #it is only trained to detect face image not the main gray image
        roi_gray = gray[y:y+h, x:x+w]
                
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            )
        
        for i in smile: #trying to find if one smile is detected or not
            if len(smile)>1: #if there is atleast one smile the next statement will print
            
    #(x,y-30) is the corrdn for the txt smiling on the detection box same as for the image of the main image
    #cv2.FONT_HERSHEY_SIMPLEX is the font for the txt
    #2 is the size of the font
    #(0,255,0) the colors used on the box
    #3 is the width of the detection box
    #cv2.LINE_AA the type of line the font should be using to inorder to render the word smiling
                cv2.putText(img,"Smiling",(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,
                    2,(0,255,0),3,cv2.LINE_AA)
               
    cv2.imshow('video', img) #to show the image
    k = cv2.waitKey(30) & 0xff #without waitkey the imshow won't work
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()