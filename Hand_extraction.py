import numpy as np
import cv2
import math
import os
import Block_face

#Open Camera
draw = cv2.VideoCapture(0)

while draw.isOpened():

    # Read the frame
    ret, frame = draw.read()
    
    #Block the face
    Block_face.detect_face(frame, block=True)
    
    #Draw rectangle in which hand is to be placed 
    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
    crop = frame[100:300, 100:300]

    #Perform Gaussian blur
    blurred = cv2.GaussianBlur(crop, (3,3), 0)
    
    #change color format
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    #In range selection
    masknew = cv2.inRange(hsv, np.array([2,0,0]), np.array([20,255,255]))
       
    mask1 = np.ones((5,5))

    #perform dilation and erosion
    dilated = cv2.dilate(masknew, mask1, iterations = 1)
    eroded = cv2.erode(dilated, mask1, iterations = 1)    

    #gaussian blurring on eroded image   
    filter1 = cv2.GaussianBlur(eroded, (3,3), 0)
    
    #thresholding
    ret,thre = cv2.threshold(filter1, 127, 255, 0)
    
    cv2.imshow("Thresholded", thre)
    
    #contouring
    contours, hierarchy = cv2.findContours(thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
    
    try:

        contour = max(contours, key = lambda x: cv2.contourArea(x))
        
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(crop,(x,y),(x+w,y+h),(0,0,255),0)
        
        hull = cv2.convexHull(contour)

        drawing = np.zeros(crop.shape,np.uint8)
        cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
        cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
        
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour,hull)
        
        cd = 0
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
            
            if angle <= 90:
                cd += 1
                cv2.circle(crop,far,1,[0,0,255],-1)

            cv2.line(crop,start,end,[0,255,0],2)

        if cd== 0:
            cv2.putText(frame,"Video play", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
            os.system("vlc-ctrl play")
        elif cd == 1:
            cv2.putText(frame,"Video pause", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
            os.system("vlc-ctrl pause")
        elif cd == 2:
            cv2.putText(frame,"Mute", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
            os.system("vlc-ctrl volume 0")
        elif cd == 3:
            cv2.putText(frame,"Volume up", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
            os.system("vlc-ctrl volume +10%")
        elif cd == 4:
            cv2.putText(frame,"Volume down", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)
            os.system("vlc-ctrl volume -10%")
        else:
            pass
    except:
        pass


    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop))
    cv2.imshow('Contours', all_image)
      
    if cv2.waitKey(1) == ord('q'):
        break

draw.release()
cv2.destroyAllWindows()
