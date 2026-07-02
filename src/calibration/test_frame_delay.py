import cv2 
import time

cam=cv2.VideoCapture(0)
def ang():

    _,img=cam.read()
    print(img[0:3,0:2,:])

    return None

ang() 

print('Put you hand in front of camera or just change something in front of camera so it can detect different things for 10 seconds')


time.sleep(10) # put hand infront of camera

ang() # you should see different matrices, if they are same means same frame is captured


