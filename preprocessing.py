#first 2 lines and 4th line not required on machines other than mine
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
    import cv2
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy     # for capturing videos
import cv2
import numpy as np    # for mathematical operations
from skimage.transform import resize   # for resizing images
from cv2 import VideoWriter, VideoWriter_fourcc
count = 0
f = open("trainlist.txt", "r")
videoFile = f.readline()
videoFile=videoFile.rstrip("\n")
print(videoFile)
# cap = cv2.VideoCapture("./Dataset_Samples/Normal_Videos004_x264.mp4")   # capturing the video from the given path
cap=cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
x=1
Y=[] # empty list to store all the segments of videos generated(final output!)
X=[] # empty list to store the individual segments of videos
fourcc = VideoWriter_fourcc(*'MP42') #some code required for VideoWriter
video = VideoWriter('./segment0.avi', fourcc, float(24), (128, 128)) #creates video to store 1st segment
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    a = resize(frame, preserve_range=True, output_shape=(128,128)).astype(np.uint8) #reshape frame to 128x128x3
    video.write(a) #write frame to generated video
    if(frameId % 16 == 0 and frameId>0):
        X1=np.stack(X) #convert X from list to numpy array
        Y.append(X1) 
        X=[]
        video = VideoWriter('./segment%d.avi' % (frameId/16), fourcc, float(24), (128, 128)) #create new video
    X.append(a)
    #break when video ends
    if (ret != True):
        break
    print("in loop", frameId)
cap.release()
print ("Done!")
