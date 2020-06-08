import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
    import cv2
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy     # for capturing videos
import cv2
import numpy as np    # for mathematical operations
from skimage.transform import resize   # for resizing images
def createMiniBatches(mini_batch_size,input_file):
    # open the .txt file which have names of test videos
    mini_batch_count = 0
    f = open(input_file, "r")
    videoFile = f.readline()
    videoFile=videoFile.rstrip("\n")
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = cap.get(5)
    print(cap.isOpened())
    Y=[] # empty list to store all the segments of videos generated(final output!)
    X=[] # empty list to store the individual segments of videos
    total_frames=0
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        #current video ended , load new one
        if (ret != True):
            videoFile = f.readline()
            videoFile=videoFile.rstrip("\n")
            if(videoFile==""):
                cap.release()
                print ("Done!")
                return mini_batch_count       ##reached EOF
            cap.release()
            cap = cv2.VideoCapture(videoFile)
            frameRate = cap.get(5)
            continue

        a = resize(frame, preserve_range=True, output_shape=(128,128)).astype(np.uint8) #reshape frame to 128x128x3
        if(total_frames % 16 == 0 and total_frames>0):
            X1=np.stack(X) #convert X from list to numpy array
            Y.append(X1) 
            X=[]
        X.append(a)
        if(len(Y)==mini_batch_size):
            Y1=np.stack(Y) #convert Y from list to numpy array
            print(Y1.shape)
            Y=[]
            print("saving minibatch",mini_batch_count)
            np.savez('./minibatches/minibatch%d.npz' % (mini_batch_count)) #save Y1 to a numpy array
            mini_batch_count=mini_batch_count+1
        print("in loop",videoFile,frameId,total_frames)
        total_frames=total_frames+1
        
if __name__ == "__main__":
    print(createMiniBatches(16,'trainlist.txt'))
