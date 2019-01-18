import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import math
import imageio
import numpy as np 
from PIL import ImageGrab as ig
import time

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg')

img = scipy.misc.imresize(img, [200, 200])
rows,cols,c = img.shape
print("Dimensions of Steering Wheel: ",(rows,cols))
smoothed_angle = 0

#read data.txt
xs=[]
ys=[]
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/"+line.split()[0])
        ys.append(float(line.split()[1])*scipy.pi /180)
#get number of images
num_images = len(xs)


i = math.ceil(num_images*0.8)
print("Starting frame of video:" + str(i))
images = []
while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    rows1,cols1,nc = image.shape
    print("Dimensions of Road Image: ",(rows1,cols1))
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    #call("cls",shell = True)# for linux call("clear")
    #print("Predicted steering angle: " + str(degrees) + " degrees")
    print("Steering angle: " +str(degrees) + " (pred)\t"+str(ys[i]*180/scipy.pi)+" (actual)")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    # capture the screen
    # https://stackoverflow.com/questions/35097837/capture-video-data-from-screen-in-python
    screen = ig.grab()
    images.append(np.array(screen))
    i += 1
cv2.destroyAllWindows()
imageio.mimsave('output.gif', images)
