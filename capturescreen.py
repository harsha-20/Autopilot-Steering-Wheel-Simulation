import numpy as np
import cv2
from PIL import ImageGrab as ig
import time

last_time = time.time()
while(True):
    screen = ig.grab()
    #print('Loop took {} seconds',format(time.time()-last_time))
    cv2.imshow("test", np.array(screen))
    last_time = time.time()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break