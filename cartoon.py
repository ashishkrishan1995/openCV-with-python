import numpy as np
import matplotlib.pyplot as plt

import cv2

ddepth = cv2.CV_8U
kernel_size = 300
img = cv2.imread('IMG.jpg')

img = np.asarray(img,np.uint8)

#img = np.unicode(img)
#img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #color coreectness regarding OpenCV reversed representation of BRG instead of RGB 
#MISBEHAVING !!!!

grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# cv2.imshow('irigion',img)

#numpy_horizontal_concat = np.concatenate((img, grayImage), axis=1)

median = cv2.medianBlur(grayImage,7) #MedianFilter to Reduce Noise DIDN"T FIND SMOOTH()

 
Laplaced = cv2.Laplacian(median, ddepth, kernel_size) ##Kernel Size is not doing any difference 

ret,thresh = cv2.threshold(Laplaced,5,125,cv2.THRESH_BINARY)
#thresh = np.ndarray(thresh.tolist().append([[0 for i in range(660)] for j in range(417)]))
#thresh.append([[0 for i in range(660)] for j in range(417)])
#print(len(thresh))
#cv2.threshold()
#thresh.append(np.array([[0 for i in range(660)] for j in range(417)]))

#thresh.append([[0 for i in range(660)] for j in range(417)])
thresh =255-thresh
thresh2 = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)

small = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
# cv2.imshow('Smalled',small)



painting = cv2.bilateralFilter(small,9,75,75)


painting_resized = cv2.resize(painting, (thresh2.shape[1], thresh2.shape[0]))

print("My thresh2 is ",thresh2.shape)	
print("My painting_resized is ",painting_resized.shape)	
print("My painting is ",painting.shape)	

dst = cv2.bitwise_and(painting_resized,thresh2)
cv2.imshow('dst',dst)
cv2.imshow('image',img)

#####################################################
#same window showing

titles = ['Original Image','Gray','Medained','thresh','Laplaced','small','painting','dst']
images = [img, grayImage, median,thresh2,Laplaced,small,painting,dst]


for i in range(8):
    plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
 
cv2.waitKey(0)
cv2.destroyAllWindows()

#####################################################
