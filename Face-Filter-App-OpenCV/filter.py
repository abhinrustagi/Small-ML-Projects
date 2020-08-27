import cv2
import numpy as np

img_path = 'Jamie_Before.png'

image = cv2.imread('Jamie_Before.jpg')

nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'Nose18x15.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'frontalEyes35x16.xml')

if nose_cascade.empty():
  raise IOError('Unable to load the nose cascade classifier xml file')

eyes = eyes_cascade.detectMultiScale(image, 1.3, 5)
nose = nose_cascade.detectMultiScale(image, 1.3, 5)
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
