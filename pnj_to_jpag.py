import cv2 as cv
from PIL import Image
def png_to_jpg(path=None):
  im = cv.imread("/content/ECG/ecg.png")
  im = Image.fromarray(im)
  im.save("/content/ECG/ECG.jpg")