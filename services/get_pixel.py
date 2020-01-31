import cv2
import numpy as np

"""
This is a tool function to get the HSV parameters of an pixel by
the mouse click. Press "ESC" to quit.
"""


def show_HSV(event, x, y, flags, param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        print(hsv[y, x, :])


img = cv2.imread("svm14.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.namedWindow('image')
cv2.setMouseCallback("image", show_HSV)
while (1):
    cv2.imshow("image", img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()