import cv2
import numpy as np


class detection():

    def __init__(self):
        """
        Init the parameters
        """
        self.H_low_thresh = 112
        self.H_high_thresh = 125
        self.S_low_thresh = 120
        self.V_low_thresh = 100
        self.use_h_only = 0

    def show_circles(self, im, circles):
        """
        This function draws circles on the image im.
        :param im: image, opencv format
        :param circles, 2-d list which contains the positions and radiuses of circles
               format: [[c_x,c_y,r],[c_x,c_y,r]...]
        """
        if len(circles) != 0:
            for i in range(0, len(circles)):
                cv2.circle(im, (circles[i][0], circles[i][1]), circles[i][2], (0, 0, 255), 2)
        cv2.imshow("circles", im)
        cv2.waitKey(0)

    def show_rects(self, im, rects):
        """
        This function draws rectangles on the image im.
        :param im: image, opencv format
        :param rects:rectangles, 2-d list or 1-d list which contain the positions of rectangles
               for 2-d list
               format: [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax]...]
               for 1-d list
               format:[xmin,ymin,xmax,ymax]
        """
        if len(rects) != 0:
            if type(rects[0]) == list:
                for i in range(0, len(rects)):
                    cv2.rectangle(im, (rects[i][0], rects[i][1]), (rects[i][2], rects[i][3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(im, (rects[0], rects[1]), (rects[2], rects[3]), (0, 0, 255), 2)
        cv2.imshow("rects", im)
        cv2.waitKey(0)

    def change_threshold(self, h_low, h_high):
        """
        This function change the h thresholds for color segmentation which is used for extend experimrnt only.
        :param h_low: the low threshold of H
        :param h_high: the high threshold of H
        """
        self.H_low_thresh = h_low
        self.H_high_thresh = h_high
        self.use_h_only = 1

    def seg_color(self, im):
        """
        This function detects the sign by color segmentation.
        :param im: image,opencv format
        :return: the positions of the objects
                2-d list, format:
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax]....]
        """
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        blue = im[:, :, 0]
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        if self.use_h_only == 0:
            label = (h > self.H_low_thresh) * (h < self.H_high_thresh) * (s > self.S_low_thresh) * (
                    v > self.V_low_thresh)
        else:
            label = (h > self.H_low_thresh) * (h < self.H_high_thresh)

        gray = gray * label
        gray[label] = 255
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        eroded = cv2.erode(gray, element)
        dilated = cv2.dilate(eroded, element)
        ret, thresh = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
        # for opencv3
        # binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for opencv2
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        rects = []

        for i in range(0, len(contours)):
            rect = cv2.boundingRect(contours[i])
            w = rect[2]
            h = rect[3]
            if w > 15 and h > 15:
                rects.append([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])
        return rects

    def det_circle(self, img):
        """
        This function detects the circles by hough transform.
        :param img:image,opencv format
        :return: the positions of the circles, a 2-d list
                format: [[c_x-r,c_y-r,c_x+r,c_y+r],[c_x-r,c_y-r,c_x+r,c_y+r]....]
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # for opencv3
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=120, param2=40, minRadius=5, maxRadius=40)
        # for opencv2
        # circles=cv2.HoughCircles(gray,cv2.cv.CV_HOUGH_GRADIENT,1,100,param1=120,param2=40,minRadius=5,maxRadius=40)
        res_circles = []

        if type(circles) == np.ndarray:
            for circle in circles[0]:
                x = int(circle[0])
                y = int(circle[1])
                r = int(circle[2])
                res_circles.append([x, y, r])
        return res_circles

    def ensemble(self, img):
        """
        This function detects the signs combining the hough transform and color segmentation
        :param img: image,opencv format
        :return: the position of the largest sign, a 1-d list
                format: [xmin,ymin,xmax,ymax]
        """
        res_rects = []
        max_index = []
        circles = self.det_circle(img)
        rects = self.seg_color(img)

        if len(circles) != 0 and len(rects) != 0:
            for i in range(0, len(circles)):
                for j in range(0, len(rects)):
                    r = circles[i][2]
                    inter_xmin = max(circles[i][0] - r, rects[j][0])
                    inter_ymin = max(circles[i][1] - r, rects[j][1])
                    inter_xmax = min(circles[i][0] + r, rects[j][2])
                    inter_ymax = min(circles[i][1] + r, rects[j][3])

                    rect_w = rects[j][2] - rects[j][0]
                    rect_h = rects[j][3] - rects[j][1]

                    inter_w = inter_xmax - inter_xmin
                    inter_h = inter_ymax - inter_ymin
                    if inter_w > 0 and inter_h > 0 and abs(
                            (2 * inter_w * inter_h * 1.0) / (4 * r * r + rect_w * rect_h) - 1) < 0.3:
                        res_rects.append(rects[j])
                        max_index.append(rect_w * rect_h)

        if len(max_index) != 0:
            index = max_index.index(max(max_index))
            return res_rects[index]
        else:
            return res_rects


detector = detection()

im = cv2.imread("../static/signs/test.jpg")

#roi = im[50:400,90:460,:]

roi = im

rects = detector.ensemble(roi)

detector.show_rects(roi,rects)