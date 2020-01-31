import numpy as np
import cv2
from services.detection import detection
from services.svm import SVM

# if you want print some log when your program is running, 
# just append a string to this variable
log = []


def image_to_speed(view1, view2, state):
  """This is the function where you should write your code to 
  control your car.
  
  You need to calculate your car wheels' speed based on the views.
  
  Whenever you need to print something, use log.append().

  Args:
      view1 (ndarray): The left-bottom view, 
                        it is grayscale, 1 * 120 * 160
      view2 (ndarray): The right-bottom view, 
                        it is colorful, 3 * 120 * 160
      state: your car's state, initially None, you can 
             use it by state.set(value) and state.get().
             It will persist during continuous calls of
             image_to_speed. It will not be reset to None 
             once you have set it.

  Returns:
      (left, right): your car wheels' speed
  """
  if state.get() is None:
    state.set(0)
  '''
  else:
    state.set(state.get()+1)'''
  log.append("#" + str(state.get()))
  cv2.imwrite("stoplne8.jpg",view1)
  
  #red light detect
  red_flag=0
  im3=view2.copy()
  im3[:,0:300]=0
  #cv2.imshow("im",im3)
  gray = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
  hsv = cv2.cvtColor(im3, cv2.COLOR_BGR2HSV)
  H, S, V = cv2.split(hsv)

  label = (H > 170) * (H < 185) * (S > 170) * (V > 190)
  thresh= gray*label
  thresh[label]=255
  #cv2.imshow("thresh",thresh)
  element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
  dilated = cv2.dilate(thresh, element)
  ret, thresh = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
  #cv2.imshow("thresh2",thresh)

  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  rect_max=[]
  area = 0
  for i in range(0, len(contours)):
    contour_area = cv2.contourArea(contours[i])
    if contour_area > area:
      area = contour_area
      rect = cv2.boundingRect(contours[i])
      w = rect[2]
      h = rect[3]
      aspect_ratio = max(w, h) / (min(w, h) * 1.0)
      if w > 10 and h > 10 and aspect_ratio < 1.5:
        rect_max.clear()
        rect_max.append([rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]])
  if len(rect_max)!=0:
    red_flag=1
    log.append("red_flag= "+str(red_flag))
    
  #stop_flag
  stop_flag=0
  im4=view1.copy()
  gray= cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
  im4_hsv= cv2.cvtColor(im4, cv2.COLOR_BGR2HSV)
  H, S, V = cv2.split(im4_hsv)
  label = (H > 90) * (H < 140) * (S < 10) * (V > 150)
  thresh= gray*label
  thresh[label]=255
  #cv2.imshow("im",im4)
  #cv2.imshow("thresh",thresh)
  _, maxVal, _, _= cv2.minMaxLoc(thresh[60:90,70:90] )
  if maxVal==255:
    stop_flag=1
    log.append("stop")
  
  sign_flag=0
  if view2 is not None:
    sign_classes = {
      14: 'Stop',
      33: 'Turn right',
      34: 'Turn left',
      35: 'Straight'
    }
    svm=SVM()
    detector=detection()
    im=view2
    rect=detector.ensemble(im)
    if rect:
      xmin,ymin,xmax,ymax=rect
      roi=im[ymin:ymax,xmin:xmax,:] 
      id_num=svm.predict(roi,"hog")
      sign_flag=1
      log.append("id:" + str(id_num))
      log.append(sign_classes[id_num])
  log.append(str(view1.shape))
  # image is left-bottom view, shaped as 160*120
  #view1 = view1.astype(int)
  
  hsv = cv2.cvtColor(view1, cv2.COLOR_BGR2HSV)
  H, S, V = cv2.split(hsv)
  #cv2.imshow("s",S)
  ret, thresh=cv2.threshold(S,70,255,cv2.THRESH_BINARY)
  #cv2.imshow("thresh",thresh)
  drawpic=im

  leftedge = []
  leftLossNum = 0
  leftpoint=[]
  rowst=40
  rowed=60
  distance=45
  for row in range(rowst, rowed):
    leftpt = -1
    for col in range(0, 160):
      if thresh[row][col] == 255:
        #drawpic = cv2.circle(drawpic, (col,row), 1, (255, 0, 0))
        if row==rowst or row==(rowed-1):
          leftpoint.append((col,row))
          #drawpic = cv2.circle(drawpic, (col, row), 1, (0, 0, 255))
        leftpt = col
        leftedge.append(leftpt)
        break
    if leftpt == -1:
      leftpt = 0
      leftLossNum += 1

  
  if len(leftpoint)==2:
    x1=leftpoint[0][0]
    y1=leftpoint[0][1]
    x2=leftpoint[1][0]
    y2=leftpoint[1][1]
    middle_x= ( x1+x2 )/2
    print("middle_x: "+str(middle_x))
    if x1 != x2:
      k1= (y1-y2)/(x1-x2)
      delta_x= pow(pow( distance,2)/(1+1/pow(k1,2)),0.5)
      x0=middle_x+delta_x
    else:
      delta_x= distance
      x0=middle_x+delta_x
    x0=min(x0,159)
    log.append("x0: " + str(x0))
  elif len(leftedge)>1:
    x0=np.mean(leftedge)+distance
    x0=min(x0,159)
    log.append("x0: " + str(x0))
  else:
    x0=115
    log.append("x0: " + str(x0))

  log.append(str(len(leftedge)))
  log.append("leftLossNum: " + str(leftLossNum))
  
  if sign_flag==1:
    if id_num==33:
      x0=115
      log.append("x0: "+str(x0))
    if id_num==34:
      state.set(state.get()+1)
      sign_flag=1
      id_num=35
  
  if sign_flag==0 and state.get()>0 and state.get()<10:
    #state.set(0)
    state.set(state.get()+100)
    x0=40
  elif sign_flag==0 and state.get()>=100:
    x0=40
    state.set(state.get()+100)
    if state.get()>=100*8:
      state.set(0)
    
      
  error= (x0-80.0) /80.0
  log.append("error: "+str(error))
  
  speed = 0.5 

  if abs(error)>0.2:
    Kp=1.3
  else:
    Kp=1.0
  SENSE = 1.0
  left_speed = speed + Kp * error * SENSE
  right_speed = speed - Kp * error * SENSE
  
  if sign_flag==1:
    if id_num==35:
      right_speed=1.0
      left_speed=1.0
  
  if (stop_flag==1 and red_flag==1):
    left_speed = 0
    right_speed = 0
    
  #if left_speed>0 and right_speed>0:
  return left_speed, right_speed
  #else:
  #return 0,5


