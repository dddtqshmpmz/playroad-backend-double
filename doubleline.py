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
      
      
  #if left_speed>0 and right_speed>0:
  return left_speed, right_speed
  #else:
  #return 0,5


