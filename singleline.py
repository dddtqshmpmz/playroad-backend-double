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
  #cv2.imwrite('svm3.jpg',view1)
  
  hsv = cv2.cvtColor(view1, cv2.COLOR_BGR2HSV)
  
  view1 = cv2.cvtColor(view1, cv2.COLOR_BGR2GRAY)
  log.append(str(view1.shape))
  
  if state.get() is None:
    state.set(1)
  else:
    state.set(state.get()+1)
  log.append("#" + str(state.get()))
  log.append("### running...")
  view1 = view1.astype(int)
  
  
  H, S, V = cv2.split(hsv)
  cpx,cpy =np.where(S[50:70,:]>85)#follow yellow line
  #cpx, cpy = np.where(view1[50:70,:]>150)
  if len(cpy) != 0:
    error = (np.array(cpy).mean() - 80.0)/80.0
  else:
    error=0
  
  
  log.append('cpy: '+str(cpy.mean()))
  log.append('error: '+str(error))
  speed = 0.7
  Kp = 1.3
  SENSE = 1.0
  left_speed = speed + Kp * error * SENSE
  right_speed = speed - Kp * error * SENSE
  
  
  if view2 is not None:
    #cv2.imwrite('svm11.jpg',view2)
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
    log.append(len(rect))
    if rect:
      xmin,ymin,xmax,ymax=rect
      roi=im[ymin:ymax,xmin:xmax,:] 
      id_num=svm.predict(roi,"hog")
      log.append("id:" + str(id_num))
      log.append(sign_classes[id_num])
      if (id_num==35):
        speed = 0.8#0.45
        Kp = 1.0
        SENSE = 1.0
    
        left_speed = speed + Kp * error * SENSE
        right_speed = speed - Kp * error * SENSE
        left_speed=right_speed

      elif (id_num==34):
        speed = 0.7 #0.3
        Kp = 2.5 # 1.6
        SENSE = 1.0
        left_speed = speed + Kp * error * SENSE
        right_speed = speed - Kp * error * SENSE
        log.append("left_v:" + str(left_speed))
        log.append("right_v:" + str(right_speed))
        left_speed=speed/3
        
        
      elif (id_num==33):
        speed = 0.7
        Kp = 2.5
        SENSE = 1.0
        left_speed = speed + Kp * error * SENSE
        right_speed = speed - Kp * error * SENSE
        
        log.append("left_v:" + str(left_speed))
        log.append("right_v:" + str(right_speed))
        right_speed=speed/3


  return left_speed,right_speed


