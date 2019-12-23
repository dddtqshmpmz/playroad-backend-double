import cv2
from services.detection import detection
from services.svm import SVM


# Traffic signs dictionary
sign_classes = {14: 'Stop',
                33: 'Turn right',
                34: 'Turn left',
                35: 'Straight'}

svm=SVM()
detector=detection()
im=cv2.imread("../static/signs/left.jpg")
rect=detector.ensemble(im)
xmin,ymin,xmax,ymax=rect
roi=im[ymin:ymax,xmin:xmax,:] 
id_num=svm.predict(roi,"hog")
print("id:",id_num)
print(sign_classes[id_num])

