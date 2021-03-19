import numpy as np
import cv2
from PIL import Image
import keras 
model = keras.models.load_model('model.h5')
ex = {1:'angey',2:'disgust',3:'fear',4:'happy',5:'netural',6:'sad',7:'surprise'}
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    
    ret,frame = cap.read()
    frame = cv2.resize(frame,(600,600))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0))
         img = gray[x:x+w,y:y+h]
         img = cv2.resize(img,(48,48))
         img_data = keras.preprocessing.image.img_to_array(img)
         img_data = np.expand_dims(img_data,0)
         predict = model.predict(img_data)
         maxindex = int(np.argmax(predict))
         print(maxindex)
         cv2.putText(frame, 'face', (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
         

    cv2.imshow('frame',frame)
     
    if cv2.waitKey(1) == 0xFF == ord('q'):
        break
cap.release()
cap.destroyAllWindows()
