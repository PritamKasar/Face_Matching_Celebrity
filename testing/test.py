import pickle
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

all_features=np.array(pickle.load(open('Face_Matching_Celb/features.pkl','rb')))
names=pickle.load(open('Face_Matching_Celb/names.pkl','rb'))

model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
detector=MTCNN()
demo_img=cv2.imread('Face_Matching_Celb/pritam_photo.jpg')
final=detector.detect_faces(demo_img)
x,y,w,h=final[0]['box']
face=demo_img[y:y+h,x:x+w]

img=Image.fromarray(face)
img=img.resize((244,244))
face_array=np.asfarray(img)
exp_img=np.expand_dims(face_array,axis=0)
preprocess_img=preprocess_input(exp_img)
final=model.predict(preprocess_img).flatten()

similar=[]

def key(x):
    return x[1]

for i in range(len(all_features)):
    similar.append(cosine_similarity(final.reshape(1,-1),all_features[i].reshape(1,-1))[0][0])
index_position=(sorted(list(enumerate(similar)),reverse=True,key=key)[0][0])
temp=cv2.imread(names[index_position])
img_rgb = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
cv2.imshow('out',img_rgb)
cv2.waitKey(0)