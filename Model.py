import pickle
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from tqdm import tqdm

name=pickle.load(open('names.pkl','rb'))

model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
def extractor(img_path,model):
    img=image.load_img(img_path,target_size=(224,224,3))
    img_array=image.img_to_array(img)
    extra_dimn=np.expand_dims(img_array,axis=0)
    preprocess_image=preprocess_input(extra_dimn)

    final=model.predict(preprocess_image).flatten()
    return final

feature=[]
for file in tqdm(name):
    feature.append(extractor(file,model))
pickle.dump(feature,open('Face_Matching_Celb/features.pkl','wb'))


