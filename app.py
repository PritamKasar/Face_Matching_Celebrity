from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image
import random

app = Flask(__name__)
app.secret_key = 'Pritam_Kasar' 
all_features = np.array(pickle.load(open('Face_Matching_Celb/features.pkl', 'rb')))
names = pickle.load(open('Face_Matching_Celb/names.pkl', 'rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
detector = MTCNN()

def predict_face(image_path):
    try:
        img = cv2.imread(image_path)
        
        faces = detector.detect_faces(img)
        
        if faces:
            x, y, w, h = faces[0]['box']
            face = img[y:y+h, x:x+w]
            
            img = Image.fromarray(face)
            img = img.resize((224, 224))
            face_array = np.asfarray(img)
            exp_img = np.expand_dims(face_array, axis=0)
            preprocess_img = preprocess_input(exp_img)
            
            features = model.predict(preprocess_img).flatten()
            
            similarities = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in all_features]
            
            best_match_index = np.argmax(similarities)
            
            predicted_name =  " ".join(names[best_match_index].split('\\')[1].split('_'))
            predicted_img=names[best_match_index].replace("\\","/")

            return predicted_name, predicted_img
        else:
            flash("No face detected in the input image.")
            return None, None
    except Exception as e:
        flash(f"Error processing image: {str(e)}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']

        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
           
            p=random.randint(1,99999)
            p=str(p)
            filename = 'Face_Matching_Celb/static/Saved_Images/Input_Image.jpg'
            file.save(filename) 

            
            
            predicted_name, predicted_img = predict_face(filename)
            
            if predicted_name:
                return render_template('index.html', predicted_img=predicted_img, predicted_name=predicted_name)
    

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
