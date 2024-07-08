import os
import pickle

face= os.listdir('Face_Matching_Celb/dataset')
names=[]
for i in face:
    for j in os.listdir(os.path.join('Face_Matching_Celb/dataset',i)):
        names.append(os.path.join('Face_Matching_Celb/dataset',i,j))

pickle.dump(names,open('Face_Matching_Celb/names.pkl','wb'))
