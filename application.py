import pandas as pd
import streamlit as application
from pickle import load
from PIL import Image
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
species_dict={'Setosa':0,'Versicolor':1,'Virginica':2}
value_list=list(species_dict.values())
key_list=list(species_dict.keys())
application.set_page_config(page_title='ML App')
application.title('Iris Prediction App')
image=Image.open('confusion_matrix.png')
image_iris=Image.open('iris.jpg')
image_iris2=Image.open('iris2.jpg')
image_iris3=Image.open('iris3.jpg')
col1,col2 = application.beta_columns(2)
with col1:
    application.image(image_iris, use_column_width=True)
    application.write('''# 🌼     🌻      🌷      🌺      🌸  🍁  \n #  🌹   ☘    💐''')

with col2:
    application.image(image_iris2, use_column_width=True)
    application.image(image_iris3, use_column_width=True)

application.write('**Random** rows of the *iris_dataset* ! ')
df=pd.read_csv('iris.csv')
application.table(df.sample(7))
application.write('## Select the Features: ')
sepal_length=application.slider('Sepal Length',min_value=4.30,max_value=8.0,value=5.1,step=.001,key='sp')
application.write('You selected sepal_length: ',sepal_length)
sepal_width=application.slider('Sepal Width',min_value=2.0,max_value=4.40,value=3.20,step=.001,key='sw')
application.write('You selected sepal_width: ',sepal_width)
petal_length=application.slider('Petal Length',min_value=1.0,max_value=6.90,value=4.0,step=.001,key='pl')
application.write('You selected petal_length: ',petal_length)
petal_width=application.slider('Petal Width',min_value=0.10,max_value=2.50,value=1.30,step=.001,key='pw')
application.write('You selected petal_width: ',petal_width)

add_selectbox = application.sidebar.selectbox(
    "Select the Algorithm",
    ("LogisticRegression", "RandomForestClassifier", "XGBClassifier",'KNeighborsClassifier','GaussianNB',
    'SVC(kernel="rbf")','SVC','GradientBoostingClassifier','AdaBoostClassifier')
)
lr=joblib.load('lr_joblib')
abc=joblib.load('abc_joblib')
nb=joblib.load('nb_joblib')
knn=joblib.load('knn_joblib')
xg=joblib.load('xg_joblib')
svc=joblib.load('svc_joblib')
svck=joblib.load('svck_joblib')
dt=joblib.load('dt_joblib')
gb=joblib.load('gb_joblib')
model_dict={"LogisticRegression":lr, "RandomForestClassifier":dt, "XGBClassifier":xg,'KNeighborsClassifier':knn,'GaussianNB':nb,
    'SVC(kernel="rbf")':svck,'SVC':svc,'GradientBoostingClassifier':gb,'AdaBoostClassifier':abc}

sc=StandardScaler()
model=model_dict[add_selectbox]
x_test=np.array(([[sepal_length,sepal_width,petal_length,petal_width]]))
sc = load(open('scaler.pkl', 'rb'))
application.write('# Predictions')
application.write(f'''## Model: {model} 
Your Input- {x_test[0]}''')
y_pred=model.predict(sc.transform(x_test))
application.success('Class of Iris is '+str(key_list[y_pred[0]]))
application.image(image,caption='Confusion Matrix and Classification Report',width=500)

