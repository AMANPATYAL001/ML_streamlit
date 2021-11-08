import pandas as pd
import streamlit as st
from pickle import load
from PIL import Image
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.express as px
species_dict={'Setosa':0,'Versicolor':1,'Virginica':2}
value_list=list(species_dict.values())
key_list=list(species_dict.keys())
st.set_page_config(page_title='ML App')
st.title('Iris Prediction App')
image=Image.open('res/confusion_matrix.PNG')
image_iris=Image.open('res/iris.jpg')
image_iris2=Image.open('res/iris2.jpg')
image_iris3=Image.open('res/iris3.jpg')
col1,col2 = st.columns(2)

with col1:
    st.image(image_iris, use_column_width=True)
    st.write('''# 🌼     🌻      🌷      🌺      🌸  🍁  \n #  🌹   ☘    💐''')
with col2:
    st.image(image_iris2, use_column_width=True)
    st.image(image_iris3, use_column_width=True)
st.write('**Random** rows of the *iris_dataset* ! ')
df=pd.read_csv('res/iris.csv')
st.table(df.sample(7))
st.write('## Select the Features: ')
sepal_length=st.slider('Sepal Length',min_value=4.30,max_value=8.0,value=5.1,step=.001,key='sp')
st.write('You selected sepal_length: ',sepal_length)
sepal_width=st.slider('Sepal Width',min_value=2.0,max_value=4.40,value=3.20,step=.001,key='sw')
st.write('You selected sepal_width: ',sepal_width)
petal_length=st.slider('Petal Length',min_value=1.0,max_value=6.90,value=4.0,step=.001,key='pl')
st.write('You selected petal_length: ',petal_length)
petal_width=st.slider('Petal Width',min_value=0.10,max_value=2.50,value=1.30,step=.001,key='pw')
st.write('You selected petal_width: ',petal_width)

add_selectbox = st.sidebar.selectbox(
    "Select the Algorithm",
    ("LogisticRegression", "RandomForestClassifier", "XGBClassifier",'KNeighborsClassifier','GaussianNB',
    'SVC(kernel="rbf")','SVC','GradientBoostingClassifier','AdaBoostClassifier')
)
lr,abc,nb,knn,dt,xg,gb,svck,svc=[joblib.load('res/lr_joblib')]*9
model_dict={"LogisticRegression":lr, "RandomForestClassifier":dt, "XGBClassifier":xg,'KNeighborsClassifier':knn,'GaussianNB':nb,
    'SVC(kernel="rbf")':svck,'SVC':svc,'GradientBoostingClassifier':gb,'AdaBoostClassifier':abc}

sc=StandardScaler()
model=model_dict[add_selectbox]
x_test=[[sepal_length,sepal_width,petal_length,petal_width]]
sc = load(open('res/scaler.pkl', 'rb'))
st.write('# Predictions')
st.write(f'''## Model: {model} 
Your Input- {x_test[0]}''')
y_pred=model.predict(sc.transform(x_test))
st.success('Class of Iris is '+str(key_list[y_pred[0]]))
st.image(image,caption='Confusion Matrix and Classification Report',width=500)
fig=px.scatter_3d(df,x='sepal_length',y='sepal_width',z='petal_width',color='species',size='petal_length',opacity=0.5)
fig.update_layout(margin=dict(l=0,r=0,b=0,t=0))
st.plotly_chart(fig)
