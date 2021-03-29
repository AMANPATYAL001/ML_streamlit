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
image=Image.open('confusion_matrix.PNG')

st.write('**Random** rows of the *iris_dataset* ! ')
df=pd.read_csv('iris.csv')
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
lr,abc,nb,knn,dt,xg,gb,svck,svc=[joblib.load('lr_joblib')]*9
model_dict={"LogisticRegression":lr, "RandomForestClassifier":dt, "XGBClassifier":xg,'KNeighborsClassifier':knn,'GaussianNB':nb,
    'SVC(kernel="rbf")':svck,'SVC':svc,'GradientBoostingClassifier':gb,'AdaBoostClassifier':abc}

sc=StandardScaler()
model=model_dict[add_selectbox]
x_test=[[sepal_length,sepal_width,petal_length,petal_width]]
sc = load(open('scaler.pkl', 'rb'))
st.write('# Predictions')
st.write(f'''## Model: {model} 
Your Input- {x_test[0]}''')
y_pred=model.predict(sc.transform(x_test))
st.success('Class of Iris is '+str(key_list[y_pred[0]]))
st.image(image,caption='Confusion Matrix and Classification Report',width=500)
fig=px.scatter_3d(df,x='sepal_length',y='sepal_width',z='petal_width',color='species',size='petal_length',opacity=0.5)
fig.update_layout(margin=dict(l=0,r=0,b=0,t=0))
st.plotly_chart(fig)

from fpdf import FPDF
import base64

report_text = st.text_input("Report Text")


export_as_pdf = st.button("Export Report")

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'
df = px.data.gapminder().query("continent=='Oceania'")
fig = px.line(df, x="year", y="lifeExp", color='country')
st.plotly_chart(fig)
if export_as_pdf:
    fig.write_image("fig1.png")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 14)
    pdf.text(2,4,report_text)
    print(report_text)
    pdf.image('fig_user_world.png',5,15,160,150)
    pdf.image('fig1.png',5,150,150,150)
    pdf.set_font('Arial', 'B', 16)
    
    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")

    st.markdown(html, unsafe_allow_html=True)
