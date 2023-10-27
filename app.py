import pickle
import streamlit as st
import numpy as np

# streamlit is aliased as st
st.title('Insurance Charge Prediction')
st.header('Fill the details to predict Insurance Charges')


lr = pickle.load(open('ins_charges_lr_model.pkl','rb'))     # rb = read binary
dt = pickle.load(open('ins_charges_dt_model.pkl','rb'))
rf = pickle.load(open('ins_charges_rf_model.pkl','rb'))

model = st.sidebar.selectbox('Select model',['Lin_Reg','DT_Reg','RF_Reg',''])

age = st.slider('Age',18,64)
sex = st.selectbox('Sex',['Male','Female'])
bmi = st.slider('BMI',6,53)
children = st.selectbox('Children',[0,1,2,3,4,5,6])
smoker = st.selectbox('Smoker',['Yes','No'])
region = st.selectbox('Region',['NWest','SEast','SWest','NEast'])


if st.button('Predict Insurance Charge'):
    if sex=="Male":
        sex = 1
    else:
        sex = 0
    if smoker=="Yes":
        smoker = 1
    else:
        smoker="No"
    if region=="NWest":
        rnwest = 1
        seast = 0
        swest = 0
        neast = 0
    elif region=="SEest":
        rnwest = 0
        seast = 1
        swest = 0
        neast = 0
    elif region=="SWest":
        rnwest = 0
        seast = 0
        swest = 1
        neast = 0
    else:
        rnwest = 0
        seast = 0
        swest = 0
        neast = 1
    test = np.array([age,sex,bmi,children,smoker,rnwest,seast,swest])
    test = test.reshape([1,8])
    if model =="Lin_Reg":
        st.success(lr.predict(test)[0])
    elif model=="DT_Reg":
        st.success(dt.predict(test)[0])
    else:
        st.success(rf.predict(test)[0])



