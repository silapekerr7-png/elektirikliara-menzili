import streamlit as st
import pandas as pd
import pickle
from tensorflow import keras

st.title("Elektrikli Araç Menzil Tahmini (Derin Öğrenme) :battery:")
model=keras.models.load_model('ev_menzil_model.keras')
sutunlar=pickle.load(open('menzil_sutunlar.pkl','rb'))
secenekler=pickle.load(open('menzil_secenekler.pkl','rb'))
ozet=pickle.load(open('menzil_ozet.pkl','rb'))

yil=st.number_input('Model yılı',2008,2030,2022)
marka=st.selectbox('Marka',secenekler['Make'])
model_adi=st.selectbox('Model',secenekler['Model'])
tip=st.selectbox('Araç tipi',secenekler['Type'])

if st.button('Tahmin et'):
    veri=pd.DataFrame([{'Model Year':(yil-ozet['yil_ort'])/ozet['yil_std'],'Make':marka,'Model':model_adi,'Electric Vehicle Type':tip}])
    veri=pd.get_dummies(veri).reindex(columns=sutunlar,fill_value=0).astype('float32')
    tahmin=float(model.predict(veri.values)[0][0])
    st.success(f'Tahmini elektrikli menzil: {round(tahmin)} mil')
