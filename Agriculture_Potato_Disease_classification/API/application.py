import streamlit as st
from model import load_model,predict

model=load_model()


st.title("Potato Disease Classification 💀")
st.write("Upload an image of a potato leaf to predict the disease.")

file=st.file_uploader("Upload an image",type=["jpg","png","jpeg"])

if st.button("Predict"):
    if file is None:
        st.write("!!! Please upload an image")
    else:
        prediction=predict(model,file)
        st.write("The predicted disease is: ",prediction)

