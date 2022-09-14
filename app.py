import pandas as pd
import numpy as np
import pickle
import requests
import json
import streamlit as st
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
from PIL import Image

# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

def welcome():
    return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def prediction(sepal_length, sepal_width, petal_length, petal_width):

    prediction = classifier.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    return prediction
    

# this is the main function in which we define our webpage
def main():
 
        # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:pink;padding:15px">
    <h1 style ="color:orange;text-align:center;">Iris Flower Classifier ML App </h1>
    </div>
    """
    
    
    


# this line allows us to display the front end aspects we have
# defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
    
    lottie_hello = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_vps4jt0g.json")

    st_lottie(lottie_hello,speed=2,reverse=False,loop=True,quality="low",height=400,width=None,key=None)
    

    sepal_length = st.number_input("Sepal Length")
    sepal_width = st.number_input("Sepal Width")
    petal_length = st.number_input("Petal Length")
    petal_width = st.number_input("Petal Width")
    result =""

    if st.button("Predict"):
        result = prediction(sepal_length, sepal_width, petal_length, petal_width)
        
    
        
    if result==1:
        st.write('This is setosa')
    elif result==2:
        st.write('This is versicolor')
    else:
        st.write('This is virginica')
        
    

if __name__=='__main__':
    main()
