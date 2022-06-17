import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
Let's Try Predict Your Sloth!

This app predicts  **Sloth Species** based on your chosen parameters
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    claw_length_cm = st.sidebar.slider("Claw length", 1.75, 12.17, 5.4)
    size_cm = st.sidebar.slider("Sloth body size", 46.93, 68.76, 50)
    tail_length_cm = st.sidebar.slider("Tail length", -2.94, 8.54, 1.3)
    
    data = {'claw_length_cm': claw_length_cm,
            'size_cm': size_cm,
            'tail_length_cm': tail_length_cm}
    features = pd.DataFrame(data, index[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

sloth = datasets.load_sloth()
X = sloth.data
Y = sloth.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(sloth.target_names)

st.subheader('Prediction')
st.write(sloth.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
