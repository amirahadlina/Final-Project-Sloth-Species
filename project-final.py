import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

st.write("""
Let's Try Predict Your Sloth!

This app predicts  **Sloth Species** based on your chosen parameters
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    claw_length = st.sidebar.slider('Claw length', 4.3, 7.9, 5.4)
    Sloth_body_size = st.sidebar.slider('Sloth body size', 2.0, 4.4, 3.4)
    Tail_length = st.sidebar.slider('Tail length', 1.0, 6.9, 1.3)
    data = {'Claw_length': [Claw_length],
            'Sloth_body_size': [Sloth_body_size],
            'Tail_length': [Tail_length]}
    features = pd.DataFrame(data, index[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

Sloth = datasets.load_sloth()
X = Sloth.data
Y = Sloth.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Sloth.target_names)

st.subheader('Prediction')
st.write(Sloth.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
