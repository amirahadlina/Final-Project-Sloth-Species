import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

st.write("""
Let's Try Predict Your Sloth!

This app predicts  **Sloth Species** based on your chosen parameters
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    claw_length_cm = st.sidebar.slider("Claw length (cm)", 1.00, 13.00, 7.31)
    size_cm = st.sidebar.slider("Size (cm)", 46.00, 70.00, 68.76)
    tail_length_cm = st.sidebar.slider("Tail length (cm)", -3.0, 9.0, 1.1)
    data = {'claw_length_cm': claw_length_cm,
            'size_cm': size_cm,
            'tail_length_cm': tail_length_cm}
    features = pd.DataFrame(data,index=[0])
    return features
input_df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

sloth = pd.read_csv("https://raw.githubusercontent.com/amirahadlina/Final-Project-Sloth-Species/main/sloth_data_cleaned2a.csv")
X = sloth.loc[:,['claw_length_cm','size_cm','tail_length_cm']]
Y = sloth.loc[:,['species']]

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
