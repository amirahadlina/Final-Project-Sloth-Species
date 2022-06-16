import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

st.title("""
# Sloth Species Prediction
""")

st.header("This app predicts **Sloth Species** type")
st.sidebar.header('Please choose these parameters to predict your Sloth!')

def user_input_features():
    Claw_length = st.sidebar.slider('Claw Length', 1.50, 12.20, 7.31)
    Sloth_body_size = st.sidebar.slider('Sloth Body Size', 45.0, 69.10, 61.8)
    Tail_length = st.sidebar.slider('Tail Length', -2.50, 9.50, 4.5)
    data = {'claw_length': claw_length,'sloth_body_size': sloth_body_size,'tail_length': tail_length}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

sloth = pd.read_csv("https://raw.githubusercontent.com/amirahadlina/Final-Project-Sloth-Species/main/sloth_data_cleaned.csv")
X = sloth.loc[:,['claw_length','sloth_body_size','tail_length']]
Y = sloth.loc[:,['species']]
clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.table(["two_toed", "three_toed"])


st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Iris_%28plant%29.jpg/640px-Iris_%28plant%29.jpg")
  
st.subheader('Prediction')
#st.write(sloth.target_specie[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
