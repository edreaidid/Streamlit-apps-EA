import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Glucose Control Prediction App 

This app predicts the **glucose control** for you!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sbp = st.sidebar.slider('sbp', 30, 240, 120)
    dbp = st.sidebar.slider('dbp', 10, 180, 70)
    age = st.sidebar.slider('age', 18, 110, 40)
    wt = st.sidebar.slider('wt', 20, 140, 60)
    data = {'sbp': sbp,
            'dbp': dbp,
            'age': age,
            'wt': wt}
    features = pd.DataFrame(data, index=[0])
    return features

z = user_input_features()

st.subheader('User Input parameters')
st.write(z)

df = pd.read_csv('C:\\Users\\raef\\code-master\\streamlit\\part2\\healthstatus6.csv')

def filter(x):
    if x < 6.5:
        return 'good'
    if x >= 6.5:
        return 'poor'
df['glu']=df['hba1c'].apply(filter)
X = df[["sbp","dbp","age","wt"]]
Y= df["glu"]

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(z)
prediction_proba = clf.predict_proba(z)

st.subheader('Class labels and their corresponding index number')
st.write(df['glu'])

st.subheader('Prediction')
st.write(prediction)
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
