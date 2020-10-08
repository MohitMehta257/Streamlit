import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Accidents Prediction App

This app predicts the most accident prone state of USA
""")
st.write('---')
road_accidents=pd.read_csv('road-accidents.csv',sep='|',comment='#')
miles_driven=pd.read_csv('miles-driven.csv',sep='|',comment='#')
df=pd.concat([road_accidents,miles_driven['million_miles_annually']],axis=1)

X_train=df[['perc_fatl_speed', 'perc_fatl_alcohol','perc_fatl_1st_time','drvr_fatl_col_bmiles','million_miles_annually']]
y_train=df['state']

st.sidebar.header('User Input Features')

def user_input_features():
    drvr_fatl_col_bmiles=st.sidebar.slider('No. of drivers involved in fatal collision',int(X_train.drvr_fatl_col_bmiles.min()),int(X_train.drvr_fatl_col_bmiles.max()),int(X_train.drvr_fatl_col_bmiles.mean()))
    perc_fatl_speed=st.sidebar.slider('% of drivers involved in over-speeding',int(X_train.perc_fatl_speed.min()),int(X_train.perc_fatl_speed.max()),int(X_train.perc_fatl_speed.mean()))
    perc_fatl_alcohol=st.sidebar.slider('% of drivers alcohol impaired',int(X_train.perc_fatl_alcohol.min()),int(X_train.perc_fatl_alcohol.max()),int(X_train.perc_fatl_alcohol.mean()))
    perc_fatl_1st_time=st.sidebar.slider('% of drivers who had not been in accidents',int(X_train.perc_fatl_1st_time.min()),int(X_train.perc_fatl_1st_time.max()),int(X_train.perc_fatl_1st_time.mean()))
    million_miles_annually=st.sidebar.slider('million_miles_annually',int(X_train.million_miles_annually.min()),int(X_train.million_miles_annually.max()),int(X_train.million_miles_annually.mean()))

    data={'drvr_fatl_col_bmiles':drvr_fatl_col_bmiles,
    'perc_fatl_speed':perc_fatl_speed,
    'perc_fatl_alcohol':perc_fatl_alcohol,
    'perc_fatl_1st_time':perc_fatl_1st_time,
    'million_miles_annually':million_miles_annually
    }

    features=pd.DataFrame(data,index=[0])
    return features

df1=user_input_features()
st.write(df1)
st.write('---')


clf=RandomForestClassifier()
clf.fit(X_train,y_train)

prediction=clf.predict(df1)

st.write("""
### The most probable state
""")
st.write(prediction)
