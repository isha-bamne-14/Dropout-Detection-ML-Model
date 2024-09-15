# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 06:02:05 2024

@author: adwai
"""

import pandas as pd
import pickle
import streamlit as st

loaded_pipe = pickle.load(open('E:/ML/SIH/pipeline2.pkl', 'rb'))

def predict_status(input_data):
    cols = ['Attendance_Rate', 'GPA', 'Behavioral_Incidents', 'Counselor_Visits',
           'Parental_Education', 'Extracurricular_Participation',
           'Salary_Category']
    
    x = pd.DataFrame([input_data], columns = cols)
    y = loaded_pipe.predict(x)
    
    print('Prediction =', y[0])
    return y[0]


def main():
    st.title('Dropout Prediction Web App')
    
    # input data - Attendance_Rate	GPA	 Behavioral_Incidents	Counselor_Visits	Parental_Education	Extracurricular_Participation	Salary_Category
    attendance = st.text_input('Attendance Rate', placeholder='Type here')
    gpa = st.text_input('GPA', placeholder='Type here')
    behavioral_incidents = st.text_input('Behavioral_Incidents', placeholder='Type here')
    counselor_visits = st.text_input('Counselor Visits', placeholder='Type here')
    parental_education = st.text_input('Parental Education', placeholder='Type here')
    extracurricular_participation = st.text_input('Extracurricular Participation', placeholder='Type here')
    salary_category = st.text_input('Salary Category', placeholder='Type here')
    test1 = [attendance, gpa, behavioral_incidents, counselor_visits, parental_education, extracurricular_participation, salary_category]
    
    result = -1
    pred = ''
    
    if st.button('Predict'):
        result = predict_status(test1)
        if (result == 0):
            pred = 'This student will not dropout'
        else :
            pred = 'This student might dropout!!'
            
    st.success(pred)
    
if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    