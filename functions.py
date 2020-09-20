import streamlit as st
import pandas as pd
import numpy as np
import pickle
import glob,os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
import imblearn
from imblearn.over_sampling import SMOTE
#Train test split
from sklearn.model_selection import train_test_split
from PIL import Image

def best_feat():
  return ['shirt_type', 'Wash_Item', 'Pants_Colour', 'Day', 'Race', 'Attire', 'Part_of_day', 'Shirt_Colour', 'pants_type', 'With_Kids', 'Basket_Size', 'Kids_Category', 'Basket_colour']

@st.cache
def loadData():
    pkl_filename = "svm_model.pkl"
    with open(pkl_filename, 'rb') as file:
        svm_model = pickle.load(file)
    laundry_clean = pd.read_csv('laundry_clean.csv')
    laundry_clean.set_index('No', inplace=True)
    laundry_clean["Date_Time"] = pd.to_datetime(laundry_clean["Date_Time"])
    return laundry_clean,svm_model

def laundry_plot(df, by, y, stack=False, sort=False, kind='bar'):
    pivot = df.groupby(by)[y].count().unstack(y)
    pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)
    ax = pivot.plot(kind=kind, stacked=stack, figsize=(8, 8))
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    return ax

def laundry_heatmap(df,by,count_att,y):
    df = df.groupby(by)[count_att].count().unstack(y)
    fig = plt.figure(figsize=(5,5))
    sns.heatmap(df,annot =  df.values ,fmt='g')
    return fig


def laundry_hour(laundry_clean):
    laundry_hour = laundry_clean.copy()
    laundry_hour['Hour'] = [i.hour for i in laundry_clean['Date_Time']]
    print(laundry_hour['Hour'].unique())
    laundry_hour[['Hour']].head(10)
    hour=laundry_hour.groupby(["Hour"]).size().to_frame("Count")
    fig = plt.figure(figsize=(7,7))
    ax = hour.plot(kind='bar')
    for p in ax.patches:
      ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    plt.xticks(rotation=0)
    return fig

def get_encoded_data(laundry_clean):
    laundry_encode = laundry_clean.copy()
    binary = ['Gender','With_Kids','Spectacles']
    ordinal = ['Body_Size','Age_Bin','Basket_Size']
    ordinal_ts = ['Day', 'Part_of_day']
    norminal = ['Race','Kids_Category','Basket_colour','Attire','Shirt_Colour','Pants_Colour' ,'shirt_type','pants_type','Wash_Item','Washer_No', 'Dryer_No']
    binary_encoder = preprocessing.LabelBinarizer()
    for i in binary:
        laundry_encode[i] = binary_encoder.fit_transform(laundry_encode[i])
    ordinal_encoder = OrdinalEncoder()
    laundry_encode[ordinal] = ordinal_encoder.fit_transform(laundry_encode[ordinal])
    label_ts = [['Monday', 'Tuesday' ,'Wednesday','Thursday','Friday','Saturday','Sunday'],
                ['Early Morning','Morning','Afternoon','Evening']]
    ordinal_encoder_ts = OrdinalEncoder(label_ts)
    laundry_encode[ordinal_ts] = ordinal_encoder_ts.fit_transform(laundry_encode[ordinal_ts])
    norminal_encoder = preprocessing.LabelEncoder()
    for i in norminal: 
        laundry_encode[i] = norminal_encoder.fit_transform(laundry_encode[i])
    to_convert = [i for i in laundry_encode if laundry_encode[i].dtypes == 'float64']
    laundry_encode[to_convert] = laundry_encode[to_convert].astype(int)
    return laundry_encode

@st.cache
def encode_user_input(laundry_clean):
    promo_criteria = laundry_clean.copy()
    label= [['Monday', 'Tuesday' ,'Wednesday','Thursday','Friday','Saturday','Sunday'],
            ['Early Morning','Morning','Afternoon','Evening'],
            ["chinese", "foreigner", "indian", "malay"],
            ['black','blue','brown','green', 'grey','orange','pink','purple','red', 'white','yellow']]
    day_dict = {label: i for i,label in enumerate(label[0])} 
    part_of_day_dict = {label: i for i,label in enumerate(label[1])} 
    race_dict = {label: i for i,label in enumerate(label[2])} 
    color_dict_value = {label: i for i,label in enumerate(label[3])} 
    for col_name in promo_criteria.columns:
      if col_name == 'With_Kids':
          promo_criteria[col_name] = promo_criteria[col_name].map({
              'no': 0, 'yes' : 1
              })
      elif col_name == 'Kids_Category':
        promo_criteria[col_name] = promo_criteria[col_name].map({
              'baby' : 0, 'no_kids' : 1,'toddler':2 ,'young':3
              })
      elif col_name == 'Basket_Size':
          promo_criteria[col_name] = promo_criteria[col_name].map({
            'big' : 0,'small' : 1
            })
      elif col_name == 'Attire':
          promo_criteria[col_name] = promo_criteria[col_name].map({
            'casual' : 0,'formal' : 1, 'traditional' : 2
            })
      elif col_name == 'shirt_type':
          promo_criteria[col_name] = promo_criteria[col_name].map({
            'long sleeve' : 0,'short_sleeve' : 1
            })
      elif col_name == 'pants_type':
          promo_criteria[col_name] = promo_criteria[col_name].map({
            'long' : 0,'short' : 1
            })
      elif col_name == 'Wash_Item':
          promo_criteria[col_name] = promo_criteria[col_name].map({
            'clothes' : 1,'blankets' : 0
            })
      elif col_name == 'Day':
          promo_criteria[col_name] = promo_criteria[col_name].map(day_dict)
      elif col_name == 'Part_of_day':
          promo_criteria[col_name] = promo_criteria[col_name].map(part_of_day_dict)
      elif col_name == 'Race':
          promo_criteria[col_name] = promo_criteria[col_name].map(race_dict)
      else:
          promo_criteria[col_name] = promo_criteria[col_name].map(color_dict_value)
    return promo_criteria

def get_user_input(data):
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """
    laundry_clean = data.copy()
    selected_feat = [col for col in laundry_clean.columns if col in best_feat()]
    input_arr=[]
    for i,col in enumerate(selected_feat):
      if (len(laundry_clean[col].unique())>3):
        input_arr.append(st.sidebar.selectbox(col,(laundry_clean[col].unique())))
      else:
        input_arr.append(st.sidebar.radio(col,(laundry_clean[col].unique()),index=1))
    df_input = pd.DataFrame([input_arr],columns = selected_feat, dtype=str) 
    encoded_input = encode_user_input(df_input)
    return df_input,encoded_input

#To show the initial 
def show_unique(df):
    count = []
    names = []
    for i in df.columns[:]:
      count.append(len(df[i].unique()))
      names.append(np.sort(df[i].unique().astype(str)))
    h = pd.DataFrame([df.columns[:],count,names])
    h = h.transpose()
    h.columns=['Attribute','Count','Data']
    h.set_index('Attribute',inplace=True)
    return h

@st.cache
def roc_curve_smote():
    image = Image.open('roc.png')
    return image

def show_plots(laundry_clean):
    ax2 = laundry_hour(laundry_clean)
    # another bar chart
    st.info('**Customers visiting trend by hours: **')
    st.pyplot()

    st.info('''**Which gender likes to visit the laundry shop in the early morning, morning, afternoon or evening?**
                  \nAlthough the peak hour is at 4am in the early morning, from the overall view of the data, the laundry shop is regularly patronized by the customers in the evening while there are relatively less customers in the morning and afternoon. From the bar chart below, it shows that both male and female preferred to visit in the evening. The laundry shop operation might need to take note of this and focus their business more in the evening and early morning.''')
    ax = laundry_plot(laundry_clean,['Part_of_day','Gender'],'Gender')
    st.pyplot()

    st.info('''**Which race likes to visit the laundry shop in the early morning, morning, afternoon or evening?**
                  \nChinese preferred to visit the laundry shop in the early morning while most of the Indian and Malay preferred to visit the laundry shop in the evening. The foreigners like to visit the laundry shop in the afternoon. It could be observed that both the race of Indian and Malay have a similar amount of visitation in the evening and afternoon.''')
    laundry_plot(laundry_clean,['Part_of_day','Race'],'Race')
    st.pyplot()


@st.cache
def get_group_members():
    # Group Members details
    data = {"No":[1,2,3],
            "Group Members Name":["Teo Sheng Pu", "Alex Tay Mao Xiang", "Liew Jun Xian"],
            "Student ID":[1171101665, 1171101775, 1171303519],
            "Contact Number":["+6010-7976911", "+6014-9888150", "+6012-7385789"],
            "e-Mail Address":["1171101665@student.mmu.edu.my", "1171101775@student.mmu.edu.my", "1171303519@student.mmu.edu.my"]}
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.set_index("No")
    return df