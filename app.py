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
# import missingno as msno

st.title('Data Mining Project')


@st.cache
def loadData():
    pkl_filename = "svm_model.pkl"
    with open(pkl_filename, 'rb') as file:
        svm_model = pickle.load(file)

    laundry_clean = pd.read_csv('laundry_clean.csv')
    laundry_clean.set_index('No', inplace=True)
    return laundry_clean,svm_model


def laundry_plot(df, by, y, stack=False, sort=False, kind='bar'):
    pivot = df.groupby(by)[y].count().unstack(y)
    pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)
    ax = pivot.plot(kind=kind, stacked=stack, figsize=(8, 8))
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    return ax

def laundry_hour(laundry_clean):
    laundry_hour = laundry_clean.copy()
    laundry_hour['Hour'] = [i.hour for i in laundry_clean['Date_Time']]
    print(laundry_hour['Hour'].unique())
    laundry_hour[['Hour']].head(10)
    # laundry_clean.groupby([pd.Grouper(key='Date_Time',freq='H')]).size().reset_index(name='count')
    hour=laundry_hour.groupby(["Hour"]).size().to_frame("Count")

    plt.figure(figsize=(7,7))
    ax = hour.plot(kind='bar')
    auto_label()
    plt.xticks(rotation=0)
    return ax

def get_encoded_data(laundry_clean):
    laundry_encode = laundry_clean.copy()
    # Binary
    binary = ['Gender','With_Kids','Spectacles']
    # Ordinal
    ordinal = ['Body_Size','Age_Bin','Basket_Size']
    ordinal_ts = ['Day', 'Part_of_day']
    # Norminal
    norminal = ['Race','Kids_Category','Basket_colour','Attire','Shirt_Colour','Pants_Colour' ,'shirt_type','pants_type','Wash_Item','Washer_No', 'Dryer_No']
    # NOTE: everything follows in a sorted order
    # Binary Encoding 
    #  Female= 0 , Male = 1
    #  No    = 0 , Yes  = 1 
    binary_encoder = preprocessing.LabelBinarizer()
    for i in binary:
        laundry_encode[i] = binary_encoder.fit_transform(laundry_encode[i])
    #Ordinal Encoding
    # Body Size : fat=0, moderate=1, thin =2
    # Age Bin: 28-31=0, 32-35=1, 36-39=2 .... 52-55=6
    # Big = 0 , Small =1
    ordinal_encoder = OrdinalEncoder()
    laundry_encode[ordinal] = ordinal_encoder.fit_transform(laundry_encode[ordinal])
    #Ordinal Encoding day
    # Day: Monday=0, Tuesday=1 ... Sunday =6
    # Part_of_day: Early Morning = 0, Afternoon =1 
    label_ts = [['Monday', 'Tuesday' ,'Wednesday','Thursday','Friday','Saturday','Sunday'],
                ['Early Morning','Morning','Afternoon','Evening']]
    ordinal_encoder_ts = OrdinalEncoder(label_ts)
    laundry_encode[ordinal_ts] = ordinal_encoder_ts.fit_transform(laundry_encode[ordinal_ts])
    #Norminal encoding
    # Race: Chinese=0, foreigner=1, ... malay=3
    norminal_encoder = preprocessing.LabelEncoder()
    for i in norminal: 
        laundry_encode[i] = norminal_encoder.fit_transform(laundry_encode[i])
    to_convert = [i for i in laundry_encode if laundry_encode[i].dtypes == 'float64']
    laundry_encode[to_convert] = laundry_encode[to_convert].astype(int)
    # laundry_encode.dtypes  if u uncomment this , it will show up in the streamlit
    return laundry_encode

def get_train_test(laundry_encoded_fix):
    to_predict = 'PROMO'
    # to_predict = 'Wash_Item'
    laundry_encode=laundry_encoded_fix.copy()
    X = laundry_encode.drop([to_predict,'Date_Time'], axis='columns')
    y = laundry_encode[to_predict]

    ## DROP COLS that has simialar feature from feature selection 
    # print('Initial to train features: ', len(X.columns))

    # print('\nBest features: ',len(best_feat))
    # print('List of best features :\n',best_feat)
    best_feat = ['Basket_Size', 'Pants_Colour', 'Wash_Item', 'Day', 'Part_of_day', 'With_Kids', 'Race', 'pants_type', 'Basket_colour', 'Attire', 'Shirt_Colour']
    to_drop = np.setdiff1d(list(X.columns),best_feat)

    # print('\nTo drop features: ',len(to_drop))
    # print('List of to drop features :\n',to_drop)
    X = X.drop(to_drop, axis='columns')

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=10)
    # print('\nTo train features: ', len(X.columns))
    # print('List of To train features :\n',X.columns)

    smt = imblearn.over_sampling.SMOTE(sampling_strategy="minority", random_state=10, k_neighbors=5)
    X_res, y_res = smt.fit_resample(X, y)
    X_trainS, X_testS, y_trainS, y_testS = train_test_split(X_res, y_res, test_size = 0.30, random_state = 10)
    # print('\n\nOriginal  dataset shape %s' % Counter(y))
    # print('Resampled dataset shape %s' % Counter(y_res))
    return (X_train, X_test, y_train, y_test, X, y)

def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """


    race = st.sidebar.radio("Which race? ",('Chinese', 'Malay', 'Indian'))
    withKids = st.sidebar.selectbox("With kids? ",("yes", "no"))
    basketSize = st.sidebar.radio("Basket size ",('Big', 'Small'))
    basketColor = st.sidebar.slider('Basket color', 0, 5, 10)
    attire = st.sidebar.slider('attire', 0, 1, 2)
    shirtColor = st.sidebar.slider('shirt color', 0, 5, 10)
    pantsType = st.sidebar.slider('pants type', 0, 1, 1)
    pantsColor = st.sidebar.slider('Pants color', 0, 5, 10)
    washItem = st.sidebar.slider('Wash item', 0, 1, 1)
    day = st.sidebar.slider('day', 0, 3, 6)
    partOfDay=st.sidebar.slider('part of day', 0, 1, 3)
    
    selected_race=0
    if race == 'Chinese':
        selected_race = 0
    elif race == 'Malay':
        selected_race = 1
    elif race == 'Indian':
        selected_race = 2
    
    selected_WithKids = 0 if withKids == "no" else 1
    selected_basketSize = 0 if basketSize == "Big" else 1


    features = {'race': selected_race,
            'withKids': selected_WithKids,
            'basketSize': selected_basketSize,
            'basketColor':basketColor,
            'attire': attire,
            'shirtColor': shirtColor,
            'pantsColor': pantsColor,
            'pantsType': pantsType,
            'washItem': washItem,
            'day': day,
            'partOfDay': partOfDay,
            }
    data = pd.DataFrame(features,index=[0])

    return data



# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data,svm_model = loadData()
encoded_data = get_encoded_data(data)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

st.subheader('Cleaned data')
st.write(data)

st.subheader('EXAMPLE OF SLIDER: Number of pickups by hour')
# min: 0h, max: 23h, default: 17h
hour_to_filter = st.slider('hour', 0, 23, 17)
ax = laundry_plot(data, ['With_Kids','Kids_Category'], 'Kids_Category')
# ax = laundry_hour(data)
st.pyplot()

st.subheader('Encoded data')
st.write(encoded_data)

X_train, X_test, y_train, y_test, X, y = get_train_test(encoded_data)
# svm_model.fit(X_train,y_train)
# predefinedInput = [[0, 0, 8, 0, 5, 1, 1, 1, 0, 0,1]]
predefinedInput= X_test.iloc[4,:]
y_pred = svm_model.predict([predefinedInput])

# st.write(y_pred)
st.write(y_pred)
st.markdown('Predefined data shows **_'+str(y_pred)+'_ promotion**.')
st.subheader('X_test data')
st.write(X_test)
# st.write(X_train.iloc[0,:].values.tolist())

st.subheader('User Input parameters')
user_input_df = get_user_input()
user_input = user_input_df.values.tolist()

st.write(user_input_df)
st.write(user_input)
y_pred = svm_model.predict(user_input)
st.markdown('Predicted is **_'+str(y_pred)+'_ promotion**.')