import streamlit as st
import matplotlib.pyplot as plt
# import missingno as msno
PAGE_CONFIG = {"page_title":"DM.project","page_icon":":smiley:","layout":"centered"}
st.beta_set_page_config(**PAGE_CONFIG)
from functions import *

def main():
  # st.set_option('deprecation.showPyplotGlobalUse', False)
  st.title('Streamlit for Data Mining Project')
  # st.sidebar.info('**Switch between pages:**')
  st.sidebar.header('**Switch between pages:**')
  menu = ["Home","Plots","Trained model (SVM)"]
  choice = st.sidebar.selectbox('Menu',menu)
  data_load_state = st.text('Loading data...')
  # Load data into the dataframe.
  data,svm_model = loadData()
  laundry_clean=data.copy()
  encoded_data = get_encoded_data(data)
  # Notify the reader that the data was successfully loaded.
  data_load_state.text("Loading Done! (using st.cache)")
  if choice == 'Home':
    st.subheader("** Home Page ** ")
    st.subheader("_Given dataset : -> (Preprocessing of the dataset is not shown below)_")
    sheader = st.success('Dataframe with Cleaned data: ->')
    user_input_df = st.selectbox("Change dataframe: ",("Cleaned data","Encoded data"))
    if (user_input_df=="Cleaned data"):
      st.write(data)
    else:
      sheader.warning("Dataframe with Encoded data: ->")
      st.write(encoded_data)
    _=st.text('\n'),st.text('\n'),st.text('\n')

    st.info('''**Best features selected for model training**\n
      We are taking intersection of best features obtained from rfe, boruta and chi''')
    values = st.slider('Preview best features:',0, 13, (0, 13))
    st.write('Selected:', values)
    st.table(show_unique(laundry_clean[best_feat()[values[0]:values[1]]]))
    _=st.text('\n'),st.text('\n')
    st.info('''**ROC curve after performing SMOTE on the data: **\n
      SVC model is demonstrated in this demo because the performance is better''')
    st.image(roc_curve_smote(), caption='These are the results obtained using different classifiers', use_column_width=False)    
    st.sidebar.subheader('This demo is maintained by:')
    st.sidebar.dataframe(get_group_members())
    st.subheader('This demo is maintained by:')
    st.table(get_group_members())

  elif choice == 'Plots':
    # Laundry Plot
    st.warning('''** Graphs playground : ** \n
         Take a look at the sidebar options: 
         You may select different columns to plot.''')
    st.info('** Customizable bar chart : ** Please select variable of interest from the sidebar')
    # st.subheader('(Custom selections from the sidebar) Select columns to plot: ')
    cols = [col for col in laundry_clean.columns]
    st.sidebar.subheader('Select variables for bar chart: ')
    selected_col1 = st.sidebar.selectbox("X axis: ",([col for col in cols if (col!='Age_Range') & (col!='Date_Time')]),index=18)
    selected_col2 = st.sidebar.selectbox("Y axis: ",([col for col in cols if (col!=selected_col1) & (col!='Age_Range') & (col!='Date_Time')]),index=0)
    plot_against = st.sidebar.selectbox("Plot against: ",([col for col in [selected_col1,selected_col2]]),index=1)
    ax = laundry_plot(laundry_clean, [selected_col1,selected_col2], plot_against)
    st.pyplot()

    st.info('** Customizable heatmap : ** Please select variable of interest from the sidebar')
    heatmap_cols = [col for col in laundry_clean.columns]
    st.sidebar.subheader('Select variables for heatmap : ')
    heatmap_col1 = st.sidebar.selectbox("Vertical : ",([col for col in cols if (col!='Age_Range') & (col!='Date_Time')]),index=18)
    heatmap_col2 = st.sidebar.selectbox("Horizontal : ",([col for col in cols if (col!=heatmap_col1) & (col!='Age_Range') & (col!='Date_Time')]),index=0)
    ax = laundry_heatmap(laundry_clean, [heatmap_col1,heatmap_col2], heatmap_col1,heatmap_col2)
    st.pyplot()

    st.warning('''** Exploratory data analysis : ** \n
         Detailed questions are available in the submitted Jupyter file''')
    show_plots(laundry_clean)

  elif choice == 'Trained model (SVM)':
    st.sidebar.subheader('Select data for prediction : ')
    st.subheader('** Example : ** Default values to show eligible for promotion ')
    st.subheader('** Customizable predictor : ** Please select variable of interest from the sidebar')
    user_input_df,encoded = get_user_input(data)
    user_input_df.rename(index={0: "SELECTED USER INPUT"},inplace=True)
    encode_input_df = encoded.rename(index={0: "ENCODED USER INPUT"})
    input_df = user_input_df.append(encode_input_df.astype(str)).T
    st.table(input_df)
    encode_input_df = encode_input_df.values.tolist()
    y_pred = svm_model.predict(encode_input_df)
    st.subheader('** Prediction outcome : ** Is customer eligible for promotion? ')
    st.info('Results predicted using pretrained SVC model')
    if y_pred==0:
      st.error("**_NO_ , better dont give him/her! **")
    else:
      st.success("**_YES_ give him/her promotions!**")
    

if __name__ == '__main__':
  main()