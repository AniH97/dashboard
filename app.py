# -*- coding: utf-8 -*-
"""
Created on Sat May 15 12:11:38 2021
@author: Ani
"""
#import needed packages
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
from datetime import datetime
#import seaborn as sb  # Statistics data visualization base on matplotlib
from sklearn.cluster import KMeans
from pandas import DataFrame
#import matplotlib.ticker as ticker  # import a special package
from numpy import inf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.io as io
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash_table

#define stylesheet
external_stylesheets = ['mystyle.css']

#import data files:
raw_data_holiday = pd.read_csv('holiday_17_18_19.csv')
raw_data_meteo = pd.read_csv('IST_meteo_data_2017_2018_2019.csv')
raw_data_energy_17 = pd.read_csv('IST_Central_Pav_2017_Ene_Cons.csv')
raw_data_energy_18 = pd.read_csv('IST_Central_Pav_2018_Ene_Cons.csv')

#Rename colums
raw_data_meteo.rename(columns = {'yyyy-mm-dd hh:mm:ss': 'Date_Time'}, inplace = True)
raw_data_meteo.rename(columns = {'temp_C': 'Temperature [°C]'}, inplace = True)
raw_data_meteo.rename(columns = {'HR': 'Relative Humidity [%]'}, inplace = True)
raw_data_meteo.rename(columns = {'windSpeed_m/s': 'Wind Speed in [m/s]'}, inplace = True)
raw_data_meteo.rename(columns = {'windGust_m/s': 'Wind Gust speed [m/s]'}, inplace = True)
raw_data_meteo.rename(columns = {'pres_mbar': 'Air Pressure [mbar]'}, inplace = True)
raw_data_meteo.rename(columns = {'solarRad_W/m2': 'Solar Radiation [W/m²]'}, inplace = True)
raw_data_meteo.rename(columns = {'rain_mm/h': 'Rainfall [mm/h]'}, inplace = True)
raw_data_meteo.rename(columns = {'rain_day': 'Rainfall [yes/no]'}, inplace = True)

raw_data_energy_17.rename(columns = {'Date_start': 'Date_Time'}, inplace = True)
raw_data_energy_18.rename(columns = {'Date_start': 'Date_Time'}, inplace = True)
raw_data_energy_17.rename(columns = {'Power_kW': 'Power [kW]'}, inplace = True)
raw_data_energy_18.rename(columns = {'Power_kW': 'Power [kW]'}, inplace = True)

#unify Date datatype
raw_data_meteo.Date_Time = pd.to_datetime(raw_data_meteo.Date_Time)
raw_data_energy_17.Date_Time = pd.to_datetime(raw_data_energy_17.Date_Time, format='%d-%m-%Y %H:%M')
raw_data_energy_18.Date_Time = pd.to_datetime(raw_data_energy_18.Date_Time, format='%d-%m-%Y %H:%M')
raw_data_holiday.Date = pd.to_datetime(raw_data_holiday.Date)
#raw_data_holiday = raw_data_holiday.set_index(['Date'])
#print(raw_data_holiday)
#Join the temperature data sets
data_energy_17_18 = pd.concat([raw_data_energy_17, raw_data_energy_18], ignore_index=True)

#increase holiday data to hourly data

first_date = pd.to_datetime('2017-01-01 00:00') #first date of Data in datetime format
index = pd.Series(pd.date_range(first_date, periods = data_energy_17_18.shape[0], freq='1H')) #define timeframe of data
data_holiday = pd.DataFrame(dict(Date_Time = index ))

data_holiday['Date'] = data_holiday.Date_Time.dt.date
data_holiday.Date = pd.to_datetime(data_holiday.Date)

data_holiday['Holiday'] = data_holiday['Date'].isin(raw_data_holiday['Date']).astype(float) #adds Holiday column marking the holidays with a 1     

data_holiday = data_holiday.drop(columns = 'Date')

#reduce Meteo Data to hourly values and drop measurements from 2019:
data_meteo = raw_data_meteo.resample('1H', on = 'Date_Time').mean()
#data_meteo = data_meteo[:-28980]

#Check for empty columns
raw_data_meteo[raw_data_meteo.isnull().any(axis = 'columns')]
raw_data_energy_17[raw_data_energy_17.isnull().any(axis = 'columns')]
raw_data_energy_18[raw_data_energy_18.isnull().any(axis = 'columns')]
raw_data_holiday[raw_data_holiday.isnull().any(axis = 'columns')]

#Create hourly dataset with all values
all_data = pd.merge(data_energy_17_18, data_meteo, on = 'Date_Time')
all_data = pd.merge(all_data, data_holiday, on = 'Date_Time')
all_data['Hour'] = all_data['Date_Time'].dt.hour
all_data['Date'] = all_data['Date_Time'].dt.date
all_data['Day_of_the_week'] = all_data['Date_Time'].dt.weekday
all_data['Month'] = all_data['Date_Time'].dt.month
all_data = all_data.drop(columns=['Date_Time'])
all_data = all_data.set_index ('Date', drop = True)

#Drop empty columns
all_data = all_data.dropna()


# Plot:
all_data1 = all_data.reset_index()
all_data1['Date'] = pd.to_datetime(all_data1['Date'], format='%Y-%m-%d')
all_data_17 = all_data1.loc[(all_data1['Date'] <= '2017-12-31')]
all_data_18 = all_data1.loc[(all_data1['Date'] >= '2018-01-01') & (all_data1['Date'] <= '2018-12-31')]
all_data_1718 = all_data1.loc[(all_data1['Date'] <= '2018-12-31')]
fig_1718_P = px.scatter(all_data1, x = 'Date', y='Power [kW]', color_discrete_sequence = ["dodgerblue"], width=900, height=400)
fig_17_P = px.scatter(all_data_17, x = 'Date', y='Power [kW]', color_discrete_sequence = ["dodgerblue"], width=900, height=400)
fig_18_P = px.scatter(all_data_18, x = 'Date', y='Power [kW]', color_discrete_sequence = ["dodgerblue"], width=900, height=400)
fig_1718_T = px.scatter(all_data1, x = 'Date', y='Temperature [°C]', color_discrete_sequence = ["dodgerblue"], width=900, height=400)
fig_17_T = px.scatter(all_data_17, x = 'Date', y='Temperature [°C]', color_discrete_sequence = ["dodgerblue"], width=900, height=400)
fig_18_T = px.scatter(all_data_18, x = 'Date', y='Temperature [°C]', color_discrete_sequence = ["dodgerblue"], width=900, height=400)
#Notes: There are some very low values, that only occur in 2017 
        # There are some upper outliers as well
#Remove outliers
from scipy import stats
#import numpy as np
z = np.abs(stats.zscore(all_data['Power [kW]']))
#Combine Zcore and IQR to get cleaned data
from scipy import stats
#import numpy as np
z = np.abs(stats.zscore(all_data['Power [kW]']))
threshold = 3 # 3 sigma...Includes 99.7% of the data
all_data_Zcore=all_data[(z < 3)]
all_data_clean = all_data_Zcore[all_data_Zcore['Power [kW]'] >all_data_Zcore['Power [kW]'].quantile(0.25) ]

#Clustering
# create kmeans object
cluster_data = all_data_clean.drop(columns = ['Wind Gust speed [m/s]', 'Rainfall [yes/no]', 'Wind Speed in [m/s]', 'Air Pressure [mbar]', 'Rainfall [mm/h]', 'Relative Humidity [%]', 'Solar Radiation [W/m²]'])
cluster_data.isnull()
model = KMeans(n_clusters=5).fit(cluster_data)
pred = model.labels_


cluster_data['Cluster']=pred
cluster_data["Cluster"] = cluster_data["Cluster"].astype(str)
ax1=px.scatter(cluster_data, x='Power [kW]',y='Day_of_the_week',color='Cluster', color_discrete_map = {"0":"lightgrey","1":"lightblue","2":"steelblue", "3":"darkturquoise", "4":"dodgerblue"}, width=900, height=400)
ax2=px.scatter(cluster_data, x='Power [kW]',y='Temperature [°C]',color='Cluster', color_discrete_map = {"0":"lightgrey","1":"lightblue","2":"steelblue", "3":"darkturquoise", "4":"dodgerblue"}, width=900, height=400)
ax3=px.scatter(cluster_data, x='Power [kW]',y='Month',color='Cluster', color_discrete_map = {"0":"lightgrey","1":"lightblue","2":"steelblue", "3":"darkturquoise", "4":"dodgerblue"}, width=900, height=400)
ax4=px.scatter(cluster_data, x='Power [kW]',y='Hour',color='Cluster', color_discrete_map = {"0":"lightgrey","1":"lightblue","2":"steelblue", "3":"darkturquoise", "4":"dodgerblue"}, width=900, height=400)
ax5=px.scatter(cluster_data, x='Power [kW]',y='Holiday',color='Cluster', color_discrete_map = {"0":"lightgrey","1":"lightblue","2":"steelblue", "3":"darkturquoise", "4":"dodgerblue"}, width=900, height=400)

#3D plots Hour and Day of the week:
cluster_0=cluster_data[pred==0]
cluster_1=cluster_data[pred==1]
cluster_2=cluster_data[pred==2]
cluster_3=cluster_data[pred==3]
cluster_4=cluster_data[pred==4]
fig_3D_TH = px.scatter_3d(cluster_data, x='Temperature [°C]', y='Hour', z='Power [kW]',
              color='Cluster', color_discrete_map = {"0":"lightgrey","1":"lightblue","2":"steelblue", "3":"darkturquoise", "4":"dodgerblue"}, width=900, height=400)
fig_3D_DM = px.scatter_3d(cluster_data, x='Day_of_the_week', y='Month', z='Power [kW]',
              color='Cluster', color_discrete_map = {"0":"lightgrey","1":"lightblue","2":"steelblue", "3":"darkturquoise", "4":"dodgerblue"}, width=900, height=400)

#Feature Creation
#Workingday: If holiday or weekend than 0 else 1
all_data_clean['working_day'] = 1
all_data_clean.loc[(all_data_clean['Holiday'] == 1) | (all_data_clean['Day_of_the_week'] >4 ), 'working_day'] = 0 
all_data_clean['logtemp']=np.log(all_data_clean['Temperature [°C]'])
all_data_clean['Power-1'] = all_data_clean['Power [kW]'].shift(1)
all_data_clean=all_data_clean.dropna()

#Feature Selection
table_f1 = go.Figure(data=[go.Table(header=dict(values=['Feature 1', 'Feature 2', 'Feature 3'],line_color='darkslategray',
                fill_color='lightskyblue',font=dict(color='black'),
    height=40),
                 cells=dict(values=[['Holiday'], ['Workingday'], ['Log Temperature']],line_color='darkslategray',
               fill_color='lightcyan', font=dict(color='black'),
    height=40
))
                     ])
table_f2 = go.Figure(data=[go.Table(header=dict(values=['Feature 1', 'Feature 2', 'Feature 3'],line_color='darkslategray',
                fill_color='lightskyblue'),
                 cells=dict(values=[['Holiday'], ['Workingday'], ['Log Temperature']],line_color='darkslategray',
               fill_color='lightcyan', font=dict(color='black'),
    height=40
))
                     ])
table_f3 = go.Figure(data=[go.Table(header=dict(values=['Feature 1', 'Feature 2', 'Feature 3'],line_color='darkslategray',
                fill_color='lightskyblue'),
                 cells=dict(values=[['Power-1'], ['Hour'], ['Day of the Week']],line_color='darkslategray',
               fill_color='lightcyan', font=dict(color='black'),
    height=40
))
                     ])

features = {'Method':['K-best', 'Recursive Feature Elimination', 'Ensembling Methods'],'Feature 1':['Holiday', 'Holiday', 'Power-1'], 'Feature 2': ['Workingday', 'Workingday','Hour'], 'Feature 3':['Log Temperature', 'Log Temperature', 'Day of the week']} 
features_df = pd.DataFrame(data = features)

#Regression
all_data_clean = all_data_clean.reset_index(drop=True)
df_model=all_data_clean.drop(columns=['Relative Humidity [%]', 'Wind Speed in [m/s]', 'Wind Gust speed [m/s]','Air Pressure [mbar]','Solar Radiation [W/m²]','Rainfall [mm/h]', 'Rainfall [yes/no]'  ])
df_model.to_csv('IST_Total_Hourly_Model.csv', encoding='utf-8', index=True)
df_data = df_model
X=df_data.values

Y=X[:,0]
X=X[:,[3,5,6,7,8]]
X_train, X_test, y_train, y_test = train_test_split(X,Y)

#Linear Regression
from sklearn import  linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred_LR = regr.predict(X_test) 
data_lin=pd.DataFrame(y_test[1:2000], columns=['Test'])
data_lin['Prediction'] = y_pred_LR[1:2000]
fig_lin_reg = px.line(data_lin, color_discrete_map = {"Prediction":"darkturquoise", "Test":"dodgerblue"}, width=700, height=400, labels=dict(x="Time", y="Power [kW]"))
fig_lin_scat = px.scatter(x = y_test, y = y_pred_LR, color_discrete_sequence = ["dodgerblue"], width=700, height=400, labels=dict(x="Real data", y="Regression Results"))
#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
data = {'Test':['MAE', 'MSE', 'RMSE','cvRMSE']}
Results = pd.DataFrame(data)
Results['LR']=(MAE_LR,MSE_LR,RMSE_LR,cvRMSE_LR)
#Support Vecotr Regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_SVR = sc_X.fit_transform(X_train)
y_train_SVR = sc_y.fit_transform(y_train.reshape(-1,1))
regr = SVR(kernel='rbf')
regr.fit(X_train_SVR,y_train_SVR)
y_pred_SVR = regr.predict(sc_X.fit_transform(X_test))
y_test_SVR=sc_y.fit_transform(y_test.reshape(-1,1))
y_pred_SVR2=sc_y.inverse_transform(y_pred_SVR)
data_vec=pd.DataFrame(y_test_SVR[1:2000], columns=['Test'])
data_vec['Prediction'] = y_pred_SVR2[1:2000]
fig_vec_reg = px.line(data_vec, color_discrete_map = {"Prediction":"darkturquoise", "Test":"dodgerblue"}, width=700, height=400, labels=dict(x="Time", y="Power [kW]"))
fig_vec_scat = px.scatter(x = y_test, y = y_pred_SVR2, color_discrete_sequence = ["dodgerblue"], width=700, height=400, labels=dict(x="Real data", y="Regression Results"))
#Error
MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2) 
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)  
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)
Results['SVR']=(MAE_SVR,MSE_SVR,RMSE_SVR,cvRMSE_SVR)

#Random forest
from sklearn.ensemble import RandomForestRegressor
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)
data_RF=pd.DataFrame(y_test[1:2000], columns=['Test'])
data_RF['Prediction'] = y_pred_RF[1:2000]
fig_RF_reg = px.line(data_RF, color_discrete_map = {"Prediction":"darkturquoise", "Test":"dodgerblue"}, width=700, height=400, labels=dict(x="Time", y="Power [kW]"))
fig_RF_scat = px.scatter(x = y_test, y = y_pred_RF, color_discrete_sequence = ["dodgerblue"], width=700, height=400, labels=dict(x="Real data", y="Regression Results"))
#Error
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
Results['RF']=(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)


table_LR = go.Figure(data=[go.Table(header=dict(values=['MAE', 'MSE', 'RMSE','cvRMSE'],line_color='darkslategray',
                fill_color='lightskyblue',),
                 cells=dict(values= Results['LR'],line_color='darkslategray',
               fill_color='lightcyan',
))
                     ])
table_SVR = go.Figure(data=[go.Table(header=dict(values=['MAE', 'MSE', 'RMSE','cvRMSE'],line_color='darkslategray',
                fill_color='lightskyblue',),
                 cells=dict(values= Results['SVR'],line_color='darkslategray',
               fill_color='lightcyan',
))
                     ])
table_RF = go.Figure(data=[go.Table(header=dict(values=['MAE', 'MSE', 'RMSE','cvRMSE'],line_color='darkslategray',
                fill_color='lightskyblue',),
                 cells=dict(values= Results['RF'],line_color='darkslategray',
               fill_color='lightcyan',
))
                     ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Img(src=app.get_asset_url('IST_logo.png'), height="70"),
    html.H2('Main Building Tecnico - Power Consumption'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Clustering', value='tab-2'),
        dcc.Tab(label='Feature Selection', value='tab-3'),
        dcc.Tab(label='Regression', value='tab-4'),
        
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
             

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Power and Temperature for 2017 & 2018'),
            dcc.Checklist(        
        id='radio_year',
        options=[
            {'label': '2017', 'value': 2017},
            {'label': '2018', 'value': 2018},
            
        ],

        value=[2017,2018]
        
        ),
        
            dcc.Dropdown( 
        id='dropdown_measurement',
        options=[
            {'label': 'Power [kW]', 'value': 1},
            {'label': 'Temperature [°C]', 'value': 2},
        ], 
        value=1
        ),   
            
        html.Div(id='graphyear_png'),
        
                    ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.H3('Plotting 2D & 3D clusters'),
                html.Div([
                    html.H4('2D Clusters'),
                    dcc.Dropdown(
                    id='2D_Cluster',
                    options=[
                        {'label': 'Day of the week', 'value': 1},
                        {'label': 'Temperature [°C]', 'value': 2},
                        {'label': 'Month', 'value': 3},
                        {'label': 'Hour', 'value': 4},
                        {'label': 'Holiday', 'value': 5},
                        ],
                    value= 1 )
                ],    
                style={'width': '48%', 'display': 'inline-block'}),
                html.Div(id='graph_cluster_2D'),
    
                html.Div([
                    html.H4('3-D Clusters'),
                    dcc.Dropdown(
                    id='3D_Cluster',
                    options=[
                        {'label': 'Day of the week & Month', 'value': 1},
                        {'label': 'Temperature [°C] & Hour', 'value': 2},
                        ],
                    value= 1 )
                ],    
                style={'width': '48%', 'display': 'inline-block'}),
                html.Div(id='graph_cluster_3D'),
            ]),
        ])
    elif tab == 'tab-3':
        return html.Div([
            dcc.Dropdown(
                id = 'feature',
                options=[
                        {'label': 'K-Best', 'value': 1},
                        {'label': 'Recursive Feature Elimination', 'value': 2},
                        {'label': 'Ensembling Methods', 'value': 3},
                        ],
                        value= 1),
            html.Div(id='feature_table'),
            ])
    
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Results of different Regression methods'),
            dcc.Dropdown(
                id='Regression_method',
                options=[
                        {'label': 'Linear Regression', 'value': 1},
                        {'label': 'Support Vector Regression', 'value': 2},
                        {'label': 'Random Forest', 'value': 3},
                        ],
                        value= 1),  
        html.Div([
            html.H4('Line Plot'),
                html.Div(id='graph_regression'),  
        ], 
        style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4('Scatter Plot'),
                html.Div(id='graph_regression_scatter'),  
        ],
        style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4('Errors'),
            html.Div(id='table_Errors'),    
            ])
        
        
        ])

@app.callback(Output('graphyear_png', 'children'), 
              Input('radio_year', 'value'),
              Input('dropdown_measurement', 'value'))


def render_figure_png(radio_year, dropdown_measurement): 
    
    if radio_year == [2017]:
        if dropdown_measurement == 1: 
            return  html.Div([dcc.Graph(figure=fig_17_P),])
        elif dropdown_measurement == 2:
            return  html.Div([dcc.Graph(figure=fig_17_T),])
    elif radio_year == [2018]:
        if dropdown_measurement == 1: 
            return  html.Div([dcc.Graph(figure=fig_18_P),])
        elif dropdown_measurement == 2:
            return  html.Div([dcc.Graph(figure=fig_18_T),])    
    elif radio_year == [2017,2018]:
        if dropdown_measurement == 1:
            return  html.Div([dcc.Graph(figure=fig_1718_P),])
        elif dropdown_measurement == 2:
            return  html.Div([dcc.Graph(figure=fig_1718_T),])
    elif radio_year == [2018,2017]:
        if dropdown_measurement == 1:
            return  html.Div([dcc.Graph(figure=fig_1718_P),])
        elif dropdown_measurement == 2:
            return  html.Div([dcc.Graph(figure=fig_1718_T),])
    elif radio_year == []:
            return html.P('no cell selected')

        
@app.callback(Output('graph_cluster_2D', 'children'), 
              Input('2D_Cluster', 'value'))

def figure_cluster_2D(Cluster2D): 
    if Cluster2D == 1:
        return  html.Div([dcc.Graph(figure=ax1),])
    if Cluster2D == 2:
        return  html.Div([dcc.Graph(figure=ax2),])
    if Cluster2D == 3:
        return  html.Div([dcc.Graph(figure=ax3),])
    if Cluster2D == 4:
        return  html.Div([dcc.Graph(figure=ax4),])
    if Cluster2D == 5:
        return  html.Div([dcc.Graph(figure=ax5),])


@app.callback(Output('graph_cluster_3D', 'children'), 
              Input('3D_Cluster', 'value'))
def figure_cluster_3D(Cluster3D):
    if Cluster3D == 1:
        return html.Div([dcc.Graph(figure=fig_3D_TH),])
    if Cluster3D == 2:
        return html.Div([dcc.Graph(figure=fig_3D_DM),])
    
@app.callback(Output('graph_regression', 'children'), 
              Input('Regression_method', 'value'))

def figure_regression(Regression_method):
    if Regression_method == 1:
        return html.Div([dcc.Graph(figure=fig_lin_reg),])
    if Regression_method == 2:
        return html.Div([dcc.Graph(figure=fig_vec_reg),])
    if Regression_method == 3:
        return html.Div([dcc.Graph(figure=fig_RF_reg),])
    
@app.callback(Output('graph_regression_scatter', 'children'), 
              Input('Regression_method', 'value'))
def scatter_regression(Regression_method):
    if Regression_method == 1:
        return html.Div([dcc.Graph(figure=fig_lin_scat),])
    if Regression_method == 2:
        return html.Div([dcc.Graph(figure=fig_vec_scat),])
    if Regression_method == 3:
        return html.Div([dcc.Graph(figure=fig_RF_scat),])

@app.callback(Output('table_Errors', 'children'), 
              Input('Regression_method', 'value'))
def table_Regression_error(Regression_method):
    if Regression_method == 1:
        return html.Div([dcc.Graph(figure=table_LR),])
    if Regression_method == 2:
        return html.Div([dcc.Graph(figure=table_SVR),])
    if Regression_method == 3:
        return html.Div([dcc.Graph(figure=table_RF),])
    
@app.callback(Output('feature_table', 'children'), 
              Input('feature', 'value'))
def table_feature(Regression_method):
    if Regression_method == 1:
        return html.Div([dcc.Graph(figure=table_f1),])
    if Regression_method == 2:
        return html.Div([dcc.Graph(figure=table_f2),])
    if Regression_method == 3:
        return html.Div([dcc.Graph(figure=table_f3),])

if __name__ == '__main__':
    app.run_server(debug=True)
