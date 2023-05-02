import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pmdarima.utils import diff
from statsmodels.tsa.stattools import adfuller

dash.register_page(__name__, name='2-Stationarity', title='SARIMA | 2-Stationarity')

from assets.fig_layout import my_figlayout, my_linelayout
from assets.acf_pacf_plots import acf_pacf

_data_airp = pd.read_csv('data/AirPassengers.csv', usecols = [0,1], names=['Time','Values'], skiprows=1)
_data_airp['Time'] = pd.to_datetime(_data_airp['Time'], errors='raise')

### PAGE LAYOUT ###############################################################################################################

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([html.H3(['Transform dataset to make it Stationary'])], width=12, className='row-titles')
    ]),
    # Transformations
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            dcc.Checklist(['1) Apply log'], persistence=True, persistence_type='session', id='log-check')
        ], width = 2),
        dbc.Col([
            dcc.Checklist(['2) Apply difference'], persistence=True, persistence_type='session', id='d1-check'),
            dcc.Dropdown(options=[], value='', clearable=True, disabled=True, searchable=True, placeholder='Choose lag', persistence=True, persistence_type='session', id='d1-dropdown')
        ], width = 2),
        dbc.Col([
            dcc.Checklist(['3) Apply difference'], persistence=True, persistence_type='session', id='d2-check'),
            dcc.Dropdown(options=[], value='', clearable=True, disabled=True, placeholder='Choose lag', persistence=True, persistence_type='session', id='d2-dropdown')
        ], width = 2),
        dbc.Col([
            dcc.Checklist(['4) Apply difference'], persistence=True, persistence_type='session', id='d3-check'),
            dcc.Dropdown(options=[], value='', clearable=True, disabled=True, placeholder='Choose lag', persistence=True, persistence_type='session', id='d3-dropdown')
        ], width = 2),
        dbc.Col([], width = 2),
        
    ], className='row-content'),
    # Augmented Dickey-Fuller test
    dbc.Row([
        dbc.Col([], width = 3),
        dbc.Col([html.P(['Augmented Dickey-Fuller test: '], className='par')], width = 2),
        dbc.Col([
            dcc.Loading(id='p2-1-loading', type='circle', children=html.Div([], id = 'stationarity-test'))
        ], width = 4),
        dbc.Col([], width = 3)
    ]),
    # Graphs
    dbc.Row([ 
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-transformed', className='my-graph'))
        ], width=6, className='multi-graph'),
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-acf', className='my-graph'))
        ], width=6, className='multi-graph')
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-boxcox', className='my-graph'))
        ], width = 6, className='multi-graph'),
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-pacf', className='my-graph'))
        ], width=6, className='multi-graph')
    ])

])


### PAGE CALLBACKS ###############################################################################################################

#################
# Complete dropdown config

# Activate dropdowns (repeat three times, one for each dropdown)
@callback(
    Output(component_id='d1-dropdown', component_property='disabled'),
    Output(component_id='d1-dropdown', component_property='options'),
    Input(component_id='d1-check', component_property='value')
)
def dropdown_activation(_check):
    _data = _data_airp
    # Calculate disabled
    if _check:
        _disabled = False
    else:
        _disabled = True    
    # Calculate options
    _opts = range(1,int(len(_data['Values'])/2),1)

    return _disabled, list(_opts)

@callback(
    Output(component_id='d2-dropdown', component_property='disabled'),
    Output(component_id='d2-dropdown', component_property='options'),
    Input(component_id='d2-check', component_property='value')
)
def dropdown_activation(_check):
    _data = _data_airp
    # Calculate disabled
    if _check:
        _disabled = False
    else:
        _disabled = True    
    # Calculate options
    _opts = range(1,int(len(_data['Values'])/2),1)

    return _disabled, list(_opts)

@callback(
    Output(component_id='d3-dropdown', component_property='disabled'),
    Output(component_id='d3-dropdown', component_property='options'),
    Input(component_id='d3-check', component_property='value')
)
def dropdown_activation(_check):
    _data = _data_airp
    # Calculate disabled
    if _check:
        _disabled = False
    else:
        _disabled = True    
    # Calculate options
    _opts = range(1,int(len(_data['Values'])/2),1)

    return _disabled, list(_opts)

#################
# Apply transformations to the data
@callback(
    Output(component_id='stationarity-test', component_property='children'),
    Output(component_id='fig-transformed', component_property='figure'),
    Output(component_id='fig-boxcox', component_property='figure'),
    Output(component_id='fig-acf', component_property='figure'),
    Output(component_id='fig-pacf', component_property='figure'),
    Input(component_id='log-check', component_property='value'),
    Input(component_id='d1-check', component_property='value'),
    Input(component_id='d1-dropdown', component_property='value'),
    Input(component_id='d2-check', component_property='value'),
    Input(component_id='d2-dropdown', component_property='value'),
    Input(component_id='d3-check', component_property='value'),
    Input(component_id='d3-dropdown', component_property='value')
)
def data_transform(_logtr, _d1check, _d1v, _d2check, _d2v, _d3check, _d3v):
    _data = _data_airp.copy()
    # Transform the data
    if _logtr:
        _min_value = min(_data['Values']) # correct for 0 or negative values
        if _min_value == 0:
            _data['Values'] = _data['Values'] + 0.5
        elif _min_value < 0:
            _data['Values'] = _data['Values'] + np.abs(_min_value) + 0.5
        _data['Values'] = list(np.log(_data['Values'])) # apply log transformation
    if _d1check and _d1v:
        _dvalues = diff(list(_data['Values']), lag=_d1v, differences=1)
        _data = _data.iloc[_d1v:]
        _data['Values'] = _dvalues
    if _d2check and _d2v:
        _dvalues = diff(list(_data['Values']), lag=_d2v, differences=1)
        _data = _data.iloc[_d2v:]
        _data['Values'] = _dvalues
    if _d3check and _d3v:
        _dvalues = diff(list(_data['Values']), lag=_d3v, differences=1)
        _data = _data.iloc[_d3v:]
        _data['Values'] = _dvalues
    # Perform test
    stat_test = adfuller(_data['Values'])
    pv = stat_test[1]
    if pv <= .05: # p-value
        #Stationary
        _test_output = dbc.Alert(children=['Test p-value: {:.4f}'.format(pv),html.Br(),'The data is ',html.B(['stationary'], className='alert-bold')], color='success')
    else:
        _test_output =dbc.Alert(children=['Test p-value: {:.4f}'.format(pv),html.Br(),'The data is ',html.B(['not stationary'], className='alert-bold')], color='danger')
    # Charts
    # Transformed data linechart
    fig_1 = go.Figure(layout=my_figlayout)
    fig_1.add_trace(go.Scatter(x=_data['Time'], y=_data['Values'], line=dict()))
    fig_1.update_layout(title='Transformed Data Linechart', xaxis_title='Time', yaxis_title='Values')
    fig_1.update_traces(overwrite=True, line=my_linelayout)

    # Box-Cox plot
    v_ = np.array(_data['Values'])
    rolling_avg = []; rolling_std = []
    for i in range(0,len(v_),1):
        rolling_avg.append(v_[:i+1].mean())
        rolling_std.append(v_[:i+1].std())
    _data['rolling_avg'] = rolling_avg
    _data['rolling_std'] = rolling_std
    fig_2 = go.Figure(layout=my_figlayout)
    _hovertemplate = (
        "<i>Rolling Avg</i>: %{x:.2f}<br>"+
        "<i>Rolling Std</i>: %{y:.2f}"+
        "<extra></extra>")
    fig_2.add_trace(go.Scatter(x=_data['rolling_avg'], y=_data['rolling_std'], mode='markers', marker_size=4, marker_color='#3DED97', hovertemplate=_hovertemplate))
    fig_2.update_layout(title='Box-Cox Plot', xaxis_title='Rolling Average', yaxis_title='Rolling Standard Deviation')

    # ACF, PACF
    fig_3, fig_4 = acf_pacf(_data, 'Values')

    return _test_output, fig_1, fig_2, fig_3, fig_4