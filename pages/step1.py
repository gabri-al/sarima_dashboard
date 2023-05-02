import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

dash.register_page(__name__, name='1-Data set up', title='SARIMA | 1-Data set up')

from assets.fig_layout import my_figlayout, my_linelayout

_data_airp = pd.read_csv('data/AirPassengers.csv', usecols = [0,1], names=['Time','Values'], skiprows=1)
_data_airp['Time'] = pd.to_datetime(_data_airp['Time'], errors='raise')

### PAGE LAYOUT ###############################################################################################################

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([html.H3(['Your dataset'])], width=12, className='row-titles')
    ]),

    # data input
    dbc.Row([
        dbc.Col([], width = 3),
        dbc.Col([html.P(['Select a dataset:'], className='par')], width=2),
        dbc.Col([
            dcc.RadioItems(['Air passenger'], value = 'Air passenger', persistence=True, persistence_type='session', id='radio-dataset')
        ], width=4),
        dbc.Col([], width = 3)
    ], className='row-content'),

    # raw data fig
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            dcc.Loading(id='p1_1-loading', type='circle', children=dcc.Graph(id='fig-pg1', className='my-graph'))
        ], width = 8),
        dbc.Col([], width = 2)
    ], className='row-content')
    
])

### PAGE CALLBACKS ###############################################################################################################

# Update fig
@callback(
    Output(component_id='fig-pg1', component_property='figure'),
    Input(component_id='radio-dataset', component_property='value')
)
def plot_data(value):
    fig = None

    if value == 'Air passenger':
        _data = _data_airp

    fig = go.Figure(layout=my_figlayout)
    fig.add_trace(go.Scatter(x=_data['Time'], y=_data['Values'], line=dict()))

    fig.update_layout(title='Dataset Linechart', xaxis_title='Time', yaxis_title='Values', height = 500)
    fig.update_traces(overwrite=True, line=my_linelayout)

    return fig