import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd
import plotly.graph_objects as go

dash.register_page(__name__, name='1-Data set up', title='SARIMA | 1-Data set up')

from assets.fig_layout import my_figlayout, my_linelayout

### PAGE LAYOUT ###############################################################################################################

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([html.H3(['Pick your dataset'])], width=12, className='row-titles')
    ]),

    # data input
    dbc.Row([
        dbc.Col([], width = 3),
        dbc.Col([html.P(['Select a dataset:'], className='par')], width=2),
        dbc.Col([
            dcc.RadioItems(['Default dataset (Air passenger)','Import own file'], value = 'Default dataset (Air passenger)', persistence=True, persistence_type='session', id='radio-dataset')
        ], width=4),
        dbc.Col([], width = 3)
    ], className='row-content'),

    dbc.Row([
        dbc.Col([], width = 3), dbc.Col([dcc.Upload([], className='data-upload-invisible', id='data-upload')], width = 6), dbc.Col([], width = 3),
    ], className='row-content'),

    # raw data fig
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([dcc.Graph(id='fig-pg1', className='my-graph')], width = 8),
        dbc.Col([], width = 2)
    ], className='row-content')
    
])

### PAGE CALLBACKS ###############################################################################################################

# Show upload area
@callback(
    Output(component_id='data-upload', component_property='children'),
    Output(component_id='data-upload', component_property='className'),
    Input(component_id='radio-dataset', component_property='value')
)
def show_import(value):
    _children = None; _class = 'data-upload-invisible'
    if value=='Import own file':
        _children =  'Drag or Select a .csv file with two col: Time, Values'
        _class = 'data-upload-area'
    return _children, _class

# Handle default data
@callback(
    Output(component_id='browser-memo', component_property='data'),
    State(component_id='browser-memo', component_property='data'),
    Input(component_id='radio-dataset', component_property='value')
)
def data_upload(_memo, value):
    if value == 'Default dataset (Air passenger)': # Custom data won't be overwritten thanks to radio-items persistence
        _data = pd.read_csv('data/AirPassengers.csv', usecols = [0,1], names=['Time','Values'], skiprows=1)
        _data['Time'] = pd.to_datetime(_data['Time'], errors='raise')
        _memo['original dataset'] = _data.to_dict('records')
        _memo['is_time_numeric'] = False
    return _memo

# Handle custom data
@callback(
    Output(component_id='browser-memo', component_property='data', allow_duplicate=True),
    State(component_id='browser-memo', component_property='data'),
    State(component_id='radio-dataset', component_property='value'),
    Input(component_id='data-upload', component_property='contents'),
    prevent_initial_call='initial_duplicate'
)
def data_upload(_memo, value, contents):
    if value == 'Import own file':
        if contents is not None:
            c_type, c_data = contents.split(',')
            if 'csv' in c_type:
                decoded_data = base64.b64decode(c_data)
                _data = pd.read_csv(io.StringIO(decoded_data.decode('utf-8')), usecols = [0,1], names=['Time','Values'], skiprows=1)
                _databkp = _data.copy()
                #print("Custom dataset imported successfully")
                is_time_numeric = True
                _time_col = list(_data['Time'])
                for v in _time_col:
                    if not str(v).isnumeric():
                        is_time_numeric = False
                        break

                if not is_time_numeric:
                    try:
                        _data['Time'] = pd.to_datetime(_data['Time'], errors='raise')
                        #print("Timestamp formatted successfully")
                    except:
                        _data = _databkp.copy()
                if 'original dataset' in _memo.keys():
                    _memo.pop('original dataset')
                _memo['original dataset'] = _data.to_dict('records')
                if 'is_time_numeric' in _memo.keys():
                    _memo.pop('is_time_numeric')
                _memo['is_time_numeric'] = is_time_numeric

    return _memo

# Update fig
@callback(
    Output(component_id='fig-pg1', component_property='figure'),
    Input(component_id='browser-memo', component_property='data')
)
def plot_data(_memo):
    fig = None

    if 'original dataset' in _memo.keys():
        _data = pd.DataFrame(_memo['original dataset'])
        fig = go.Figure(layout=my_figlayout)
        fig.add_trace(go.Scatter(x=_data['Time'], y=_data['Values'], line=dict()))

        fig.update_layout(title='Dataset Linechart', xaxis_title='Time', yaxis_title='Values', height = 500)
        fig.update_traces(overwrite=True, line=my_linelayout)

    return fig