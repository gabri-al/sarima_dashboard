import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")


dash.register_page(__name__, name='4-Prediction', title='SARIMA | 4-Prediction')

from assets.fig_layout import my_figlayout, train_linelayout, test_linelayout, pred_linelayout
from assets.acf_pacf_plots import acf_pacf

_data_airp = pd.read_csv('data/AirPassengers.csv', usecols = [0,1], names=['Time','Values'], skiprows=1)
_data_airp['Time'] = pd.to_datetime(_data_airp['Time'], errors='raise')
_trainp = 80
idx_split = round(len(_data_airp['Values']) * (_trainp/100)) # Split train-test
_datatrain = _data_airp.iloc[:idx_split+1]
_opts = range(0,int(len(_datatrain['Values'])/2),1)
_opts = list(_opts)

### PAGE LAYOUT ###############################################################################################################

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([html.H3(['Final Model: Fit & Prediction'])], width=12, className='row-titles')
    ]),
    # final-model
    dbc.Row([
        dbc.Col(width=3),
        dbc.Col([html.P(['Final SARIMA(p,d,q; P,D,Q,m) model parameters: '], className='par')], width=6),
        dbc.Col(width=3)
    ]),
    dbc.Row([
        dbc.Col(width=3),
        dbc.Col([dcc.Dropdown(options=_opts, placeholder='p', value=0, clearable=False, searchable=True, persistence=True, persistence_type='memory', id='p-fin')], width=1),
        dbc.Col([dcc.Dropdown(options=_opts, placeholder='d', value=1, clearable=False, searchable=True, persistence=True, persistence_type='memory', id='d-fin')], width=1),
        dbc.Col([dcc.Dropdown(options=_opts, placeholder='q', value=1, clearable=False, searchable=True, persistence=True, persistence_type='memory', id='q-fin')], width=1),
        dbc.Col([dcc.Dropdown(options=_opts, placeholder='P', value=0, clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sp-fin')], width=1),
        dbc.Col([dcc.Dropdown(options=_opts, placeholder='D', value=1, clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sd-fin')], width=1),
        dbc.Col([dcc.Dropdown(options=_opts, placeholder='Q', value=1, clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sq-fin')], width=1),
        dbc.Col([dcc.Dropdown(options=_opts, placeholder='m', value=12, clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sm-fin')], width=1),        
        dbc.Col(width=2)
    ]),
    dbc.Row([], style={'margin':'20px 0px 0px 0px'}),
    dbc.Row([
        dbc.Col(width=2),
        dbc.Col([html.P(['Generate out of sample forecasts for: '], className='par')], width=4),
        dbc.Col([dcc.Dropdown(options=list(range(0,366,1)), placeholder='n', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='n-offset')], width=1),
        dbc.Col([dcc.Dropdown(options=['datapoints','years','months','days','hours'], placeholder='datapoints', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='n-timetype')], width=2),
        dbc.Col(width=3),
    ]),
    dbc.Row([
        dbc.Col([], width = 1),
        dbc.Col([
            dcc.Loading(id='m1-loading', type='circle', children=dcc.Graph(id='fig-pg41', className='my-graph'))
        ], width = 10),
        dbc.Col([], width = 1)
    ], className='row-content'),
    dbc.Row([
        dbc.Col([], width = 1),
        dbc.Col([
            dcc.Loading(id='m2-loading', type='circle', children=dcc.Graph(id='fig-pg42', className='my-graph'))
        ], width = 5),
        dbc.Col([
            dcc.Loading(id='m3-loading', type='circle', children=dcc.Graph(id='fig-pg43', className='my-graph'))
        ], width = 5),
        dbc.Col([], width = 1)
    ])
])

### PAGE CALLBACKS ###############################################################################################################

# Set options & placeholder
@callback(
    Output(component_id='n-timetype', component_property='options'),
    Output(component_id='n-timetype', component_property='placeholder'),
    Input(component_id='n-offset', component_property='value')
)
def init_(_v):
    _data = _data_airp.copy()
    if str(_data.dtypes[0]) in (['<M8[ns]','>M8[ns]','datetime64[ns]']):
        return ['years','months','days','hours'], 'months'
    else:
        return ['datapoints'], 'datapoints'

# Generate predictions & Graph
@callback(
    Output(component_id='fig-pg41', component_property='figure'),
    Output(component_id='fig-pg42', component_property='figure'),
    Output(component_id='fig-pg43', component_property='figure'),
    Input(component_id='p-fin', component_property='value'),
    Input(component_id='d-fin', component_property='value'),
    Input(component_id='q-fin', component_property='value'),
    Input(component_id='sp-fin', component_property='value'),
    Input(component_id='sd-fin', component_property='value'),
    Input(component_id='sq-fin', component_property='value'),
    Input(component_id='sm-fin', component_property='value'),
    Input(component_id='n-offset', component_property='value'),
    Input(component_id='n-timetype', component_property='value')
)
def predict_(_p, _d, _q, _P, _D, _Q, _m, _noffs, _ntype):
    if _noffs is None:
        _noffs = int(0)
    else:
        _noffs = int(_noffs)
    _p = int(_p); _d = int(_d); _q = int(_q); _P = int(_P); _D = int(_D); _Q = int(_Q); _m = int(_m)
    if not _ntype:
        _ntype = 'datapoints'        
    # Get data
    _data = _data_airp.copy()
    _data['Values'] = list(np.log(_data['Values']))
    idx_split = round(len(_data['Values']) * (_trainp/100)) # Split train-test # Split train-test
    _data_train = _data.iloc[:idx_split+1]
    _data_train['_is_train'] = 1
    _data_test = _data.iloc[idx_split+1:]
    _data_test['_is_train'] = 0
    tt = []
    if _ntype != 'datapoints':
        max_t = pd.to_datetime(max(list(_data_test['Time'])))
        if _ntype == 'years':
            for i in range(1, _noffs+1, 1):
                tt.append( max_t + pd.DateOffset( years = i))
        elif _ntype == 'months':
            for i in range(1, _noffs+1, 1):
                tt.append( max_t + pd.DateOffset( months = i))                    
        elif _ntype == 'days':
            for i in range(1, _noffs+1, 1):
                tt.append( max_t + pd.DateOffset( days = i))                    
        elif _ntype == 'hours':
            for i in range(1, _noffs+1, 1):
                tt.append( max_t + pd.DateOffset( hours = i))                    
    elif _ntype == 'datapoints':
        max_t = max(_data_test['Time'])
        [tt.append(max_t+i) for i in range(1,_noffs+1,1)]
    _data_pred = pd.DataFrame() # out of sample predictions        
    _data_pred['Time'] = tt
    _data_pred['Values'] = np.nan
    _data_pred['_is_train'] = 2
    _data_all = pd.concat([_data_train, _data_test, _data_pred], ignore_index=True)

    # Fit model
    _best_model = SARIMAX(endog = _data_train['Values'], order=(_p, _d, _q), seasonal_order=(_P, _D, _Q, _m)).fit(disp=-1)
    # Calculate predictions
    _model_pred = _best_model.get_prediction(start=0, end=len(list(_data_all['Time'])))
    _data_all['Values Predicted'] = _model_pred.predicted_mean
    _data_all['Pred CI lower'] = _model_pred.conf_int(alpha=0.05).iloc[:,0]
    _data_all['Pred CI upper'] = _model_pred.conf_int(alpha=0.05).iloc[:,1]
    _data_all['Values'] = np.exp(_data_all['Values'])
    _data_all['Values Predicted'] = np.exp(_data_all['Values Predicted'])
    _data_all['Pred CI lower'] = np.exp(_data_all['Pred CI lower'])
    _data_all['Pred CI upper'] = np.exp(_data_all['Pred CI upper'])
    _data_all.loc[_data_all['Values Predicted'] <= min(list(_data_all['Values']))/2, 'Values Predicted'] = np.nan # Correcting outliers
    _data_all.loc[_data_all['Values Predicted'] >= max(list(_data_all['Values']))*2, 'Values Predicted'] = np.nan
    _data_all.loc[_data_all['Pred CI lower'] <= min(list(_data_all['Values']))/2, 'Pred CI lower'] = np.nan
    _data_all.loc[_data_all['Pred CI upper'] >= max(list(_data_all['Values']))*2, 'Pred CI upper'] = np.nan
    
    # Show model results
    fig1 = go.Figure(layout=my_figlayout)
    # CIs
    fig1.add_trace(go.Scatter(x=_data_all['Time'], y=_data_all['Pred CI lower'], mode='lines',
                    line = dict(width=0.5, color = 'rgba(255,255,255,0)'), name='95%-CI', showlegend=False))
    fig1.add_trace(go.Scatter(x=_data_all['Time'], y=_data_all['Pred CI upper'], mode='lines', fill='tonexty',
                            line = dict(width=0.5, color = 'rgba(255,255,255,0)'),
                            fillcolor = 'rgba(178, 211, 194,0.11)', name='95%-CI'))
    # Lines
    fig1.add_trace(go.Scatter(x=_data_all.loc[_data_all['_is_train']==1, 'Time'],
                                y=_data_all.loc[_data_all['_is_train']==1, 'Values'], mode='lines', name='Train', line=train_linelayout))
    fig1.add_trace(go.Scatter(x=_data_all.loc[_data_all['_is_train']==0, 'Time'],
                                y=_data_all.loc[_data_all['_is_train']==0, 'Values'], mode='lines', name='Test', line=test_linelayout))
    fig1.add_trace(go.Scatter(x=_data_all['Time'], y=_data_all['Values Predicted'], mode='lines', name='Predictions', line=pred_linelayout))
    fig1.update_xaxes(title_text = 'Time')
    fig1.update_yaxes(title_text = 'Values')
    fig1.update_layout(title="Final Model Results", height=500)
    #_min_yrange = round( min(list(_data_train['Values'])) - 1.25 * min(list(_data_train['Values'])) )
    #_max_yrange = round( max(list(_data_train['Values'])) + 1.25 * max(list(_data_train['Values'])) )
    #fig1.update_yaxes(range = [_min_yrange, _max_yrange])

    # Show residuals ACF and PACF
    resid_df = pd.DataFrame(_best_model.resid, columns = ['Residuals'])
    fig_2, fig_3 = acf_pacf(resid_df, 'Residuals')
    fig_2.update_layout(title="Model Residuals: Autocorrelation (ACF)")
    fig_3.update_layout(title="Model Residuals: Partial Autocorrelation (PACF)")

    return fig1, fig_2, fig_3