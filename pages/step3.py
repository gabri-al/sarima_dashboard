import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from itertools import product

dash.register_page(__name__, name='3-Model Selection', title='SARIMA | 3-Model Selection')

from assets.sarima_gridsearch import sarima_grid_search

_data_airp = pd.read_csv('data/AirPassengers.csv', usecols = [0,1], names=['Time','Values'], skiprows=1)
_data_airp['Time'] = pd.to_datetime(_data_airp['Time'], errors='raise')

### PAGE LAYOUT ###############################################################################################################

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([
            html.H3(['Hyperparameter Tuning']),
            html.P([html.B(['SARIMA(p,d,q; P,D,Q,m) grid search'])], className='par')
        ], width=12, className='row-titles')
    ]),
    # train-test split
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([html.P(['Select the ',html.B(['train']),' percentage: '], className='par')], width = 4),
        dbc.Col([
            html.Div([
                dcc.Slider(50, 95, 5, value=80, marks=None, tooltip={"placement": "bottom", "always_visible": True}, id='train-slider', persistence=True, persistence_type='session')
            ], className = 'slider-div')
        ], width = 3),
        dbc.Col([], width = 3),
    ]),
    # model params
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Row([
                    dbc.Col([], width = 1),
                    dbc.Col([html.P(['Set ',html.B(['p, d, q']),' parameters range (from, to)'], className='par')], width = 10),
                    dbc.Col([], width = 1),
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['p']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='p-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='p-to')], width=4),
                    dbc.Col([], width=1)
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['d']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='d-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='d-to')], width=4),
                    dbc.Col([], width=1)
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['q']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='q-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='q-to')], width=4),
                    dbc.Col([], width=1)
                ])                
            ], className = 'div-hyperpar')
        ], width = 6, className = 'col-hyperpar'),
        dbc.Col([
            html.Div([
                dbc.Row([
                    dbc.Col([], width = 1),
                    dbc.Col([html.P(['Set ',html.B(['P, D, Q, m']),' seasonal parameters range (from, to)'], className='par')], width = 10),
                    dbc.Col([], width = 1),   
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['P']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sp-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sp-to')], width=4),
                    dbc.Col([], width=1)
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['D']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sd-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sd-to')], width=4),
                    dbc.Col([], width=1)
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['Q']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sq-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sq-to')], width=4),
                    dbc.Col([], width=1)
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['m']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sm-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sm-to')], width=4),
                    dbc.Col([], width=1)
                ])
            ], className = 'div-hyperpar')
        ], width = 6, className = 'col-hyperpar')
    ], style={'margin':'20px 0px 0px 0px'}),
    dbc.Row([
        dbc.Col([], width=3),
        dbc.Col([
            html.P(['Grid Search combinations: ', html.B([], id='comb-nr')], className='par')
        ], width=3),
        dbc.Col([
            html.Button('Start Grid Search', id='start-gs', n_clicks=0, title='The grid search may take several minutes', className='my-button')
        ], width=3, style={'text-align':'left', 'margin':'5px 1px 1px 1px'}),
        dbc.Col([], width=3)
    ]),
    # grid Search results
    dbc.Row([
        dbc.Col([], width = 4),
        dbc.Col([
            dcc.Loading(id='gs-loading', type='circle', children=html.Div(id='gs-results'))
        ], width = 4),
        dbc.Col([], width = 4),
    ])
])

### PAGE CALLBACKS ###############################################################################################################


# Dropdowns options
@callback(
    Output(component_id='p-from', component_property='options'),
    Output(component_id='d-from', component_property='options'),
    Output(component_id='q-from', component_property='options'),
    Output(component_id='sp-from', component_property='options'),
    Output(component_id='sd-from', component_property='options'),
    Output(component_id='sq-from', component_property='options'),
    Output(component_id='sm-from', component_property='options'),
    Input(component_id='train-slider', component_property='value')
)
def dropdown_opt_from(_trainp):
    _data = _data_airp.copy()
    if _trainp:
        idx_split = round(len(_data['Values']) * int(_trainp)/100) # Split train-test
        _data = _data.iloc[:idx_split+1]
    _opts = range(0,int(len(_data['Values'])/2),1)
    _opts = list(_opts)
    return _opts, _opts, _opts, _opts, _opts, _opts, _opts

@callback(
    Output(component_id='p-to', component_property='options'),
    Input(component_id='p-from', component_property='value')
)
def dropdown_opt_to(_from):
    _data = _data_airp.copy()
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='d-to', component_property='options'),
    Input(component_id='d-from', component_property='value')
)
def dropdown_opt_to(_from):
    _data = _data_airp.copy()
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='q-to', component_property='options'),
    Input(component_id='q-from', component_property='value')
)
def dropdown_opt_to(_from):
    _data = _data_airp.copy()
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='sp-to', component_property='options'),
    Input(component_id='sp-from', component_property='value')
)
def dropdown_opt_to(_from):
    _data = _data_airp.copy()
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='sd-to', component_property='options'),
    Input(component_id='sd-from', component_property='value')
)
def dropdown_opt_to(_from):
    _data = _data_airp.copy()
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='sq-to', component_property='options'),
    Input(component_id='sq-from', component_property='value')
)
def dropdown_opt_to(_from):
    _data = _data_airp.copy()
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='sm-to', component_property='options'),
    Input(component_id='sm-from', component_property='value')
)
def dropdown_opt_to(_from):
    _data = _data_airp.copy()
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

# Grid Search & Show combinations
@callback(
    Output(component_id='comb-nr', component_property='children'),
    Output(component_id='gs-results', component_property='children'),
    Output(component_id='browser-memo', component_property='data', allow_duplicate=True),
    Input(component_id='train-slider', component_property='value'),
    Input(component_id='start-gs', component_property='n_clicks'),
    Input(component_id='p-from', component_property='value'),
    Input(component_id='p-to', component_property='value'),
    Input(component_id='d-from', component_property='value'),
    Input(component_id='d-to', component_property='value'),
    Input(component_id='q-from', component_property='value'),
    Input(component_id='q-to', component_property='value'),
    Input(component_id='sp-from', component_property='value'),
    Input(component_id='sp-to', component_property='value'),
    Input(component_id='sd-from', component_property='value'),
    Input(component_id='sd-to', component_property='value'),
    Input(component_id='sq-from', component_property='value'),
    Input(component_id='sq-to', component_property='value'),
    Input(component_id='sm-from', component_property='value'),
    Input(component_id='sm-to', component_property='value'),
    State(component_id='browser-memo', component_property='data'),
    prevent_initial_call='initial_duplicate'
)
def grid_search_results(_trainp, _nclicks, p_from,p_to,d_from,d_to,q_from,q_to,sp_from,sp_to,sd_from,sd_to,sq_from,sq_to,sm_from,sm_to,_memo):
    # Calculate combinations
    _p = list(range(int(p_from), int(p_to)+1, 1))
    _d = list(range(int(d_from), int(d_to)+1, 1))
    _q = list(range(int(q_from), int(q_to)+1, 1))
    _P = list(range(int(sp_from), int(sp_to)+1, 1))
    _D = list(range(int(sd_from), int(sd_to)+1, 1))
    _Q = list(range(int(sq_from), int(sq_to)+1, 1))
    _m = list(range(int(sm_from), int(sm_to)+1, 1))
    _combs = list(product(_p, _d, _q, _P, _D, _Q, _m))
    # Split data
    _datatrain = None; _datatest = None
    if _trainp:
        _data = _data_airp.copy()
        _data['Values'] = list(np.log(_data['Values']))
        idx_split = round(len(_data['Values']) * int(_trainp)/100) # Split train-test
        _datatrain = _data.iloc[:idx_split+1]
        _datatest = _data.iloc[idx_split+1:]
    # Grid search
    if int(_nclicks) > 0 and _datatrain is not None and _datatest is not None:
        _gs_res = sarima_grid_search(_data, _combs)
        _gs_res_tbl = _gs_res.iloc[:10]
        _gs_res_tbl.columns = ['Parameters (p,d,q)(P,D,Q)m', 'AIC Score']
        _gs_res_tbl['AIC Score'] = round(_gs_res_tbl['AIC Score'], 3)
        if 'grid_search_results' in _memo.keys():
            _memo.pop('grid_search_results')
        _memo['grid_search_results'] = _gs_res_tbl.to_dict('records')
    if 'grid_search_results' in _memo.keys():
        _gs_res_tbl = pd.DataFrame(_memo['grid_search_results'])
        tbl_ = dbc.Table.from_dataframe(_gs_res_tbl, index=False, striped=False, bordered=True, hover=True, size='sm')
        title_ = html.P([html.B(['Top-10 models by AIC score'])], className='par')
        _res = [html.Hr([], className = 'hr-footer'), title_, tbl_]
    else:
        _res = None
    return len(_combs), _res, _memo