import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf
import numpy as np
import pandas as pd

from assets.fig_layout import my_figlayout

def acf_pacf_plot(_arr, _upper_ci, _lower_ci, is_acf=True):
    fig = go.Figure(layout=my_figlayout)
    _hovertemplate = (
        "<i>Lag</i>: %{x}<br>"+
        "<i>Correlation</i>: %{y:.3f}"+
        "<extra></extra>")
    # Add vertical lines
    [fig.add_scatter(x=(x,x), y=(0,_arr[0][x]), mode='lines', line_color='#0b2124', hoverinfo='none') for x in range(len(_arr[0]))]
    # Add top lines markers
    fig.add_scatter(x=np.arange(len(_arr[0])), y=_arr[0], mode='markers', marker_color='#3DED97', hovertemplate=_hovertemplate)
    # Add CI
    fig.add_scatter(x=np.arange(len(_arr[0])), y=_upper_ci, mode='lines', line_color='rgba(255,255,255,0)', hoverinfo='none')
    fig.add_scatter(x=np.arange(len(_arr[0])), y=_lower_ci, mode='lines', fillcolor='rgba(178, 211, 194,0.11)', fill='tonexty', line_color='rgba(255,255,255,0)', hoverinfo='none')

    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,len(_arr[0])])
    fig.update_yaxes(range=[-1.1,1.1])
    if is_acf:
        fig.update_layout(title="Autocorrelation (ACF)")
    else:
        fig.update_layout(title="Partial Autocorrelation (PACF)")
    fig.update_yaxes(zerolinecolor='#000000')

    return fig

def acf_pacf(_df, _colname):
    lags_ = min(int((len(_df[_colname])-1)/2),75)
    acf_corr_array = acf(list(_df[_colname]), nlags = min(int((len(_df[_colname])-2)/2),75), alpha=0.05) # Includes 95% CI
    acf_lower_ci = acf_corr_array[1][:,0] - acf_corr_array[0]
    acf_upper_ci = acf_corr_array[1][:,1] - acf_corr_array[0]
    pacf_corr_array = pacf(list(_df[_colname]), nlags = min(int((len(_df[_colname])-2)/2),75), alpha=0.05) # Includes 95% CI
    pacf_lower_ci = pacf_corr_array[1][:,0] - pacf_corr_array[0]
    pacf_upper_ci = pacf_corr_array[1][:,1] - pacf_corr_array[0]
    
    corr_df = pd.DataFrame()
    corr_df['lag'] = lags_
    corr_df['ACF'] = acf_corr_array[0]
    corr_df['PACF'] = pacf_corr_array[0]
    
    #Plot ACF
    fig_acf = acf_pacf_plot(acf_corr_array, acf_upper_ci, acf_lower_ci, True)
    fig_pacf = acf_pacf_plot(pacf_corr_array, pacf_upper_ci, pacf_lower_ci, False)
    
    return fig_acf, fig_pacf