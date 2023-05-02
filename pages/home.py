import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home', title='SARIMA | Home')

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([
            html.H3(['Welcome!']),
            html.P([html.B(['App Overview'])], className='par')
        ], width=12, className='row-titles')
    ]),
    # Guidelines
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            html.P([html.B('1) Built-in dataset'),html.Br(),
                    'The default dataset used is Air Passenger. The app could work with any .csv file.'], className='guide'),
            html.P([html.B('2) Apply transformations to make the data stationary'),html.Br(),
                    'The tools available on the page are: log and differencing, the Box-Cox plot and the A. Dickey Fuller test.',html.Br(),
                    'Once the data is stationary, check the ACF and PACF plots for suitable model parameters.'], className='guide'),
            html.P([html.B('3) Perform a SARIMA model grid search'),html.Br(),
                    'Choose the train-test split and provide from-to ranges for any parameter.'
                    'The seasonality component of the model can be excluded by leaving all right-hand parameters to 0.',html.Br(),
                    'The 10 top-performer models (according to the AIC score), are shown.'], className='guide'),
            html.P([html.B('4) Set up your final model'),html.Br(),
                    'The parameters for the best model from the previous step are suggested.',html.Br(),
                    'The SARIMA model with the input parameters is automatically fitted to the train data; predictions are made for the train and test sets',html.Br(),
                    'The model residuals ACF and PACF are shown.'], className='guide')
        ], width = 8),
        dbc.Col([], width = 2)
    ])
])