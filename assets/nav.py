from dash import html
import dash_bootstrap_components as dbc
import dash

_nav = dbc.Container([
	dbc.Row([
        dbc.Col([
            html.Div([
                html.I(className="fa-solid fa-chart-simple fa-2x")],
		    className='logo')
        ], width = 4),
        dbc.Col([html.H1(['Sarima Tuner'], className='app-brand')], width = 8)
	]),
	dbc.Row([
        dbc.Nav(
	        [dbc.NavLink(page["name"], active='exact', href=page["path"]) for page in dash.page_registry.values()],
	        vertical=True, pills=True, class_name='my-nav')
    ])
])