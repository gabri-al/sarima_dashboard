import plotly.graph_objects as go

###### FIG LAYOUT
font_style = {
    'color' : '#f6f6f6'
}

margin_style = {
    'b': 10,
    'l': 50,
    'r': 8,
    't': 50,
    'pad': 0
}

xaxis_style = {
    'linewidth' : 1,
    'linecolor' : 'rgba(0, 0, 0, 0.35%)',
    'showgrid' : False,
    'zeroline' : False
}

yaxis_style = {
    'linewidth' : 1,
    'linecolor' : 'rgba(0, 0, 0, 0.35%)',
    'showgrid' : True,
    'gridwidth' : 1,
    'gridcolor' : 'rgba(0, 0, 0, 0.11%)',
    'zeroline' : False
}

my_figlayout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)', # Figure background is transparend and controll by css on dcc.Graph() components
    plot_bgcolor='rgba(0,0,0,0)',
    font = font_style,
    margin = margin_style,
    xaxis = xaxis_style,
    yaxis = yaxis_style,
    height = 300
)

###### TRACES LAYOUT
my_linelayout = {
    'width' : 3,
    'color' : '#3DED97'
}

train_linelayout = {
    'width' : 2,
    'color' : '#3DED97'
}

test_linelayout = {
    'width' : 2,
    'color' : '#3DED97',
    'dash' : 'dot'
}

pred_linelayout = {
    'width' : 2,
    'color' : '#042f33'
}