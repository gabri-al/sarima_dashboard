import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def sarima_grid_search(_data, _params):
    _scores = []
    _pars = []
    for par in _params:
        try: 
            model = SARIMAX(endog=_data['Values'],
                            order=(par[0], par[1], par[2]),
                            seasonal_order=(par[3], par[4], par[5], par[6]),
                            simple_differencing=True).fit(disp=-1) # disp to silence verbose
        except:
            continue

        aic_score = model.aic
        _scores.append(aic_score)
        _pars.append(str(par))
    _scores_df = pd.DataFrame({'Params SARIMA (p,d,q) (P,D,Q)m': _pars, 'AIC score': _scores})
    _scores_df = _scores_df.sort_values(by='AIC score', ascending=True).reset_index(drop=True)
    return _scores_df