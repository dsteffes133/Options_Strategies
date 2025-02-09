import pandas as pd

def sub_derivs(df):
    df['vanna'] = df['vega']*(df['delta']/df['implied_volatility'])
    df['charm'] = -1*(df['theta']*df['delta'])/df['last']
    return df['vanna'], df['charm']