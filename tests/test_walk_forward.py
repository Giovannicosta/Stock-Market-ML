import unittest
import numpy as np
import pandas as pd
from src.walk_forward import spy_features, build_panel, train_before, folds_for, portfolio, experiment

def fake_prices(n=1000):
    rng=np.random.default_rng(42)
    dates=pd.bdate_range('2015-01-01',periods=n)
    result={}
    market=rng.normal(.0004,.01,n)
    for ticker in ['SPY','AAPL','MSFT','JPM']:
        close=100*np.exp(np.cumsum(market+rng.normal(0,.005,n)))
        opn=close*np.exp(rng.normal(0,.003,n))
        result[ticker]=pd.DataFrame(dict(Open=opn,High=np.maximum(opn,close)*1.01,
            Low=np.minimum(opn,close)*.99,Close=close,
            Volume=rng.integers(1000000,5000000,n)),index=dates)
    return result

class WalkForwardTests(unittest.TestCase):
    def test_market_feature_uses_past_only(self):
        prices=fake_prices()
        original=spy_features(prices['SPY'])
        changed=prices['SPY'].copy()
        changed.iloc[-1,changed.columns.get_loc('Close')]*=10
        adjusted=spy_features(changed)
        self.assertAlmostEqual(original['spy_trend_200'].iloc[-2],adjusted['spy_trend_200'].iloc[-2])

    def test_training_target_purge(self):
        panel,labeled,_=build_panel(fake_prices())
        boundaries,holdout=folds_for(labeled['Date'])
        for start,end in boundaries:
            train=train_before(labeled,start)
            self.assertTrue((train['label_end'] < start).all())
            self.assertTrue((labeled.loc[(labeled['Date']>=start)&(labeled['Date']<end),'Date'] < holdout).all())

    def test_trading_fees_and_cash(self):
        dates=pd.bdate_range('2025-01-01',periods=3)
        table=pd.DataFrame(dict(Date=dates,ticker=['AAA']*3,
            prediction=[.01,.01,.01],target=[.02,-.01,.03],
            spy_target=[.01,.01,.01],regime=['bull_normal_vol']*3))
        _,out=portfolio(table,0,fee_bps=10)
        self.assertAlmostEqual(out['turnover_sides'],2,places=6)
        days,flat=portfolio(table,.5,fee_bps=10)
        self.assertEqual(flat['ml']['total_return'],0)
        self.assertTrue((days['ml']==0).all())

    def test_end_to_end(self):
        result=experiment(fake_prices(),n_folds=3)
        self.assertEqual(len(result['folds']),3)
        self.assertTrue(np.isfinite(result['heldout']['model_mae']))
        self.assertIn(result['selection']['model'],['ridge','gradient_boosting'])
        self.assertEqual(len(result['assets']),3)

if __name__=='__main__':
    unittest.main()
