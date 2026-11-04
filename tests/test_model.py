import unittest

import numpy as np
import pandas as pd

from src.model import (FEATURES, GAP_SESSIONS, backtest, evaluate_one,
                       prepare_features, split_chronological)


class MarketModelTests(unittest.TestCase):
    @staticmethod
    def synthetic_prices(n=800):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range('2010-01-01', periods=n)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
        opening = close * np.exp(rng.normal(0, 0.003, n))
        high = np.maximum(opening, close) * 1.01
        low = np.minimum(opening, close) * 0.99
        volume = rng.integers(1000000, 5000000, n)
        return pd.DataFrame({'Open': opening, 'High': high, 'Low': low,
                             'Close': close, 'Volume': volume}, index=dates)

    def test_target_uses_next_two_opens(self):
        prices = self.synthetic_prices(850)
        frame = prepare_features(prices)
        t = 250
        self.assertAlmostEqual(frame['target'].iloc[t],
                               prices['Open'].iloc[t + 2] / prices['Open'].iloc[t + 1] - 1)
        self.assertTrue(np.isnan(frame['target'].iloc[-1]))
        self.assertTrue(np.isnan(frame['target'].iloc[-2]))

    def test_chronological_and_purged(self):
        frame = prepare_features(self.synthetic_prices()).dropna()
        train, val, fit_for_test, test = split_chronological(frame)
        self.assertLess(train.index[-1], val.index[0])
        self.assertLess(val.index[-1], test.index[0])
        self.assertEqual(len(frame.loc[train.index[-1]:val.index[0]]) - 2, GAP_SESSIONS)
        self.assertLess(fit_for_test.index[-1], test.index[0])

    def test_zero_trades_and_costs(self):
        y = np.array([0.02, -0.01, 0.03, -0.02])
        flat = backtest(y, np.full(len(y), -1.0), 0.0, 10.0)
        self.assertEqual(flat['total_return'], 0.0)
        self.assertEqual(flat['buy_count'], 0)
        single = backtest(y, np.array([1, -1, -1, -1]), 0.0, 10.0)
        self.assertEqual(single['turnover_sides'], 2)
        self.assertAlmostEqual(single['total_return'], (1 + 0.02 - 0.001) * (1 - 0.001) - 1)

    def test_full_evaluation(self):
        output = evaluate_one(self.synthetic_prices(), 'TEST')
        self.assertIn(output['selected_model'], ('ridge', 'hist_gradient_boosting'))
        self.assertTrue(np.isfinite(output['test_backtest']['total_return']))
        self.assertIn(output['forecast']['research_signal'], ('BUY', 'FLAT/SELL'))
        self.assertGreater(output['rows']['test'], 100)


if __name__ == '__main__':
    unittest.main()
