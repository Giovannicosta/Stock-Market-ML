"""V3 regression tests with generated prices (not claims of market performance)."""
import unittest

import numpy as np
import pandas as pd

from src.v3 import (build_weekly_panel, experiment, forecast_models, performance,
                    rolling_folds, rotation_backtest, train_before)


def fake_prices(n=1650):
    rng = np.random.default_rng(102)
    dates = pd.bdate_range("2010-01-04", periods=n)
    market = rng.normal(.0003, .011, n)
    result = {}
    for ticker in ("SPY", "AAPL", "MSFT", "JPM"):
        close = 120 * np.exp(np.cumsum(market + rng.normal(0, .004, n)))
        opening = close * np.exp(rng.normal(0, .002, n))
        result[ticker] = pd.DataFrame({
            "Open": opening,
            "High": np.maximum(opening, close) * 1.009,
            "Low": np.minimum(opening, close) * .991,
            "Close": close,
            "Volume": rng.integers(2_000_000, 9_000_000, n),
        }, index=dates)
    return result


class WeeklyRelativeTests(unittest.TestCase):
    def test_target_is_next_open_to_open_five_sessions_later(self):
        prices = fake_prices()
        panel, labeled, market, dates = build_weekly_panel(prices)
        first_date = dates[70]
        t = prices["AAPL"].index.get_loc(first_date)
        expected = prices["AAPL"]["Open"].iloc[t+6] / prices["AAPL"]["Open"].iloc[t+1] - 1
        spy = prices["SPY"]["Open"].iloc[t+6] / prices["SPY"]["Open"].iloc[t+1] - 1
        row = labeled[(labeled["Date"] == first_date) & (labeled["ticker"] == "AAPL")].iloc[0]
        self.assertAlmostEqual(row["stock_period_return"], expected)
        self.assertAlmostEqual(row["target_excess"], expected - spy)
        self.assertEqual(row["label_end"], prices["SPY"].index[t+6])
        self.assertTrue(all((dates[1:] - dates[:-1]) >= pd.Timedelta(days=5)))

    def test_present_features_ignore_future_open_edit(self):
        prices = fake_prices()
        panel, _, _, dates = build_weekly_panel(prices)
        date = dates[90]
        old = panel[(panel["Date"] == date) & (panel["ticker"] == "AAPL")].iloc[0]
        edited = {k: v.copy() for k, v in prices.items()}
        future_day = edited["AAPL"].index.get_loc(date) + 3
        edited["AAPL"].iloc[future_day, edited["AAPL"].columns.get_loc("Open")] *= 5
        changed, _, _, _ = build_weekly_panel(edited)
        new = changed[(changed["Date"] == date) & (changed["ticker"] == "AAPL")].iloc[0]
        for feature in ["return_1", "return_5", "overnight_gap",
                        "volatility_21", "spy_trend_200"]:
            self.assertAlmostEqual(old[feature], new[feature])

    def test_fold_purges_future_labels(self):
        _, labeled, _, _ = build_weekly_panel(fake_prices())
        folds, holdout = rolling_folds(labeled["Date"], folds=3)
        for start, end in folds:
            train = train_before(labeled, start)
            self.assertTrue((train["label_end"] < start).all())
            self.assertTrue((train["Date"] < start).all())
            valid = labeled[(labeled["Date"] >= start) &
                            (labeled["Date"] < end) &
                            (labeled["label_end"] < holdout)]
            self.assertTrue((valid["label_end"] < holdout).all())

    def test_spy_only_equals_spy_after_same_costs(self):
        dates = pd.bdate_range("2025-01-03", periods=3, freq="5B")
        frame = pd.DataFrame({
            "Date": dates, "ticker": ["AAPL"] * 3,
            "ridge": [.01, .01, .01], "boosting": [.01, -.01, .01],
            "stock_period_return": [.04, -.03, .04],
            "spy_period_return": [.01, -.02, .01],
            "regime": ["bull_normal_vol"] * 3,
        })
        daily, stats = rotation_backtest(frame, spy_only=True)
        np.testing.assert_allclose(daily["net_return"], daily["spy_net"])
        self.assertAlmostEqual(stats["turnover_sides"], 2.0, places=6)
        self.assertAlmostEqual(stats["strategy"]["total_return"],
                               stats["spy"]["total_return"])

    def test_disagreement_blocks_weak_signal(self):
        date = pd.Timestamp("2025-01-03")
        frame = pd.DataFrame([{
            "Date": date, "ticker": "AAPL", "ridge": .02, "boosting": -.02,
            "stock_period_return": .03, "spy_period_return": .01,
            "regime": "bull_normal_vol",
        }])
        daily, metrics = rotation_backtest(frame, threshold=0,
                                           disagreement_penalty=0)
        self.assertEqual(daily["stocks_count"].iloc[0], 0)
        self.assertAlmostEqual(metrics["stock_rotation_fraction"], 0)
        frame["boosting"] = .02
        daily, metrics = rotation_backtest(frame, threshold=0.001,
                                           disagreement_penalty=1.0)
        self.assertEqual(daily["stocks_count"].iloc[0], 1)

    def test_full_pipeline_synthetic(self):
        result = experiment(fake_prices(), n_folds=2)
        self.assertEqual(result["version"], 3)
        self.assertEqual(len(result["walk_forward_folds"]), 2)
        self.assertEqual(len(result["assets"]), 3)
        self.assertIn(result["selection"]["mode"],
                      ["weekly_stock_rotation", "SPY_fallback"])
        self.assertTrue(np.isfinite(result["heldout"]["model_mae_excess"]))
        self.assertIn("regimes", result["heldout"])


if __name__ == "__main__":
    unittest.main()
