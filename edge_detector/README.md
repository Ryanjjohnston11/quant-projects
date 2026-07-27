# Edge Detector

A lie detector for trading strategies. Takes any price series — real or
synthetic — and delivers an honest verdict on whether a strategy has a
real edge or is just luck wearing a costume.

## The idea

Most beginner quant projects build a backtest, watch a strategy "make
money," and conclude they found something. This project does the
opposite: it assumes every good-looking result is a lie until proven
otherwise, and builds the tools to catch the lie.

## How it works

Any strategy is run through three tests:

- **The monkey test** — 5,000 random strategies are run on the same
  price data. If the strategy can't beat most coin-flippers, it has no edge.
- **Out-of-sample testing** — the strategy is run on 1,000 fresh markets
  it was never tuned on. Real edges survive; luck averages to zero.
- **The Sharpe ratio** — return per unit of risk, the industry-standard
  single measure of risk-adjusted performance.

## Key finding

A simple momentum strategy makes a ~$3,000 "profit" on real Bitcoin data
(pulled live via API) — and the monkey test shows it is beaten by ~46% of
random strategies. The profit is real; the edge is imaginary. That gap is
the entire point.

## Run it

```
pip install numpy requests
python3 edge_detector.py
```
