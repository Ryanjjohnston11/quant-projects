import requests
import numpy as np

# =============================================================
#  EDGE DETECTOR
#  Takes any price series, runs a strategy, and delivers an
#  HONEST verdict on whether it has a real edge or is luck.
# =============================================================


def make_coinflip_market(days=350, seed=None):
    """Build a fake market from pure coin flips. Zero edge by construction."""
    if seed is not None:
        np.random.seed(seed)
    flips = np.random.choice([-1, 1], size=days)
    return 100 + np.cumsum(flips)


def get_bitcoin_prices():
    """Pull ~350 days of real daily BTC closing prices from Coinbase."""
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    response = requests.get(url, params={"granularity": 86400})
    data = sorted(response.json())            # oldest first
    return np.array([candle[4] for candle in data])


def momentum_signal(prices, n):
    """Bet UP if price rose over the last n days, else DOWN. Shifted to avoid peeking."""
    momentum = prices[n:] - prices[:-n]       # change over last n days
    return np.sign(momentum[:-1])             # drop last: no future day to trade into


def run_strategy(prices, n):
    """Score a momentum strategy: total profit AND its Sharpe ratio."""
    changes = np.diff(prices)                 # daily dollar moves
    signal = momentum_signal(prices, n)
    daily_pnl = signal * changes[n:]          # profit each day
    total = daily_pnl.sum()
    # Sharpe = average daily profit / how much it bounces around (risk-adjusted return)
    sharpe = daily_pnl.mean() / daily_pnl.std() if daily_pnl.std() > 0 else 0
    return total, sharpe, changes[n:]


def monkey_test(real_pnl, trading_days, num_monkeys=5000):
    """What % of random strategies did as well or better? High % = your result is just luck."""
    results = []
    for _ in range(num_monkeys):
        bets = np.random.choice([-1, 1], size=len(trading_days))
        results.append(np.sum(bets * trading_days))
    results = np.array(results)
    beat_you = np.sum(results >= real_pnl)
    return 100 * beat_you / num_monkeys


def out_of_sample_test(n, days=350, num_markets=1000):
    """Run strategy N on 1000 FRESH coin-flip markets. Real edge profits consistently; luck averages zero."""
    profits = []
    for _ in range(num_markets):
        prices = make_coinflip_market(days=days)
        total, _, _ = run_strategy(prices, n)
        profits.append(total)
    profits = np.array(profits)
    return profits.mean(), np.sum(profits > 0), num_markets


def verdict(prices, n, label):
    """The full battery: run it, monkey-test it, print an honest verdict."""
    total, sharpe, trading_days = run_strategy(prices, n)
    luck_pct = monkey_test(total, trading_days)

    print(f"\n===== VERDICT: {label} (momentum N={n}) =====")
    print(f"  Total profit:        ${total:,.0f}")
    print(f"  Sharpe ratio:        {sharpe:.3f}  (per-day; higher = better risk-adjusted)")
    print(f"  Beaten by luck:      {luck_pct:.1f}% of random monkeys")
    if luck_pct > 30:
        print(f"  >> NO EDGE. This is a coin flip. The profit is a liar.")
    elif luck_pct > 5:
        print(f"  >> WEAK/UNCLEAR. Inside the range of luck. Don't trust it.")
    else:
        print(f"  >> INTERESTING. Rare vs luck -- but test out-of-sample before believing.")


# =============================================================
#  THE FINAL EXAM: same tool, two markets, read the verdicts.
# =============================================================
if __name__ == "__main__":
    N = 3

    # 1. A market we KNOW is pure luck
    fake = make_coinflip_market(days=350, seed=42)
    verdict(fake, N, "COIN-FLIP MARKET")

    # 2. Real Bitcoin
    real = get_bitcoin_prices()
    verdict(real, N, "REAL BITCOIN")

    # 3. The cure: does N=3 survive on markets it has never seen?
    avg, wins, total = out_of_sample_test(N)
    print(f"\n===== OUT-OF-SAMPLE: N={N} on {total} fresh markets =====")
    print(f"  Average profit:  ${avg:,.1f}  (near zero = no real edge)")
    print(f"  Markets won:     {wins}/{total}  (~half = coin flip)")
    print(f"\nProject complete. Trust it over your P&L.\n")