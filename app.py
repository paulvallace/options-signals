"""
Options Signal Web App
Flask app that runs your options pipeline and displays results.
Deploy to Railway for free phone-accessible dashboard.
"""

import os
import json
import logging
import threading
from datetime import datetime, date, timedelta
from pathlib import Path

import yfinance as yf
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from apscheduler.schedulers.background import BackgroundScheduler

# ── CONFIG ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
DATA_DIR.mkdir(exist_ok=True)

SIGNALS_FILE   = DATA_DIR / "signals.json"
TRADES_FILE    = DATA_DIR / "trades.json"
BACKTEST_FILE  = DATA_DIR / "backtest.json"

SMALL_TICKERS = ["SOFI", "ROKU", "RIOT", "MARA", "SNAP"]
BIG_TICKERS   = ["TSLA", "RIOT", "MARA", "AFRM", "UPST", "HOOD", "DKNG", "SNAP"]

SMALL_ACCOUNT  = 2_000
BIG_ACCOUNT    = 20_000
RISK_PCT       = 0.20
MA_WINDOWS     = [20, 50]
HOLD_DAYS      = 10
CALL_COST_PCT  = 0.05
PUT_COST_PCT   = 0.02
STRAD_COST_PCT = 0.08
EARNINGS_MIN   = 3
EARNINGS_MAX   = 7
START_DATE     = "2020-01-01"

# ── TIGHTENED SIGNAL FILTERS ─────────────────────────────────────────────────
MIN_PRICE        = 6.0    # skip sub-$6 tickers (wide spreads, poor liquidity)
MIN_PULLBACK_PCT = 0.03   # pullback from recent high must be at least 3%
MAX_PULLBACK_PCT = 0.12   # but not more than 12% (that's a breakdown, not a dip)
MIN_RSI          = 38     # not in freefall
MAX_RSI          = 68     # not overbought
MIN_VOL_RATIO    = 0.8    # volume at least 80% of 20d avg (avoid dead days)
MA_SLOPE_DAYS    = 5      # MA must be rising over last N days (confirmed uptrend)
MIN_ABOVE_MA_PCT = 0.02   # price must be at least 2% above MA (not barely touching)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

pipeline_lock   = threading.Lock()
backtest_lock   = threading.Lock()
pipeline_status = {"running": False, "last_run": None, "last_result": "Never run"}
backtest_status = {"running": False, "progress": "", "done": False}

# ── PERSISTENCE ───────────────────────────────────────────────────────────────

def load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default

def save_json(path, data):
    path.write_text(json.dumps(data, default=str, indent=2))

# ── DATA HELPERS ──────────────────────────────────────────────────────────────

def _ensure_series(s):
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s.squeeze()

def get_prices(ticker, start=START_DATE):
    df = yf.download(ticker, start=start, auto_adjust=False,
                     group_by="column", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().astype(float)
    return df

def get_current_price(ticker):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None

def get_next_earnings(ticker):
    try:
        tk = yf.Ticker(ticker)
        ed = tk.get_earnings_dates(limit=8)
        if ed is None or len(ed) == 0:
            return None
        idx = ed.index
        today_ts = pd.Timestamp.today().normalize()
        future = idx[idx >= today_ts]
        if len(future) == 0:
            return None
        return future[0].date()
    except Exception:
        return None

def add_ma(df, windows):
    df = df.copy()
    close = _ensure_series(df["Close"]).astype(float)
    for w in windows:
        df[f"MA_{w}"] = close.rolling(w).mean()
    return df

def trading_days_in_range(start_date, end_date):
    days, cur = [], start_date
    while cur <= end_date:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days

def calc_rsi(close, period=14):
    """Wilder RSI on a price series."""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ── SIGNAL GENERATORS ─────────────────────────────────────────────────────────

def call_hedge_signal(ticker, account_size, df=None, as_of_date=None):
    try:
        if df is None:
            df = get_prices(ticker)
        if df is None or len(df) < 60:
            return None
        df = add_ma(df, MA_WINDOWS)

        if as_of_date is not None:
            df_slice = df[df.index <= pd.Timestamp(as_of_date)]
            if len(df_slice) < 60:
                return None
        else:
            df_slice = df

        close  = _ensure_series(df_slice["Close"]).astype(float)
        volume = _ensure_series(df_slice["Volume"]).astype(float)
        price  = float(close.iloc[-1])

        # ── Filter 1: minimum price ──────────────────────────────────────────
        if price < MIN_PRICE:
            return {"ticker": ticker, "strategy": "CALL+HEDGE", "is_buy": False,
                    "price": round(price, 2), "detail": f"Price ${price:.2f} below minimum ${MIN_PRICE}",
                    "account": "small" if account_size == SMALL_ACCOUNT else "big"}

        # ── Filter 2: RSI ─────────────────────────────────────────────────────
        rsi_series = calc_rsi(close)
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
        if not (MIN_RSI <= rsi <= MAX_RSI):
            return {"ticker": ticker, "strategy": "CALL+HEDGE", "is_buy": False,
                    "price": round(price, 2),
                    "detail": f"RSI {rsi:.0f} outside range {MIN_RSI}–{MAX_RSI}",
                    "account": "small" if account_size == SMALL_ACCOUNT else "big"}

        # ── Filter 3: volume confirmation ─────────────────────────────────────
        vol_today  = float(volume.iloc[-1])
        vol_20avg  = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else vol_today
        vol_ratio  = vol_today / vol_20avg if vol_20avg > 0 else 1.0
        if vol_ratio < MIN_VOL_RATIO:
            return {"ticker": ticker, "strategy": "CALL+HEDGE", "is_buy": False,
                    "price": round(price, 2),
                    "detail": f"Volume too low ({vol_ratio:.1f}x avg, need {MIN_VOL_RATIO}x)",
                    "account": "small" if account_size == SMALL_ACCOUNT else "big"}

        for w in MA_WINDOWS:
            ma_col = f"MA_{w}"
            if ma_col not in df_slice.columns:
                continue
            ma_series = _ensure_series(df_slice[ma_col]).astype(float)
            ma_today  = float(ma_series.iloc[-1])
            if pd.isna(ma_today):
                continue

            # ── Filter 4: price meaningfully above MA ─────────────────────────
            above_pct = (price - ma_today) / ma_today
            if above_pct < MIN_ABOVE_MA_PCT:
                continue

            # ── Filter 5: MA slope is rising ──────────────────────────────────
            if len(ma_series.dropna()) < MA_SLOPE_DAYS + 1:
                continue
            ma_prev = float(ma_series.dropna().iloc[-(MA_SLOPE_DAYS + 1)])
            if ma_today <= ma_prev:
                continue

            # ── Filter 6: meaningful pullback from recent high ────────────────
            lookback   = close.iloc[-10:]
            recent_high = float(lookback.max())
            pullback_pct = (recent_high - price) / recent_high

            if pullback_pct < MIN_PULLBACK_PCT:
                continue  # barely any dip — not a real pullback
            if pullback_pct > MAX_PULLBACK_PCT:
                continue  # too much damage — this isn't a dip, it's a breakdown

            # ── Filter 7: previous day was also down (confirming dip) ─────────
            if len(close) < 3:
                continue
            prev_close = float(close.iloc[-2])
            if prev_close >= price * 1.005:
                pass  # prev day higher = currently pulling back, good
            # Allow signal if today is the dip day regardless

            # ── All filters passed ────────────────────────────────────────────
            risk_budget       = account_size * RISK_PCT
            cost_per_contract = CALL_COST_PCT * price * 100
            max_contracts     = int(risk_budget // cost_per_contract) if cost_per_contract > 0 else 0

            return {
                "ticker": ticker, "strategy": "CALL+HEDGE", "is_buy": True,
                "price": round(price, 2), "ma_window": w,
                "hold_days": HOLD_DAYS,
                "call_cost_pct": CALL_COST_PCT, "put_cost_pct": PUT_COST_PCT,
                "max_contracts": max_contracts, "risk_budget": round(risk_budget, 2),
                "rsi": round(rsi, 1), "pullback_pct": round(pullback_pct * 100, 1),
                "vol_ratio": round(vol_ratio, 1),
                "detail": (f"MA{w} uptrend · {pullback_pct*100:.1f}% pullback · "
                           f"RSI {rsi:.0f} · vol {vol_ratio:.1f}x · hold {HOLD_DAYS}d"),
                "account": "small" if account_size == SMALL_ACCOUNT else "big",
            }

        return {"ticker": ticker, "strategy": "CALL+HEDGE", "is_buy": False,
                "price": round(price, 2), "detail": "No signal — filters not met",
                "account": "small" if account_size == SMALL_ACCOUNT else "big"}

    except Exception as e:
        log.warning(f"call_hedge_signal {ticker}: {e}")
        return None

def straddle_signal(ticker, df=None, as_of_date=None, next_earnings=None):
    try:
        if df is None:
            df = get_prices(ticker)
        if df is None or len(df) < 20:
            return None

        if as_of_date is not None:
            close = _ensure_series(df["Close"]).astype(float)
            close = close[close.index <= pd.Timestamp(as_of_date)]
            if len(close) < 20:
                return None
        else:
            close = _ensure_series(df["Close"]).astype(float)

        price = float(close.iloc[-1])

        # Min price filter
        if price < MIN_PRICE:
            return {"ticker": ticker, "strategy": "STRADDLE", "is_buy": False,
                    "price": round(price, 2), "detail": f"Price below ${MIN_PRICE} minimum",
                    "account": "big"}

        # Volatility sweet spot: need SOME vol for a straddle to pay off,
        # but not so much that premium is already priced in
        recent    = close.iloc[-10:]
        pct_range = (recent.max() - recent.min()) / recent.iloc[-1]
        if pct_range < 0.03:
            return {"ticker": ticker, "strategy": "STRADDLE", "is_buy": False,
                    "price": round(price, 2), "detail": "Too quiet — not enough volatility for a straddle",
                    "account": "big"}
        if pct_range > 0.15:
            return {"ticker": ticker, "strategy": "STRADDLE", "is_buy": False,
                    "price": round(price, 2), "detail": "Too volatile — premium already expensive",
                    "account": "big"}

        # RSI filter — avoid entering into extreme moves
        rsi_series = calc_rsi(close)
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
        if rsi < 30 or rsi > 75:
            return {"ticker": ticker, "strategy": "STRADDLE", "is_buy": False,
                    "price": round(price, 2),
                    "detail": f"RSI {rsi:.0f} extreme — wait for calm before earnings",
                    "account": "big"}

        # Earnings filter
        if next_earnings is None:
            next_earnings = get_next_earnings(ticker)
        if next_earnings is None:
            return {"ticker": ticker, "strategy": "STRADDLE", "is_buy": False,
                    "price": round(price, 2), "detail": "No upcoming earnings date",
                    "account": "big"}

        ref_date = as_of_date if as_of_date else date.today()
        delta    = (next_earnings - ref_date).days
        if not (EARNINGS_MIN <= delta <= EARNINGS_MAX):
            return {"ticker": ticker, "strategy": "STRADDLE", "is_buy": False,
                    "price": round(price, 2),
                    "detail": f"Earnings in {delta}d (need {EARNINGS_MIN}–{EARNINGS_MAX}d)",
                    "account": "big"}

        risk_budget       = BIG_ACCOUNT * RISK_PCT
        cost_per_contract = STRAD_COST_PCT * price * 100
        max_contracts     = int(risk_budget // cost_per_contract) if cost_per_contract > 0 else 0

        return {
            "ticker": ticker, "strategy": "STRADDLE", "is_buy": True,
            "price": round(price, 2), "hold_days": 5, "strad_cost_pct": STRAD_COST_PCT,
            "earnings_date": str(next_earnings), "days_to_earnings": delta,
            "max_contracts": max_contracts, "risk_budget": round(risk_budget, 2),
            "rsi": round(rsi, 1), "vol_range_pct": round(pct_range * 100, 1),
            "detail": (f"Earnings in {delta}d · 10d range {pct_range*100:.1f}% · "
                       f"RSI {rsi:.0f} · hold 5d"),
            "account": "big",
        }
    except Exception as e:
        log.warning(f"straddle_signal {ticker}: {e}")
        return None

# ── TRADE SIMULATION ──────────────────────────────────────────────────────────

def simulate_trade_return(trade, price_df):
    try:
        close    = _ensure_series(price_df["Close"]).astype(float)
        entry_ts = pd.Timestamp(trade["entry_date"])
        hold     = trade.get("hold_days", HOLD_DAYS)

        locs = [i for i, d in enumerate(close.index) if d >= entry_ts]
        if not locs:
            return None
        entry_loc = locs[0]
        exit_loc  = min(entry_loc + hold, len(close) - 1)

        ep = float(close.iloc[entry_loc])
        xp = float(close.iloc[exit_loc])

        if trade["strategy"] == "STRADDLE":
            cost   = trade.get("strad_cost_pct", STRAD_COST_PCT) * ep
            payoff = abs(xp - ep)
        else:
            call_cost   = trade.get("call_cost_pct", CALL_COST_PCT) * ep
            call_payoff = max(0.0, xp - ep)
            window      = close.iloc[entry_loc:exit_loc + 1]
            max_gain    = max(0.0, float(window.max()) - ep)
            hedged      = (max_gain / call_cost >= 0.20) if call_cost > 0 else False
            if hedged:
                put_cost   = trade.get("put_cost_pct", PUT_COST_PCT) * ep
                put_payoff = max(0.0, ep - xp)
                cost       = call_cost + put_cost
                payoff     = call_payoff + put_payoff
            else:
                cost, payoff = call_cost, call_payoff

        ret = max((payoff - cost) / cost, -1.0) if cost else 0
        return {
            "exit_date":  str(close.index[exit_loc].date()),
            "exit_price": round(xp, 2),
            "return_pct": round(ret * 100, 2),
        }
    except Exception as e:
        log.warning(f"simulate_trade_return: {e}")
        return None

def trading_days_elapsed(entry_date_str, today=None):
    """Count actual trading days (weekdays) between entry and today."""
    entry = date.fromisoformat(entry_date_str)
    end   = today or date.today()
    count, cur = 0, entry + timedelta(days=1)
    while cur <= end:
        if cur.weekday() < 5:
            count += 1
        cur += timedelta(days=1)
    return count

def close_expired_trades(trades):
    today_str   = str(date.today())
    open_trades, closed = [], []
    for t in trades:
        if t.get("status") != "open":
            closed.append(t); continue
        held = trading_days_elapsed(t["entry_date"])
        if held < t.get("hold_days", HOLD_DAYS):
            open_trades.append(t); continue
        ep         = t["entry_price"]
        exit_price = get_current_price(t["ticker"]) or ep
        if t["strategy"] == "STRADDLE":
            cost   = t.get("strad_cost_pct", STRAD_COST_PCT) * ep
            payoff = abs(exit_price - ep)
        else:
            cost   = t.get("call_cost_pct", CALL_COST_PCT) * ep
            payoff = max(0, exit_price - ep)
        ret = max((payoff - cost) / cost, -1.0) if cost else 0
        t.update({"status":"closed","exit_date":today_str,
                  "exit_price":round(exit_price,2),"return_pct":round(ret*100,2)})
        closed.append(t)
    return open_trades + closed

# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────

def run_backtest_engine(months=2):
    end_date   = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=months * 31)
    days       = trading_days_in_range(start_date, end_date)

    backtest_status["progress"] = "Downloading price data…"
    price_cache = {}
    all_tickers = list(set(SMALL_TICKERS + BIG_TICKERS))
    for i, ticker in enumerate(all_tickers):
        backtest_status["progress"] = f"Downloading {ticker} ({i+1}/{len(all_tickers)})…"
        df = get_prices(ticker, start="2019-01-01")
        if df is not None:
            price_cache[ticker] = add_ma(df, MA_WINDOWS)

    earnings_cache = {t: get_next_earnings(t) for t in BIG_TICKERS}

    trades = []
    for day_i, sim_date in enumerate(days):
        backtest_status["progress"] = f"Simulating {sim_date} ({day_i+1}/{len(days)})…"

        day_buys = []
        seen_today = set()  # deduplicate: one signal per ticker per day

        for ticker in SMALL_TICKERS:
            if ticker not in price_cache: continue
            sig = call_hedge_signal(ticker, SMALL_ACCOUNT,
                                    df=price_cache[ticker], as_of_date=sim_date)
            if sig and sig.get("is_buy") and ticker not in seen_today:
                sig["date"] = str(sim_date)
                day_buys.append(sig)
                seen_today.add(ticker)

        for ticker in BIG_TICKERS:
            if ticker not in price_cache: continue
            strad = straddle_signal(ticker, df=price_cache[ticker],
                                    as_of_date=sim_date,
                                    next_earnings=earnings_cache.get(ticker))
            if strad and strad.get("is_buy"):
                strad["date"] = str(sim_date)
                day_buys.append(strad)  # straddles are always unique (different strategy)

            if ticker not in seen_today:
                call = call_hedge_signal(ticker, BIG_ACCOUNT,
                                         df=price_cache[ticker], as_of_date=sim_date)
                if call and call.get("is_buy"):
                    call["date"] = str(sim_date)
                    day_buys.append(call)
                    seen_today.add(ticker)

        for sig in day_buys:
            ticker = sig["ticker"]
            if ticker not in price_cache: continue
            trade = {
                "ticker": ticker, "strategy": sig["strategy"],
                "entry_date": str(sim_date), "entry_price": sig["price"],
                "hold_days": sig.get("hold_days", HOLD_DAYS),
                "call_cost_pct": sig.get("call_cost_pct", CALL_COST_PCT),
                "put_cost_pct": sig.get("put_cost_pct", PUT_COST_PCT),
                "strad_cost_pct": sig.get("strad_cost_pct", STRAD_COST_PCT),
                "account": sig.get("account", "big"), "backtest": True,
            }
            result = simulate_trade_return(trade, price_cache[ticker])
            if result:
                trade.update(result)
                trade["status"] = "closed"
                trades.append(trade)

    returns  = [t["return_pct"] for t in trades if "return_pct" in t]
    win_rate = round(len([r for r in returns if r>0])/len(returns)*100,1) if returns else 0
    avg_ret  = round(sum(returns)/len(returns),2) if returns else 0

    sorted_trades = sorted(trades, key=lambda t: t.get("exit_date",""))
    cum, pnl_series = 0.0, []
    for t in sorted_trades:
        cum += t.get("return_pct", 0)
        pnl_series.append({"date":t["exit_date"],"cum":round(cum,2),
                            "ticker":t["ticker"],"ret":t.get("return_pct",0)})

    result = {
        "period":       f"{start_date} → {end_date}",
        "trading_days": len(days),
        "total_trades": len(trades),
        "win_rate":     win_rate,
        "avg_return":   avg_ret,
        "best_trade":   round(max(returns),2) if returns else 0,
        "worst_trade":  round(min(returns),2) if returns else 0,
        "total_buys":   len(trades),
        "pnl_series":   pnl_series,
        "trades":       sorted_trades,
        "generated_at": str(datetime.now()),
    }
    save_json(BACKTEST_FILE, result)
    return result

def run_backtest_thread(months=2):
    if not backtest_lock.acquire(blocking=False):
        return
    backtest_status.update({"running":True,"done":False})
    try:
        run_backtest_engine(months=months)
        backtest_status.update({"done":True,"progress":"Complete ✓"})
    except Exception as e:
        log.exception("Backtest error")
        backtest_status["progress"] = f"Error: {e}"
    finally:
        backtest_status["running"] = False
        backtest_lock.release()

# ── LIVE PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline():
    if not pipeline_lock.acquire(blocking=False):
        return {"error": "Already running"}
    pipeline_status["running"] = True
    today_str = str(date.today())
    try:
        new_signals = []
        seen_live   = set()

        for ticker in SMALL_TICKERS:
            sig = call_hedge_signal(ticker, SMALL_ACCOUNT)
            if sig:
                sig["date"] = today_str
                new_signals.append(sig)
                if sig.get("is_buy"):
                    seen_live.add(ticker)

        for ticker in BIG_TICKERS:
            strad = straddle_signal(ticker)
            if strad:
                strad["date"] = today_str
                new_signals.append(strad)
            if ticker not in seen_live:
                call = call_hedge_signal(ticker, BIG_ACCOUNT)
                if call:
                    call["date"] = today_str
                    new_signals.append(call)
                    if call.get("is_buy"):
                        seen_live.add(ticker)

        all_signals = [s for s in load_json(SIGNALS_FILE,[]) if s.get("date")!=today_str]
        all_signals.extend(new_signals)
        save_json(SIGNALS_FILE, all_signals)

        trades   = close_expired_trades(load_json(TRADES_FILE,[]))
        existing = {(t["ticker"],t["strategy"],t["entry_date"]) for t in trades if t.get("status")=="open"}
        for sig in new_signals:
            if not sig.get("is_buy"): continue
            key = (sig["ticker"],sig["strategy"],today_str)
            if key not in existing:
                trades.append({
                    "ticker":sig["ticker"],"strategy":sig["strategy"],
                    "entry_date":today_str,"entry_price":sig["price"],
                    "hold_days":sig.get("hold_days",HOLD_DAYS),
                    "call_cost_pct":sig.get("call_cost_pct",CALL_COST_PCT),
                    "put_cost_pct":sig.get("put_cost_pct",PUT_COST_PCT),
                    "strad_cost_pct":sig.get("strad_cost_pct",STRAD_COST_PCT),
                    "status":"open","account":sig.get("account","big"),
                })
        save_json(TRADES_FILE, trades)

        buy_count = sum(1 for s in new_signals if s.get("is_buy"))
        result    = f"{len(new_signals)} signals · {buy_count} BUY"
        pipeline_status.update({"last_run":today_str,"last_result":result})
        return {"ok":True,"result":result}
    except Exception as e:
        log.exception("Pipeline error")
        pipeline_status["last_result"] = f"Error: {e}"
        return {"error":str(e)}
    finally:
        pipeline_status["running"] = False
        pipeline_lock.release()

# ── SCHEDULER ─────────────────────────────────────────────────────────────────

scheduler = BackgroundScheduler()
scheduler.add_job(run_pipeline,"cron",day_of_week="mon-fri",hour=9,minute=5)
scheduler.start()

# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/run",methods=["POST"])
def api_run():
    if pipeline_status["running"]: return jsonify({"error":"Already running"}),429
    threading.Thread(target=run_pipeline,daemon=True).start()
    return jsonify({"ok":True})

@app.route("/api/status")
def api_status(): return jsonify(pipeline_status)

@app.route("/api/signals")
def api_signals():
    day = request.args.get("date",str(date.today()))
    return jsonify([s for s in load_json(SIGNALS_FILE,[]) if s.get("date")==day])

@app.route("/api/trades")
def api_trades(): return jsonify(load_json(TRADES_FILE,[]))

@app.route("/api/summary")
def api_summary():
    signals = load_json(SIGNALS_FILE,[])
    trades  = load_json(TRADES_FILE,[])
    closed  = [t for t in trades if t.get("status")=="closed"]
    wins    = [t for t in closed if t.get("return_pct",0)>0]
    cum, series = 0.0, []
    for t in sorted(closed,key=lambda x:x.get("exit_date","")):
        cum += t.get("return_pct",0)
        series.append({"date":t["exit_date"],"cum":round(cum,2)})
    return jsonify({
        "total_signals":len(signals),
        "total_buys":len([s for s in signals if s.get("is_buy")]),
        "open_trades":len([t for t in trades if t.get("status")=="open"]),
        "closed_trades":len(closed),
        "win_rate":round(len(wins)/len(closed)*100,1) if closed else 0,
        "avg_return":round(sum(t.get("return_pct",0) for t in closed)/len(closed),2) if closed else 0,
        "pnl_series":series,**pipeline_status,
    })

@app.route("/api/backtest/run",methods=["POST"])
def api_backtest_run():
    if backtest_status["running"]: return jsonify({"error":"Already running"}),429
    months = int(request.json.get("months",2)) if request.is_json else 2
    threading.Thread(target=run_backtest_thread,args=(months,),daemon=True).start()
    return jsonify({"ok":True})

@app.route("/api/backtest/status")
def api_backtest_status(): return jsonify(backtest_status)

@app.route("/api/backtest/results")
def api_backtest_results():
    data = load_json(BACKTEST_FILE,None)
    if data is None: return jsonify({"error":"No results yet"}),404
    return jsonify(data)

@app.route("/api/ticker-stats")
def api_ticker_stats():
    """Per-ticker win rate, avg return, trade count from all closed trades (live + backtest)."""
    live_trades = load_json(TRADES_FILE, [])
    bt_data     = load_json(BACKTEST_FILE, None)
    bt_trades   = bt_data.get("trades", []) if bt_data else []

    # Merge: prefer backtest for historical depth, live for recency
    all_closed = [t for t in live_trades if t.get("status") == "closed"] + \
                 [t for t in bt_trades   if t.get("status") == "closed"]

    stats = {}
    for t in all_closed:
        key = t["ticker"]
        if key not in stats:
            stats[key] = {"ticker": key, "trades": [], "wins": 0}
        ret = t.get("return_pct", 0)
        stats[key]["trades"].append(ret)
        if ret > 0:
            stats[key]["wins"] += 1

    result = []
    for k, v in stats.items():
        n       = len(v["trades"])
        wr      = round(v["wins"] / n * 100, 1) if n else 0
        avg     = round(sum(v["trades"]) / n, 1) if n else 0
        best    = round(max(v["trades"]), 1) if n else 0
        worst   = round(min(v["trades"]), 1) if n else 0
        verdict = "trade"   if wr >= 45 else \
                  "caution" if wr >= 30 else "skip"
        result.append({
            "ticker":   k,
            "trades":   n,
            "win_rate": wr,
            "avg_ret":  avg,
            "best":     best,
            "worst":    worst,
            "verdict":  verdict,
        })

    result.sort(key=lambda x: x["win_rate"], reverse=True)
    return jsonify(result)

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Options Signals</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0a0b;--surface:#111113;--surface2:#1a1a1e;--border:#2a2a30;--text:#e8e8ec;--muted:#6b6b78;--green:#22c55e;--green-dim:#14532d;--red:#ef4444;--red-dim:#450a0a;--amber:#f59e0b;--blue:#3b82f6;--blue-dim:#1e3a5f;--mono:'DM Mono',monospace;--sans:'Syne',sans-serif;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html{background:var(--bg);color:var(--text);font-family:var(--sans);}
body{min-height:100vh;padding-bottom:calc(env(safe-area-inset-bottom) + 16px);}
.header{padding:20px 20px 0;display:flex;align-items:flex-start;justify-content:space-between;gap:12px;}
.header h1{font-size:22px;font-weight:700;letter-spacing:-.02em;}
.header p{font-size:12px;color:var(--muted);font-family:var(--mono);margin-top:3px;}
.sdot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;}
.dok{background:var(--green);box-shadow:0 0 6px var(--green);}
.drun{background:var(--amber);animation:pulse .8s infinite;}
.didle{background:var(--muted);}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.rbtn{background:var(--green);color:#000;border:none;border-radius:10px;padding:10px 20px;font-family:var(--sans);font-size:14px;font-weight:600;cursor:pointer;white-space:nowrap;}
.rbtn:disabled{opacity:.4;cursor:default;}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:16px 20px;}
.m{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px;}
.ml{font-size:10px;color:var(--muted);font-family:var(--mono);letter-spacing:.05em;text-transform:uppercase;}
.mv{font-size:24px;font-weight:700;margin-top:2px;}
.ms{font-size:10px;color:var(--muted);font-family:var(--mono);margin-top:1px;}
.pos{color:var(--green);}.neg{color:var(--red);}
.tabs{display:flex;gap:4px;padding:0 20px 12px;overflow-x:auto;-webkit-overflow-scrolling:touch;}
.tabs::-webkit-scrollbar{display:none;}
.tab{padding:7px 16px;border-radius:20px;border:1px solid var(--border);background:transparent;color:var(--muted);font-family:var(--sans);font-size:13px;cursor:pointer;white-space:nowrap;}
.tab.active{background:var(--surface2);border-color:#444;color:var(--text);font-weight:600;}
.cards{padding:0 20px;display:flex;flex-direction:column;gap:8px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:14px 16px;display:flex;gap:12px;align-items:flex-start;}
.card.bc{border-color:#1a4a2a;background:#0d1f13;}
.cdot{width:8px;height:8px;border-radius:50%;margin-top:6px;flex-shrink:0;}
.bdot{background:var(--green);box-shadow:0 0 6px var(--green);}.ndot{background:var(--border);}
.cbody{flex:1;min-width:0;}
.ctop{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.cticker{font-size:16px;font-weight:700;font-family:var(--mono);}
.tag{font-size:10px;font-family:var(--mono);font-weight:500;padding:2px 8px;border-radius:4px;}
.ts{background:var(--blue-dim);color:#93c5fd;}.tc{background:var(--green-dim);color:#86efac;}
.tbuy{background:var(--green);color:#000;}.tsm{background:var(--surface2);color:var(--muted);}
.tbig{background:#1a1a3a;color:#818cf8;}
.cdet{font-size:12px;color:var(--muted);font-family:var(--mono);margin-top:5px;}
.cprice{font-size:12px;color:var(--text);font-family:var(--mono);margin-top:2px;}
.trow{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 16px;display:flex;justify-content:space-between;align-items:center;gap:8px;}
.tleft{flex:1;min-width:0;}
.tticker{font-size:15px;font-weight:700;font-family:var(--mono);}
.tmeta{font-size:11px;color:var(--muted);font-family:var(--mono);margin-top:3px;}
.tret{font-size:16px;font-weight:700;font-family:var(--mono);text-align:right;white-space:nowrap;}
.bwrap{margin-top:4px;height:3px;background:var(--border);border-radius:2px;}
.bfill{height:3px;border-radius:2px;}
.cwrap{padding:0 20px;}
.cbox{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px;}
.carea{position:relative;width:100%;height:220px;}
.empty{text-align:center;padding:40px 20px;color:var(--muted);font-family:var(--mono);font-size:13px;}
.drow{padding:0 20px 10px;display:flex;align-items:center;gap:10px;}
.drow input[type=date]{background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:var(--mono);font-size:13px;padding:6px 10px;}
.btctrl{padding:0 20px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.btsel{background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:var(--mono);font-size:13px;padding:6px 10px;}
.btbtn{background:var(--blue);color:#fff;border:none;border-radius:10px;padding:9px 20px;font-family:var(--sans);font-size:14px;font-weight:600;cursor:pointer;}
.btbtn:disabled{opacity:.4;cursor:default;}
.btprog{font-size:12px;color:var(--amber);font-family:var(--mono);padding:0 20px 10px;}
.btmet{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;padding:0 20px 16px;}
.btm{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px;}
.btml{font-size:10px;color:var(--muted);font-family:var(--mono);letter-spacing:.05em;text-transform:uppercase;}
.btmv{font-size:20px;font-weight:700;margin-top:2px;}
.btlist{padding:0 20px;display:flex;flex-direction:column;gap:6px;max-height:420px;overflow-y:auto;}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Options Signals</h1>
    <p><span class="sdot didle" id="sdot"></span><span id="stxt">Loading…</span></p>
  </div>
  <button class="rbtn" id="rbtn" onclick="runPipeline()">Run ↗</button>
</div>
<div style="height:16px"></div>
<div class="metrics">
  <div class="m"><div class="ml">Buys</div><div class="mv" id="m-buys">—</div><div class="ms">all time</div></div>
  <div class="m"><div class="ml">Win rate</div><div class="mv" id="m-wr">—</div><div class="ms">closed</div></div>
  <div class="m"><div class="ml">Avg ret</div><div class="mv" id="m-ret">—</div><div class="ms">sim P&L</div></div>
</div>
<div class="tabs">
  <button class="tab active" onclick="showTab('today',this)">Today</button>
  <button class="tab" onclick="showTab('open',this)">Open trades</button>
  <button class="tab" onclick="showTab('closed',this)">History</button>
  <button class="tab" onclick="showTab('chart',this)">P&L chart</button>
  <button class="tab" onclick="showTab('backtest',this)">Backtest ↗</button>
  <button class="tab" onclick="showTab('stats',this)">Ticker stats</button>
</div>

<div id="tab-today">
  <div class="drow"><span style="font-size:12px;color:var(--muted);font-family:var(--mono)">Date:</span>
    <input type="date" id="dpick" onchange="loadSignals(this.value)"></div>
  <div class="cards" id="siglist"><div class="empty">Hit "Run ↗" to generate signals.</div></div>
</div>

<div id="tab-open" style="display:none">
  <div class="cards" id="openlist"><div class="empty">No open trades.</div></div>
</div>

<div id="tab-closed" style="display:none">
  <div class="cards" id="closedlist"><div class="empty">No closed trades yet.</div></div>
</div>

<div id="tab-chart" style="display:none">
  <div class="cwrap"><div class="cbox">
    <div style="font-size:12px;color:var(--muted);font-family:var(--mono);margin-bottom:12px">Cumulative sim P&L %</div>
    <div class="carea"><canvas id="pnlChart" role="img" aria-label="Cumulative P&L">No data.</canvas></div>
  </div></div>
</div>

<div id="tab-backtest" style="display:none">
  <div class="btctrl">
    <span style="font-size:13px;color:var(--muted);font-family:var(--mono)">Period:</span>
    <select class="btsel" id="btmonths">
      <option value="1">Last 1 month</option>
      <option value="2" selected>Last 2 months</option>
      <option value="3">Last 3 months</option>
      <option value="6">Last 6 months</option>
    </select>
    <button class="btbtn" id="btbtn" onclick="runBacktest()">Run backtest ↗</button>
  </div>
  <div class="btprog" id="btprog" style="display:none"></div>
  <div class="btmet" id="btmet" style="display:none">
    <div class="btm"><div class="btml">Total trades</div><div class="btmv" id="bt-tot">—</div></div>
    <div class="btm"><div class="btml">Win rate</div><div class="btmv" id="bt-wr">—</div></div>
    <div class="btm"><div class="btml">Avg return</div><div class="btmv" id="bt-avg">—</div></div>
    <div class="btm"><div class="btml">Best / worst</div><div class="btmv" id="bt-bw" style="font-size:14px">—</div></div>
  </div>
  <div class="cwrap" id="btcwrap" style="display:none">
    <div class="cbox">
      <div style="font-size:12px;color:var(--muted);font-family:var(--mono);margin-bottom:12px">Backtest cumulative P&L %</div>
      <div class="carea"><canvas id="btChart" role="img" aria-label="Backtest P&L">No data.</canvas></div>
    </div>
  </div>
  <div style="height:12px"></div>
  <div style="padding:0 20px;font-size:11px;color:var(--muted);font-family:var(--mono);margin-bottom:8px" id="btperiod"></div>
  <div class="btlist" id="btlist"></div>
  <div class="empty" id="btempty">Select a period and hit Run backtest.</div>
</div>

<div style="height:16px"></div>
<div id="tab-stats" style="display:none">
  <div style="padding:0 20px 8px;font-size:12px;color:var(--muted);font-family:var(--mono)" id="stats-source"></div>
  <div style="padding:0 20px;display:flex;flex-direction:column;gap:6px" id="stats-list">
    <div class="empty">Run the backtest first to see per-ticker breakdown.</div>
  </div>
  <div style="height:10px"></div>
  <div style="padding:0 20px;font-size:11px;color:var(--muted);font-family:var(--mono);line-height:1.7">
    green = trade it (45%+ win rate) &nbsp;·&nbsp; amber = caution &nbsp;·&nbsp; red = skip it
  </div>
  <div style="height:16px"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
let liveChart=null, btChart=null;
const today=new Date().toISOString().slice(0,10);
document.getElementById('dpick').value=today;

function showTab(n,btn){
  ['today','open','closed','chart','backtest','stats'].forEach(t=>
    document.getElementById('tab-'+t).style.display=t===n?'':'none');
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  if(n==='chart') drawLiveChart();
  if(n==='stats') loadTickerStats();
}

async function runPipeline(){
  const btn=document.getElementById('rbtn');
  btn.disabled=true; btn.textContent='…';
  setStat('run');
  await fetch('/api/run',{method:'POST'});
  const iv=setInterval(async()=>{
    const s=await(await fetch('/api/status')).json();
    if(!s.running){clearInterval(iv);btn.disabled=false;btn.textContent='Run ↗';setStat('ok');loadAll();}
  },2000);
}

function setStat(st){
  const d=document.getElementById('sdot'),t=document.getElementById('stxt');
  d.className='sdot '+(st==='ok'?'dok':st==='run'?'drun':'didle');
  if(st==='run') t.textContent='Running pipeline…';
}

async function loadStatus(){
  const s=await(await fetch('/api/summary')).json();
  document.getElementById('m-buys').textContent=s.total_buys??'0';
  document.getElementById('m-wr').textContent=s.win_rate?s.win_rate+'%':'—';
  const r=s.avg_return,el=document.getElementById('m-ret');
  el.textContent=r!=null?(r>=0?'+':'')+r+'%':'—';
  el.className='mv '+(r>0?'pos':r<0?'neg':'');
  const d=document.getElementById('sdot'),t=document.getElementById('stxt');
  if(s.running){d.className='sdot drun';t.textContent='Running…';}
  else if(s.last_run){d.className='sdot dok';t.textContent='Last run '+s.last_run+' · '+(s.last_result||'');}
  else{d.className='sdot didle';t.textContent='Never run';}
}

async function loadSignals(ds){
  const sigs=await(await fetch('/api/signals?date='+ds)).json();
  const el=document.getElementById('siglist');
  if(!sigs.length){el.innerHTML='<div class="empty">No signals for '+ds+'. Hit Run ↗.</div>';return;}
  el.innerHTML=sigs.map(s=>`
    <div class="card ${s.is_buy?'bc':''}">
      <div class="cdot ${s.is_buy?'bdot':'ndot'}"></div>
      <div class="cbody">
        <div class="ctop">
          <span class="cticker">${s.ticker}</span>
          <span class="tag ${s.strategy==='STRADDLE'?'ts':'tc'}">${s.strategy}</span>
          <span class="tag ${s.account==='small'?'tsm':'tbig'}">${s.account}</span>
          ${s.is_buy?'<span class="tag tbuy">BUY</span>':''}
        </div>
        <div class="cdet">${s.detail||''}</div>
        ${s.price?`<div class="cprice">$${parseFloat(s.price).toFixed(2)}${s.max_contracts!=null?' · max '+s.max_contracts+' contract'+(s.max_contracts!==1?'s':''):''}</div>`:''}
      </div>
    </div>`).join('');
}

function tradingDaysElapsed(entryDateStr) {
  const entry = new Date(entryDateStr);
  const today = new Date();
  let count = 0, cur = new Date(entry);
  cur.setDate(cur.getDate() + 1);
  while (cur <= today) {
    const d = cur.getDay();
    if (d !== 0 && d !== 6) count++;
    cur.setDate(cur.getDate() + 1);
  }
  return count;
}

async function loadTrades(){
  const trades=await(await fetch('/api/trades')).json();
  const open=trades.filter(t=>t.status==='open');
  const closed=trades.filter(t=>t.status==='closed').reverse();
  const td=new Date().toISOString().slice(0,10);
  document.getElementById('openlist').innerHTML=!open.length?'<div class="empty">No open trades.</div>':
    open.map(t=>{
      const held = tradingDaysElapsed(t.entry_date);
      const pct  = Math.min(held/t.hold_days*100,100);
      return `<div class="card">
        <div class="cdot" style="background:var(--amber);margin-top:6px"></div>
        <div class="cbody">
          <div class="ctop"><span class="cticker">${t.ticker}</span>
            <span class="tag ${t.strategy==='STRADDLE'?'ts':'tc'}">${t.strategy}</span></div>
          <div class="cdet">Entry ${t.entry_date} · $${parseFloat(t.entry_price).toFixed(2)} · day ${held}/${t.hold_days}</div>
          <div class="bwrap"><div class="bfill" style="width:${pct}%;background:var(--amber)"></div></div>
        </div></div>`;
    }).join('');
  document.getElementById('closedlist').innerHTML=!closed.length?'<div class="empty">No closed trades yet.</div>':
    closed.map(t=>{
      const ret=t.return_pct??0,col=ret>0?'var(--green)':ret<0?'var(--red)':'var(--muted)';
      return `<div class="trow">
        <div class="tleft">
          <div class="tticker">${t.ticker} <span class="tag ${t.strategy==='STRADDLE'?'ts':'tc'}" style="font-size:9px">${t.strategy}</span></div>
          <div class="tmeta">${t.entry_date} → ${t.exit_date||'?'} · $${parseFloat(t.entry_price).toFixed(2)} → $${parseFloat(t.exit_price||t.entry_price).toFixed(2)}</div>
        </div>
        <div class="tret" style="color:${col}">${ret>=0?'+':''}${ret.toFixed(1)}%</div>
      </div>`;
    }).join('');
}

async function drawLiveChart(){
  const s=await(await fetch('/api/summary')).json();
  const series=s.pnl_series||[];
  if(!series.length||!window.Chart) return;
  const ctx=document.getElementById('pnlChart').getContext('2d');
  if(liveChart) liveChart.destroy();
  liveChart=new Chart(ctx,{type:'line',
    data:{labels:series.map(p=>p.date),datasets:[{data:series.map(p=>p.cum),borderColor:'#22c55e',backgroundColor:'rgba(34,197,94,0.06)',fill:true,tension:0.4,pointRadius:3,borderWidth:2}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#6b6b78',font:{family:'DM Mono',size:10}},grid:{color:'#1a1a1e'}},
              y:{ticks:{color:'#6b6b78',font:{family:'DM Mono',size:10},callback:v=>v+'%'},grid:{color:'#1a1a1e'}}}}});
}

async function runBacktest(){
  const btn=document.getElementById('btbtn');
  const months=parseInt(document.getElementById('btmonths').value);
  btn.disabled=true; btn.textContent='Running…';
  document.getElementById('btprog').style.display='';
  document.getElementById('btprog').textContent='Starting…';
  document.getElementById('btempty').style.display='none';
  document.getElementById('btmet').style.display='none';
  document.getElementById('btcwrap').style.display='none';
  document.getElementById('btlist').innerHTML='';

  await fetch('/api/backtest/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({months})});
  const iv=setInterval(async()=>{
    const st=await(await fetch('/api/backtest/status')).json();
    document.getElementById('btprog').textContent=st.progress||'Working…';
    if(!st.running){
      clearInterval(iv);
      btn.disabled=false; btn.textContent='Run backtest ↗';
      if(st.done) loadBTResults();
    }
  },2500);
}

async function loadBTResults(){
  const r=await fetch('/api/backtest/results');
  if(!r.ok) return;
  const d=await r.json();
  document.getElementById('btprog').style.display='none';
  document.getElementById('btmet').style.display='';
  document.getElementById('btcwrap').style.display='';
  document.getElementById('btempty').style.display='none';
  document.getElementById('btperiod').textContent=d.period+' · '+d.trading_days+' trading days · '+d.total_trades+' trades';
  document.getElementById('bt-tot').textContent=d.total_trades;
  const wrEl=document.getElementById('bt-wr');
  wrEl.textContent=d.win_rate+'%';
  wrEl.className='btmv '+(d.win_rate>=50?'pos':d.win_rate<40?'neg':'');
  const aEl=document.getElementById('bt-avg');
  aEl.textContent=(d.avg_return>=0?'+':'')+d.avg_return+'%';
  aEl.className='btmv '+(d.avg_return>0?'pos':d.avg_return<0?'neg':'');
  document.getElementById('bt-bw').innerHTML=`<span class="pos">+${d.best_trade}%</span> / <span class="neg">${d.worst_trade}%</span>`;

  if(d.pnl_series&&d.pnl_series.length){
    const ctx=document.getElementById('btChart').getContext('2d');
    if(btChart) btChart.destroy();
    const col=d.avg_return>=0?'#22c55e':'#ef4444';
    btChart=new Chart(ctx,{type:'line',
      data:{labels:d.pnl_series.map(p=>p.date),datasets:[{data:d.pnl_series.map(p=>p.cum),borderColor:col,backgroundColor:col+'18',fill:true,tension:0.4,pointRadius:2,borderWidth:2}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
        scales:{x:{ticks:{color:'#6b6b78',font:{family:'DM Mono',size:10},maxTicksLimit:8},grid:{color:'#1a1a1e'}},
                y:{ticks:{color:'#6b6b78',font:{family:'DM Mono',size:10},callback:v=>v+'%'},grid:{color:'#1a1a1e'}}}}});
  }

  const trades=[...(d.trades||[])].reverse();
  document.getElementById('btlist').innerHTML=trades.map(t=>{
    const ret=t.return_pct??0,col=ret>0?'var(--green)':ret<0?'var(--red)':'var(--muted)';
    return `<div class="trow">
      <div class="tleft">
        <div class="tticker">${t.ticker}
          <span class="tag ${t.strategy==='STRADDLE'?'ts':'tc'}" style="font-size:9px">${t.strategy}</span>
          <span class="tag ${t.account==='small'?'tsm':'tbig'}" style="font-size:9px">${t.account}</span>
        </div>
        <div class="tmeta">${t.entry_date} → ${t.exit_date||'?'} · $${parseFloat(t.entry_price).toFixed(2)} → $${parseFloat(t.exit_price||t.entry_price).toFixed(2)}</div>
      </div>
      <div class="tret" style="color:${col}">${ret>=0?'+':''}${ret.toFixed(1)}%</div>
    </div>`;
  }).join('');
}

async function loadTickerStats(){
  const r=await fetch('/api/ticker-stats');
  if(!r.ok) return;
  const stats=await r.json();
  if(!stats.length){
    document.getElementById('stats-list').innerHTML='<div class="empty">Run the backtest first to see per-ticker breakdown.</div>';
    return;
  }
  const total=stats.reduce((a,s)=>a+s.trades,0);
  document.getElementById('stats-source').textContent=total+' closed trades analysed · sorted by win rate';
  document.getElementById('stats-list').innerHTML=stats.map(s=>{
    const col=s.verdict==='trade'?'#22c55e':s.verdict==='caution'?'#f59e0b':'#ef4444';
    const bgTag=s.verdict==='trade'?'background:#14532d;color:#86efac':
                s.verdict==='caution'?'background:#451a03;color:#fcd34d':
                'background:#450a0a;color:#fca5a5';
    const barW=Math.min(Math.max(s.win_rate,0),100);
    const avgCol=s.avg_ret>0?'#22c55e':s.avg_ret<0?'#ef4444':'var(--muted)';
    return `<div class="trow">
      <div class="cdot" style="width:8px;height:8px;border-radius:50%;background:${col};flex-shrink:0;margin-top:2px"></div>
      <div style="width:46px;flex-shrink:0;font-size:14px;font-weight:700;font-family:var(--mono);color:var(--text)">${s.ticker}</div>
      <div style="flex:1;min-width:0">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
          <div style="flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden">
            <div style="width:${barW}%;height:5px;background:${col};border-radius:3px"></div>
          </div>
          <span style="font-size:12px;font-family:var(--mono);color:${col};font-weight:500;min-width:36px;text-align:right">${s.win_rate}%</span>
        </div>
        <div style="font-size:11px;font-family:var(--mono);color:var(--muted)">
          avg <span style="color:${avgCol}">${s.avg_ret>=0?'+':''}${s.avg_ret}%</span>
          &nbsp;·&nbsp; best <span style="color:#22c55e">+${s.best}%</span>
          &nbsp;·&nbsp; worst <span style="color:#ef4444">${s.worst}%</span>
          &nbsp;·&nbsp; ${s.trades} trades
        </div>
      </div>
      <div style="font-size:10px;font-family:var(--mono);font-weight:500;padding:2px 8px;border-radius:4px;flex-shrink:0;${bgTag}">${s.verdict}</div>
    </div>`;
  }).join('');
}

async function loadAll(){
  await loadStatus();
  await loadSignals(document.getElementById('dpick').value);
  await loadTrades();
  try{const r=await fetch('/api/backtest/results');if(r.ok) loadBTResults();}catch(e){}
  loadTickerStats();
}

function showTab(n,btn){
  ['today','open','closed','chart','backtest','stats'].forEach(t=>
    document.getElementById('tab-'+t).style.display=t===n?'':'none');
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  if(n==='chart') drawLiveChart();
  if(n==='stats') loadTickerStats();
}

loadAll();
setInterval(loadStatus,15000);
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=False)
