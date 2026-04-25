"""
Options Signal Web App
Flask app that runs your options pipeline and displays results.
Deploy to Railway for free phone-accessible dashboard.
"""

import os
import sys
import json
import csv
import logging
import subprocess
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

SIGNALS_FILE  = DATA_DIR / "signals.json"
TRADES_FILE   = DATA_DIR / "trades.json"
PIPELINE_LOG  = DATA_DIR / "pipeline.log"

# Tickers
SMALL_TICKERS = ["PLTR", "SOFI", "CHWY", "NIO", "ROKU"]
BIG_TICKERS   = ["TSLA", "RIOT", "MARA", "AFRM", "UPST", "HOOD", "PLTR", "DKNG", "SNAP", "NIO"]

# Strategy params
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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

pipeline_lock  = threading.Lock()
pipeline_status = {"running": False, "last_run": None, "last_result": "Never run"}

# ── PERSISTENCE ─────────────────────────────────────────────────────────────

def load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default

def save_json(path, data):
    path.write_text(json.dumps(data, default=str, indent=2))

# ── DATA HELPERS ─────────────────────────────────────────────────────────────

def _ensure_series(s):
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s.squeeze()

def get_prices(ticker):
    df = yf.download(ticker, start=START_DATE, auto_adjust=False,
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

# ── SIGNAL GENERATORS ────────────────────────────────────────────────────────

def call_hedge_signal(ticker, account_size):
    try:
        df = get_prices(ticker)
        if df is None or len(df) < 60:
            return None
        df = add_ma(df, MA_WINDOWS)
        close = _ensure_series(df["Close"]).astype(float)

        price = float(close.iloc[-1])
        signals = []

        for w in MA_WINDOWS:
            ma_col = f"MA_{w}"
            if ma_col not in df.columns:
                continue
            ma_today = float(df[ma_col].iloc[-1])
            if pd.isna(ma_today):
                continue

            # Uptrend + recent pullback
            if price <= ma_today:
                continue
            recent = close.iloc[-5:]
            if not (recent.max() > price and recent.iloc[-2] > price):
                continue

            risk_budget = account_size * RISK_PCT
            cost_per_contract = CALL_COST_PCT * price * 100
            max_contracts = int(risk_budget // cost_per_contract) if cost_per_contract > 0 else 0

            signals.append({
                "ticker": ticker,
                "strategy": "CALL+HEDGE",
                "is_buy": True,
                "price": round(price, 2),
                "ma_window": w,
                "hold_days": HOLD_DAYS,
                "call_cost_pct": CALL_COST_PCT,
                "put_cost_pct": PUT_COST_PCT,
                "max_contracts": max_contracts,
                "risk_budget": round(risk_budget, 2),
                "detail": f"Above MA{w} with pullback · hold {HOLD_DAYS}d · hedge at +20%",
                "account": "small" if account_size == SMALL_ACCOUNT else "big",
            })

        if not signals:
            return {
                "ticker": ticker,
                "strategy": "CALL+HEDGE",
                "is_buy": False,
                "price": round(price, 2),
                "detail": "No signal — conditions not met",
                "account": "small" if account_size == SMALL_ACCOUNT else "big",
            }

        return signals[0]  # best (first MA window that triggered)

    except Exception as e:
        log.warning(f"call_hedge_signal {ticker}: {e}")
        return None

def straddle_signal(ticker):
    try:
        df = get_prices(ticker)
        if df is None or len(df) < 20:
            return None
        close = _ensure_series(df["Close"]).astype(float)
        price = float(close.iloc[-1])

        # Volatility filter
        recent = close.iloc[-10:]
        pct_range = (recent.max() - recent.min()) / recent.iloc[-1]
        if pct_range > 0.08:
            return {
                "ticker": ticker, "strategy": "STRADDLE", "is_buy": False,
                "price": round(price, 2),
                "detail": "Too volatile recently",
                "account": "big",
            }

        # Earnings filter
        next_earnings = get_next_earnings(ticker)
        if next_earnings is None:
            return {
                "ticker": ticker, "strategy": "STRADDLE", "is_buy": False,
                "price": round(price, 2),
                "detail": "No upcoming earnings date found",
                "account": "big",
            }

        delta = (next_earnings - date.today()).days
        if not (EARNINGS_MIN <= delta <= EARNINGS_MAX):
            return {
                "ticker": ticker, "strategy": "STRADDLE", "is_buy": False,
                "price": round(price, 2),
                "detail": f"Earnings in {delta}d (need {EARNINGS_MIN}–{EARNINGS_MAX}d)",
                "account": "big",
            }

        risk_budget = BIG_ACCOUNT * RISK_PCT
        cost_per_contract = STRAD_COST_PCT * price * 100
        max_contracts = int(risk_budget // cost_per_contract) if cost_per_contract > 0 else 0

        return {
            "ticker": ticker,
            "strategy": "STRADDLE",
            "is_buy": True,
            "price": round(price, 2),
            "hold_days": 5,
            "strad_cost_pct": STRAD_COST_PCT,
            "earnings_date": str(next_earnings),
            "days_to_earnings": delta,
            "max_contracts": max_contracts,
            "risk_budget": round(risk_budget, 2),
            "detail": f"Earnings in {delta}d · hold 5d · {int(STRAD_COST_PCT*100)}% cost",
            "account": "big",
        }

    except Exception as e:
        log.warning(f"straddle_signal {ticker}: {e}")
        return None

# ── TRADE LIFECYCLE ───────────────────────────────────────────────────────────

def close_expired_trades(trades):
    today_str = str(date.today())
    open_trades, closed = [], []

    for t in trades:
        if t.get("status") != "open":
            closed.append(t)
            continue

        entry = date.fromisoformat(t["entry_date"])
        hold  = t.get("hold_days", HOLD_DAYS)
        if (date.today() - entry).days < hold:
            open_trades.append(t)
            continue

        # Fetch exit price
        exit_price = get_current_price(t["ticker"]) or t["entry_price"]
        ep = t["entry_price"]

        if t["strategy"] == "STRADDLE":
            cost    = t.get("strad_cost_pct", STRAD_COST_PCT) * ep
            payoff  = abs(exit_price - ep)
            ret     = (payoff - cost) / cost if cost else 0
        else:
            cost    = t.get("call_cost_pct", CALL_COST_PCT) * ep
            payoff  = max(0, exit_price - ep)
            ret     = (payoff - cost) / cost if cost else 0

        t.update({
            "status": "closed",
            "exit_date": today_str,
            "exit_price": round(exit_price, 2),
            "return_pct": round(ret * 100, 2),
        })
        closed.append(t)

    return open_trades + closed

# ── PIPELINE ──────────────────────────────────────────────────────────────────

def run_pipeline():
    if not pipeline_lock.acquire(blocking=False):
        return {"error": "Pipeline already running"}

    pipeline_status["running"] = True
    today_str = str(date.today())

    try:
        log.info(f"Pipeline starting — {today_str}")
        new_signals = []

        # Small account — call+hedge only
        for ticker in SMALL_TICKERS:
            sig = call_hedge_signal(ticker, SMALL_ACCOUNT)
            if sig:
                sig["date"] = today_str
                new_signals.append(sig)

        # Big account — straddle + call+hedge
        for ticker in BIG_TICKERS:
            strad = straddle_signal(ticker)
            if strad:
                strad["date"] = today_str
                new_signals.append(strad)
            call = call_hedge_signal(ticker, BIG_ACCOUNT)
            if call:
                call["date"] = today_str
                new_signals.append(call)

        # Load + update state
        all_signals = load_json(SIGNALS_FILE, [])
        # Remove today's previous signals if re-running
        all_signals = [s for s in all_signals if s.get("date") != today_str]
        all_signals.extend(new_signals)
        save_json(SIGNALS_FILE, all_signals)

        # Trades
        trades = load_json(TRADES_FILE, [])
        trades = close_expired_trades(trades)

        # Open new trades for today's BUY signals
        existing_keys = {(t["ticker"], t["strategy"], t["entry_date"]) for t in trades if t.get("status") == "open"}
        for sig in new_signals:
            if not sig.get("is_buy"):
                continue
            key = (sig["ticker"], sig["strategy"], today_str)
            if key not in existing_keys:
                trades.append({
                    "ticker": sig["ticker"],
                    "strategy": sig["strategy"],
                    "entry_date": today_str,
                    "entry_price": sig["price"],
                    "hold_days": sig.get("hold_days", HOLD_DAYS),
                    "call_cost_pct": sig.get("call_cost_pct", CALL_COST_PCT),
                    "put_cost_pct": sig.get("put_cost_pct", PUT_COST_PCT),
                    "strad_cost_pct": sig.get("strad_cost_pct", STRAD_COST_PCT),
                    "status": "open",
                    "account": sig.get("account", "big"),
                })

        save_json(TRADES_FILE, trades)

        buy_count = sum(1 for s in new_signals if s.get("is_buy"))
        result = f"{len(new_signals)} signals · {buy_count} BUY"
        pipeline_status.update({"last_run": today_str, "last_result": result})
        log.info(f"Pipeline done — {result}")
        return {"ok": True, "signals": new_signals, "result": result}

    except Exception as e:
        log.exception("Pipeline error")
        pipeline_status["last_result"] = f"Error: {e}"
        return {"error": str(e)}
    finally:
        pipeline_status["running"] = False
        pipeline_lock.release()

# ── SCHEDULER ────────────────────────────────────────────────────────────────

scheduler = BackgroundScheduler()
scheduler.add_job(run_pipeline, "cron", day_of_week="mon-fri", hour=9, minute=5)
scheduler.start()

# ── API ROUTES ────────────────────────────────────────────────────────────────

@app.route("/api/run", methods=["POST"])
def api_run():
    if pipeline_status["running"]:
        return jsonify({"error": "Already running"}), 429
    t = threading.Thread(target=run_pipeline)
    t.daemon = True
    t.start()
    return jsonify({"ok": True, "message": "Pipeline started"})

@app.route("/api/status")
def api_status():
    return jsonify(pipeline_status)

@app.route("/api/signals")
def api_signals():
    signals = load_json(SIGNALS_FILE, [])
    today = request.args.get("date", str(date.today()))
    filtered = [s for s in signals if s.get("date") == today]
    return jsonify(filtered)

@app.route("/api/trades")
def api_trades():
    trades = load_json(TRADES_FILE, [])
    return jsonify(trades)

@app.route("/api/summary")
def api_summary():
    signals = load_json(SIGNALS_FILE, [])
    trades  = load_json(TRADES_FILE, [])
    closed  = [t for t in trades if t.get("status") == "closed"]
    open_t  = [t for t in trades if t.get("status") == "open"]
    buy_sigs = [s for s in signals if s.get("is_buy")]

    win_rate = avg_ret = 0.0
    if closed:
        wins     = [t for t in closed if t.get("return_pct", 0) > 0]
        win_rate = round(len(wins) / len(closed) * 100, 1)
        avg_ret  = round(sum(t.get("return_pct", 0) for t in closed) / len(closed), 2)

    # Cumulative P&L series for chart
    cum, series = 0.0, []
    for t in sorted(closed, key=lambda x: x.get("exit_date", "")):
        cum += t.get("return_pct", 0)
        series.append({"date": t["exit_date"], "cum": round(cum, 2)})

    return jsonify({
        "total_signals": len(signals),
        "total_buys": len(buy_sigs),
        "open_trades": len(open_t),
        "closed_trades": len(closed),
        "win_rate": win_rate,
        "avg_return": avg_ret,
        "pnl_series": series,
        **pipeline_status,
    })

# ── FRONTEND ─────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Options Signals</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0a0b;
  --surface: #111113;
  --surface2: #1a1a1e;
  --border: #2a2a30;
  --text: #e8e8ec;
  --muted: #6b6b78;
  --green: #22c55e;
  --green-dim: #14532d;
  --red: #ef4444;
  --red-dim: #450a0a;
  --amber: #f59e0b;
  --blue: #3b82f6;
  --blue-dim: #1e3a5f;
  --mono: 'DM Mono', monospace;
  --sans: 'Syne', sans-serif;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { background: var(--bg); color: var(--text); font-family: var(--sans); }
body { min-height: 100vh; padding-bottom: env(safe-area-inset-bottom); }

/* Header */
.header {
  padding: 20px 20px 0;
  display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
}
.header-left h1 { font-size: 22px; font-weight: 700; letter-spacing: -.02em; }
.header-left p  { font-size: 12px; color: var(--muted); font-family: var(--mono); margin-top: 3px; }
.status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
.dot-ok   { background: var(--green); box-shadow: 0 0 6px var(--green); }
.dot-run  { background: var(--amber); animation: pulse .8s infinite; }
.dot-idle { background: var(--muted); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

/* Run button */
.run-btn {
  background: var(--green);
  color: #000;
  border: none;
  border-radius: 10px;
  padding: 10px 20px;
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
  transition: opacity .15s;
}
.run-btn:disabled { opacity: .4; cursor: default; }
.run-btn:active   { opacity: .7; }

/* Metrics */
.metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  padding: 16px 20px;
}
.metric {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
}
.metric-label { font-size: 10px; color: var(--muted); font-family: var(--mono); letter-spacing: .05em; text-transform: uppercase; }
.metric-val   { font-size: 24px; font-weight: 700; margin-top: 2px; }
.metric-sub   { font-size: 10px; color: var(--muted); font-family: var(--mono); margin-top: 1px; }
.pos { color: var(--green); }
.neg { color: var(--red); }

/* Tabs */
.tabs {
  display: flex; gap: 4px;
  padding: 0 20px 12px;
  overflow-x: auto; -webkit-overflow-scrolling: touch;
}
.tabs::-webkit-scrollbar { display: none; }
.tab {
  padding: 7px 16px;
  border-radius: 20px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  font-family: var(--sans);
  font-size: 13px;
  cursor: pointer;
  white-space: nowrap;
  transition: all .15s;
}
.tab.active {
  background: var(--surface2);
  border-color: #444;
  color: var(--text);
  font-weight: 600;
}

/* Cards */
.cards { padding: 0 20px; display: flex; flex-direction: column; gap: 8px; }
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
  display: flex; gap: 12px; align-items: flex-start;
}
.card.buy-card { border-color: #1a4a2a; background: #0d1f13; }
.card-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 6px; flex-shrink: 0; }
.buy-dot  { background: var(--green); box-shadow: 0 0 6px var(--green); }
.no-dot   { background: var(--border); }
.card-body { flex: 1; min-width: 0; }
.card-top  { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.card-ticker { font-size: 16px; font-weight: 700; font-family: var(--mono); }
.tag {
  font-size: 10px; font-family: var(--mono); font-weight: 500;
  padding: 2px 8px; border-radius: 4px; letter-spacing: .03em;
}
.tag-straddle { background: var(--blue-dim); color: #93c5fd; }
.tag-call     { background: var(--green-dim); color: #86efac; }
.tag-buy      { background: var(--green); color: #000; }
.tag-small    { background: var(--surface2); color: var(--muted); }
.tag-big      { background: #1a1a3a; color: #818cf8; }
.card-detail  { font-size: 12px; color: var(--muted); font-family: var(--mono); margin-top: 5px; }
.card-price   { font-size: 12px; color: var(--text); font-family: var(--mono); margin-top: 2px; }

/* Trade table */
.trade-row {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 16px;
  display: flex; justify-content: space-between; align-items: center; gap: 8px;
}
.trade-left { flex: 1; min-width: 0; }
.trade-ticker { font-size: 15px; font-weight: 700; font-family: var(--mono); }
.trade-meta   { font-size: 11px; color: var(--muted); font-family: var(--mono); margin-top: 3px; }
.trade-ret    { font-size: 16px; font-weight: 700; font-family: var(--mono); text-align: right; }
.bar-wrap { margin-top: 4px; height: 3px; background: var(--border); border-radius: 2px; }
.bar-fill { height: 3px; border-radius: 2px; }

/* Chart */
.chart-wrap { padding: 0 20px; }
.chart-box  { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px; }
.chart-area { position: relative; width: 100%; height: 200px; }

/* Empty */
.empty { text-align: center; padding: 40px 20px; color: var(--muted); font-family: var(--mono); font-size: 13px; }

/* Date picker */
.date-row { padding: 0 20px 10px; display: flex; align-items: center; gap: 10px; }
.date-row input[type=date] {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  color: var(--text); font-family: var(--mono); font-size: 13px; padding: 6px 10px;
}

.section-gap { height: 16px; }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>Options Signals</h1>
    <p id="status-line"><span class="status-dot dot-idle" id="status-dot"></span><span id="status-text">Loading…</span></p>
  </div>
  <button class="run-btn" id="run-btn" onclick="runPipeline()">Run ↗</button>
</div>

<div style="height:16px"></div>

<div class="metrics">
  <div class="metric">
    <div class="metric-label">Buys</div>
    <div class="metric-val" id="m-buys">—</div>
    <div class="metric-sub">all time</div>
  </div>
  <div class="metric">
    <div class="metric-label">Win rate</div>
    <div class="metric-val" id="m-wr">—</div>
    <div class="metric-sub">closed</div>
  </div>
  <div class="metric">
    <div class="metric-label">Avg ret</div>
    <div class="metric-val" id="m-ret">—</div>
    <div class="metric-sub">sim P&L</div>
  </div>
</div>

<div class="tabs">
  <button class="tab active" onclick="showTab('today',this)">Today</button>
  <button class="tab" onclick="showTab('open',this)">Open trades</button>
  <button class="tab" onclick="showTab('closed',this)">History</button>
  <button class="tab" onclick="showTab('chart',this)">P&L chart</button>
</div>

<!-- Today -->
<div id="tab-today">
  <div class="date-row">
    <span style="font-size:12px;color:var(--muted);font-family:var(--mono)">Date:</span>
    <input type="date" id="date-picker" onchange="loadSignals(this.value)">
  </div>
  <div class="cards" id="signals-list">
    <div class="empty">Hit "Run ↗" to generate signals.</div>
  </div>
</div>

<!-- Open -->
<div id="tab-open" style="display:none">
  <div class="cards" id="open-list">
    <div class="empty">No open trades.</div>
  </div>
</div>

<!-- Closed -->
<div id="tab-closed" style="display:none">
  <div class="cards" id="closed-list">
    <div class="empty">No closed trades yet.</div>
  </div>
</div>

<!-- Chart -->
<div id="tab-chart" style="display:none">
  <div class="chart-wrap">
    <div class="chart-box">
      <div style="font-size:12px;color:var(--muted);font-family:var(--mono);margin-bottom:12px">Cumulative simulated P&L %</div>
      <div class="chart-area">
        <canvas id="pnlChart" role="img" aria-label="Cumulative P&L chart">No data yet.</canvas>
      </div>
    </div>
  </div>
</div>

<div class="section-gap"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
let pnlChart = null;
const today = new Date().toISOString().slice(0,10);
document.getElementById('date-picker').value = today;

function showTab(name, btn) {
  ['today','open','closed','chart'].forEach(t =>
    document.getElementById('tab-'+t).style.display = t===name?'':'none');
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (name === 'chart') drawChart();
}

async function runPipeline() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true; btn.textContent = '…';
  setStatus('running');
  try {
    await fetch('/api/run', {method:'POST'});
    // Poll until done
    const poll = setInterval(async () => {
      const s = await (await fetch('/api/status')).json();
      if (!s.running) {
        clearInterval(poll);
        btn.disabled = false; btn.textContent = 'Run ↗';
        setStatus('ok');
        loadAll();
      }
    }, 2000);
  } catch(e) {
    btn.disabled = false; btn.textContent = 'Run ↗';
    setStatus('idle');
  }
}

function setStatus(state) {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  dot.className = 'status-dot ' + (state==='ok'?'dot-ok':state==='running'?'dot-run':'dot-idle');
  if (state==='running') text.textContent = 'Running pipeline…';
}

async function loadStatus() {
  const s = await (await fetch('/api/summary')).json();
  document.getElementById('m-buys').textContent = s.total_buys ?? '0';
  document.getElementById('m-wr').textContent   = s.win_rate ? s.win_rate+'%' : '—';
  const ret = s.avg_return;
  const retEl = document.getElementById('m-ret');
  retEl.textContent = ret != null ? (ret>=0?'+':'')+ret+'%' : '—';
  retEl.className   = 'metric-val ' + (ret>0?'pos':ret<0?'neg':'');

  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  if (s.running) {
    dot.className = 'status-dot dot-run';
    text.textContent = 'Running…';
  } else if (s.last_run) {
    dot.className = 'status-dot dot-ok';
    text.textContent = 'Last run ' + s.last_run + ' · ' + (s.last_result||'');
  } else {
    dot.className = 'status-dot dot-idle';
    text.textContent = 'Never run';
  }
  return s;
}

async function loadSignals(dateStr) {
  const sigs = await (await fetch('/api/signals?date='+dateStr)).json();
  const el = document.getElementById('signals-list');
  if (!sigs.length) { el.innerHTML = '<div class="empty">No signals for '+dateStr+'. Hit Run ↗.</div>'; return; }

  el.innerHTML = sigs.map(s => `
    <div class="card ${s.is_buy?'buy-card':''}">
      <div class="card-dot ${s.is_buy?'buy-dot':'no-dot'}"></div>
      <div class="card-body">
        <div class="card-top">
          <span class="card-ticker">${s.ticker}</span>
          <span class="tag ${s.strategy==='STRADDLE'?'tag-straddle':'tag-call'}">${s.strategy}</span>
          <span class="tag ${s.account==='small'?'tag-small':'tag-big'}">${s.account}</span>
          ${s.is_buy ? '<span class="tag tag-buy">BUY</span>' : ''}
        </div>
        <div class="card-detail">${s.detail||''}</div>
        ${s.price ? `<div class="card-price">$${parseFloat(s.price).toFixed(2)}${s.max_contracts!=null?' · max '+s.max_contracts+' contract'+(s.max_contracts!==1?'s':''):''}</div>` : ''}
      </div>
    </div>`).join('');
}

async function loadTrades() {
  const trades = await (await fetch('/api/trades')).json();
  const open   = trades.filter(t => t.status==='open');
  const closed = trades.filter(t => t.status==='closed').reverse();
  const today  = new Date().toISOString().slice(0,10);

  const openEl = document.getElementById('open-list');
  if (!open.length) { openEl.innerHTML = '<div class="empty">No open trades.</div>'; }
  else {
    openEl.innerHTML = open.map(t => {
      const entry   = new Date(t.entry_date);
      const held    = Math.round((new Date(today)-entry)/86400000);
      const exitDt  = new Date(entry); exitDt.setDate(exitDt.getDate()+t.hold_days);
      const pct     = Math.min(held/t.hold_days*100,100);
      return `<div class="card">
        <div class="card-dot ${t.strategy==='STRADDLE'?'':''}no-dot" style="background:var(--amber)"></div>
        <div class="card-body">
          <div class="card-top">
            <span class="card-ticker">${t.ticker}</span>
            <span class="tag ${t.strategy==='STRADDLE'?'tag-straddle':'tag-call'}">${t.strategy}</span>
          </div>
          <div class="card-detail">Entry ${t.entry_date} · $${parseFloat(t.entry_price).toFixed(2)} · day ${held}/${t.hold_days}</div>
          <div class="bar-wrap"><div class="bar-fill" style="width:${pct}%;background:var(--amber)"></div></div>
        </div>
      </div>`;
    }).join('');
  }

  const closedEl = document.getElementById('closed-list');
  if (!closed.length) { closedEl.innerHTML = '<div class="empty">No closed trades yet.</div>'; }
  else {
    closedEl.innerHTML = closed.map(t => {
      const ret = t.return_pct ?? 0;
      const col = ret>0?'var(--green)':ret<0?'var(--red)':'var(--muted)';
      return `<div class="trade-row">
        <div class="trade-left">
          <div class="trade-ticker">${t.ticker} <span class="tag ${t.strategy==='STRADDLE'?'tag-straddle':'tag-call'}" style="font-size:9px">${t.strategy}</span></div>
          <div class="trade-meta">${t.entry_date} → ${t.exit_date||'?'} · $${parseFloat(t.entry_price).toFixed(2)} → $${parseFloat(t.exit_price||t.entry_price).toFixed(2)}</div>
        </div>
        <div class="trade-ret" style="color:${col}">${ret>=0?'+':''}${ret.toFixed(1)}%</div>
      </div>`;
    }).join('');
  }
}

async function drawChart() {
  const s = await (await fetch('/api/summary')).json();
  const series = s.pnl_series || [];
  if (!series.length || !window.Chart) return;
  const ctx = document.getElementById('pnlChart').getContext('2d');
  if (pnlChart) pnlChart.destroy();
  pnlChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: series.map(p=>p.date),
      datasets: [{
        data: series.map(p=>p.cum),
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34,197,94,0.06)',
        fill: true, tension: 0.4, pointRadius: 3,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#6b6b78', font: { family: 'DM Mono', size: 10 } }, grid: { color: '#1a1a1e' } },
        y: { ticks: { color: '#6b6b78', font: { family: 'DM Mono', size: 10 }, callback: v => v+'%' }, grid: { color: '#1a1a1e' } }
      }
    }
  });
}

async function loadAll() {
  await loadStatus();
  await loadSignals(document.getElementById('date-picker').value);
  await loadTrades();
}

loadAll();
setInterval(loadStatus, 15000);
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
