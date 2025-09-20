# backtest.py
import os, time, math, statistics
from datetime import datetime, timezone
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import ccxt

# =========================
# 설정 (환경변수로 바꿔서 사용)
# =========================
EXCHANGE      = os.getenv("BT_EXCHANGE", "bybit").strip().lower()
SYMBOLS       = [s.strip() for s in os.getenv("BT_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",") if s.strip()]
TIMEFRAME     = os.getenv("BT_TIMEFRAME", "15m")
START_ISO     = os.getenv("BT_START",  "2024-01-01T00:00:00Z")
END_ISO       = os.getenv("BT_END",    "2025-09-01T00:00:00Z")

# 리테스트 전략 파라미터
EMA_FAST      = int(os.getenv("BT_EMA_FAST", "20"))
EMA_TREND     = int(os.getenv("BT_EMA_TREND", "200"))
ATR_LEN       = int(os.getenv("BT_ATR_LEN", "14"))
RETEST_TOL_ATR= float(os.getenv("BT_RETEST_TOL_ATR", "0.3"))
RETEST_WINDOW = int(os.getenv("BT_RETEST_WINDOW", "30"))

# 골든크로스 전략 파라미터
GC_FAST       = int(os.getenv("BT_GC_FAST", "10"))
GC_SLOW       = int(os.getenv("BT_GC_SLOW", "20"))
MACD_FAST     = int(os.getenv("BT_MACD_FAST", "12"))
MACD_SLOW     = int(os.getenv("BT_MACD_SLOW", "26"))
MACD_SIG      = int(os.getenv("BT_MACD_SIG",  "9"))

# 거래량 필터(느슨)
VOL_LOOKBACK  = int(os.getenv("BT_VOL_LOOKBACK", "100"))
VOL_PCTL_LOOSE= int(os.getenv("BT_VOL_PCTL_LOOSE", "30"))  # 하위 30% 이하면 패스

# SL/TP
ATR_SL_MULT   = float(os.getenv("BT_ATR_SL_MULT", "1.5"))   # SL = entry ± 1.5*ATR
RR            = float(os.getenv("BT_RR", "1.5"))            # TP = entry ± RR*(SL폭)

CSV_PATH      = os.getenv("BT_CSV_PATH", "backtest_results.csv")

# =========================
# 유틸
# =========================
def ts_to_ms(dt_iso: str) -> int:
    return int(pd.Timestamp(dt_iso).tz_convert("UTC").timestamp() * 1000)

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def macd_line(close: pd.Series, fast=12, slow=26):
    return close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()

def macd_indicator(close: pd.Series, fast=12, slow=26, sig=9):
    macd = macd_line(close, fast, slow)
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def build_exchange():
    if EXCHANGE == "bybit":
        return ccxt.bybit({"enableRateLimit": True})
    elif EXCHANGE in ("binanceusdm", "binance_perp", "binance_futures"):
        return ccxt.binanceusdm({"enableRateLimit": True})
    else:
        return ccxt.binance({"enableRateLimit": True})

# OHLCV 페이징 수집
def fetch_ohlcv_range(ex, symbol, timeframe, since_ms, until_ms, step=1500) -> pd.DataFrame:
    out = []
    ms = since_ms
    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=ms, limit=step)
        if not ohlcv:
            break
        out.extend(ohlcv)
        last_ms = ohlcv[-1][0]
        # 다음 시작점으로 한 틱 증가
        ms = last_ms + 1
        # 끝에 도달하면 종료
        if last_ms >= until_ms or len(ohlcv) < step:
            break
        time.sleep(ex.rateLimit/1000.0)
    if not out:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume","dt"])
    df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["ema_fast"]  = ema(df["close"], EMA_FAST)
    df["ema_trend"] = ema(df["close"], EMA_TREND)
    df["atr"]       = atr(df, ATR_LEN)
    df["ema_gc_fast"] = ema(df["close"], GC_FAST)
    df["ema_gc_slow"] = ema(df["close"], GC_SLOW)
    macd, sig, hist = macd_indicator(df["close"], MACD_FAST, MACD_SLOW, MACD_SIG)
    df["macd"] = macd
    df["macd_sig"] = sig
    # 거래량 백분위
    if len(df) >= VOL_LOOKBACK:
        roll = df["volume"].rolling(VOL_LOOKBACK)
        df["vol_pct"] = roll.apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]*100, raw=False)
    else:
        df["vol_pct"] = np.nan
    return df

# 첫 터치로 청산
def simulate_trade(df: pd.DataFrame, i_entry: int, side: str, entry: float, atr_now: float, rr=RR) -> Tuple[int, float, str, float]:
    risk = ATR_SL_MULT * atr_now
    if side == "LONG":
        sl = entry - risk
        tp = entry + rr * risk
        for j in range(i_entry+1, len(df)):
            lo, hi, cls = df.loc[j, "low"], df.loc[j, "high"], df.loc[j, "close"]
            if lo <= sl:
                return j, sl, "SL", (sl/entry - 1)*100.0
            if hi >= tp:
                return j, tp, "TP", (tp/entry - 1)*100.0
        return len(df)-1, df.iloc[-1]["close"], "EOT", (df.iloc[-1]["close"]/entry - 1)*100.0
    else:
        sl = entry + risk
        tp = entry - rr * risk
        for j in range(i_entry+1, len(df)):
            lo, hi, cls = df.loc[j, "low"], df.loc[j, "high"], df.loc[j, "close"]
            if hi >= sl:
                return j, sl, "SL", (1 - sl/entry)*100.0
            if lo <= tp:
                return j, tp, "TP", (1 - tp/entry)*100.0
        return len(df)-1, df.iloc[-1]["close"], "EOT", (1 - df.iloc[-1]["close"]/entry)*100.0

# 리테스트 시그널 탐지 (확인형)
def signals_retest(df: pd.DataFrame) -> List[Dict]:
    sigs = []
    window_left = 0
    waiting = False
    rc_high = rc_low = None
    trend = None

    for i in range(max(EMA_TREND, ATR_LEN)+5, len(df)):
        row = df.iloc[i]
        price = row["close"]
        trend_now = "up" if price > row["ema_trend"] else "down"

        # 트렌드 바뀌면 리셋
        if trend != trend_now:
            trend = trend_now
            waiting = False
            window_left = RETEST_WINDOW

        if not waiting and window_left > 0:
            # 리테스트 근접?
            if row["atr"] > 0 and abs(price - row["ema_fast"]) <= RETEST_TOL_ATR * row["atr"]:
                waiting = True
                rc_high, rc_low = row["high"], row["low"]
            window_left -= 1
            continue

        if waiting:
            # 확인봉
            if trend == "up" and price >= rc_high:
                sigs.append({"i": i, "side": "LONG", "entry": price, "atr": row["atr"], "strategy": "RETEST"})
                waiting = False
            elif trend == "down" and price <= rc_low:
                sigs.append({"i": i, "side": "SHORT", "entry": price, "atr": row["atr"], "strategy": "RETEST"})
                waiting = False
    return sigs

# 골든크로스 + 느슨한 거래량 + MACD 동조
def signals_golden(df: pd.DataFrame) -> List[Dict]:
    sigs = []
    for i in range(max(GC_SLOW, MACD_SLOW, ATR_LEN)+1, len(df)):
        f_prev, s_prev = df.iloc[i-1][["ema_gc_fast","ema_gc_slow"]]
        f_now,  s_now  = df.iloc[i][["ema_gc_fast","ema_gc_slow"]]
        macd_now, sig_now = df.iloc[i][["macd","macd_sig"]]
        vol_pct = df.iloc[i]["vol_pct"]
        atr_now = df.iloc[i]["atr"]
        price   = df.iloc[i]["close"]

        # 느슨한 거래량 필터 (lookback 있으면만)
        vol_ok = True
        if not math.isnan(vol_pct):
            vol_ok = vol_pct >= VOL_PCTL_LOOSE

        # 크로스
        long_gc  = (f_prev <= s_prev) and (f_now > s_now)
        short_dc = (f_prev >= s_prev) and (f_now < s_now)

        if long_gc and vol_ok and macd_now > sig_now:
            sigs.append({"i": i, "side":"LONG", "entry": price, "atr": atr_now, "strategy":"GOLDEN"})
        if short_dc and vol_ok and macd_now < sig_now:
            sigs.append({"i": i, "side":"SHORT", "entry": price, "atr": atr_now, "strategy":"GOLDEN"})
    return sigs

def run_for_symbol(ex, symbol) -> pd.DataFrame:
    df = fetch_ohlcv_range(ex, symbol, TIMEFRAME, ts_to_ms(START_ISO), ts_to_ms(END_ISO))
    if df.empty:
        print(f"[{symbol}] no data")
        return pd.DataFrame()
    df = add_indicators(df).dropna().reset_index(drop=True)

    # 시그널 집계(겹치면 시간 순서대로 처리)
    sigs = signals_retest(df) + signals_golden(df)
    sigs = sorted(sigs, key=lambda x: x["i"])

    trades = []
    last_exit_i = -1
    for s in sigs:
        i = s["i"]
        if i <= last_exit_i:
            continue  # 포지션 보유 중이면 무시(단일 포지션 룰)
        entry = float(s["entry"])
        atr_now = float(s["atr"])
        side = s["side"]
        exit_i, exit_price, outcome, pnl_pct = simulate_trade(df, i, side, entry, atr_now, RR)
        bars_held = exit_i - i
        trades.append({
            "symbol": symbol,
            "strategy": s["strategy"],
            "side": side,
            "entry_ts": int(df.loc[i,"ts"]),
            "entry_dt": df.loc[i,"dt"],
            "entry": round(entry, 4),
            "atr": round(atr_now, 4),
            "rr": RR,
            "sl_mult": ATR_SL_MULT,
            "exit_ts": int(df.loc[exit_i,"ts"]),
            "exit_dt": df.loc[exit_i,"dt"],
            "exit_price": round(float(exit_price), 4),
            "outcome": outcome,  # TP / SL / EOT
            "pnl_pct": round(float(pnl_pct), 4),
            "bars_held": int(bars_held),
        })
        last_exit_i = exit_i

    return pd.DataFrame(trades)

def print_summary(df: pd.DataFrame, title="ALL"):
    if df.empty:
        print(f"[{title}] No trades")
        return
    n = len(df)
    wins = (df["outcome"]=="TP").sum()
    losses = (df["outcome"]=="SL").sum()
    eots = (df["outcome"]=="EOT").sum()
    winrate = wins / n * 100
    avg = df["pnl_pct"].mean()
    med = df["pnl_pct"].median()
    pf = df.loc[df["pnl_pct"]>0, "pnl_pct"].sum() / abs(df.loc[df["pnl_pct"]<0, "pnl_pct"].sum()) if (df["pnl_pct"]<0).any() else float('inf')
    avg_hold = df["bars_held"].mean()

    # 간단 equity & MDD
    equity = (1 + df["pnl_pct"]/100.0).cumprod()
    peak = equity.cummax()
    dd = (equity/peak - 1.0) * 100.0
    mdd = dd.min()

    print(f"\n=== SUMMARY [{title}] ===")
    print(f"Trades: {n} | TP: {wins}  SL: {losses}  EOT: {eots}")
    print(f"Winrate: {winrate:.2f}%  PF: {pf:.2f}")
    print(f"Avg PnL: {avg:.2f}%  Median: {med:.2f}%")
    print(f"Avg bars held: {avg_hold:.1f}")
    print(f"Max Drawdown: {mdd:.2f}%")

def main():
    ex = build_exchange()
    all_trades = []
    for sym in SYMBOLS:
        print(f"[{sym}] fetching & backtesting ...")
        df_tr = run_for_symbol(ex, sym)
        if not df_tr.empty:
            print_summary(df_tr, title=sym)
            all_trades.append(df_tr)
    if not all_trades:
        print("No trades overall.")
        return
    res = pd.concat(all_trades, ignore_index=True)
    print_summary(res, title="ALL")
    res.to_csv(CSV_PATH, index=False)
    print(f"\nSaved CSV -> {CSV_PATH}")

if __name__ == "__main__":
    main()
