import os, time, math, json, random, requests
from datetime import datetime, timezone
from threading import Thread

import numpy as np
import pandas as pd
import ccxt
from flask import Flask

# ===================== 환경 변수 =====================
if not os.getenv("RENDER"):
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass

EXCHANGE = os.getenv("EXCHANGE", "bybit").strip().lower()
SYMBOLS  = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
POLL_SEC  = int(os.getenv("POLL_SEC", "15"))

# 지표 파라미터
EMA_FAST  = int(os.getenv("EMA_FAST", "20"))
EMA_TREND = int(os.getenv("EMA_TREND", "200"))
EMA_SHORT = int(os.getenv("EMA_SHORT", "50"))   # 골든크로스 단기
EMA_LONG  = int(os.getenv("EMA_LONG", "200"))   # 골든크로스 장기
ATR_LEN   = int(os.getenv("ATR_LEN", "14"))

# 리테스트 규칙
RETEST_TOL_ATR = float(os.getenv("RETEST_TOL_ATR", "0.3"))
RETEST_WINDOW  = int(os.getenv("RETEST_WINDOW", "30"))

# SL/TP (A안: ATR 기반)
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))
RR          = float(os.getenv("RR", "2.0"))

# 쿨다운
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "60"))

# 텔레그램
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")

# 호출 간격/백오프
FETCH_MIN_SEC = int(os.getenv("FETCH_MIN_SEC", "55"))
BACKOFF_START = int(os.getenv("BACKOFF_START", "60"))
BACKOFF_MAX   = int(os.getenv("BACKOFF_MAX", "600"))
JITTER_MAX    = int(os.getenv("JITTER_MAX", "3"))

# ===================== 유틸/알림 =====================
def send_telegram(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        print("[WARN] Telegram env not set; msg skipped.")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code >= 400:
            print("[TG-ERR]", r.status_code, r.text[:300])
    except Exception as e:
        print("[TG-EXC]", e)

def fmt_price(x: float) -> str:
    return f"{x:,.2f}"

def fmt_pct(p: float) -> str:
    return f"{p:.2f}%"

# ===================== 거래소/데이터 =====================
def build_exchange():
    if EXCHANGE == "bybit":
        return ccxt.bybit({"enableRateLimit": True})
    else:
        return ccxt.binanceusdm({"enableRateLimit": True})

ex = build_exchange()

_last_fetch = {}
_df_cache   = {}
_backoff    = 0

def fetch_ohlcv_throttled(symbol, timeframe, limit=500):
    global _last_fetch, _df_cache, _backoff
    now = time.time()
    last_t = _last_fetch.get(symbol, 0)
    if now - last_t < FETCH_MIN_SEC and symbol in _df_cache:
        return _df_cache[symbol]
    if JITTER_MAX > 0:
        time.sleep(random.uniform(0, JITTER_MAX))
    if _backoff > 0:
        time.sleep(_backoff)
    try:
        ohlcv = ex.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        _df_cache[symbol]  = df
        _last_fetch[symbol] = now
        _backoff = max(0, int(_backoff / 2))
        return df
    except ccxt.BaseError as e:
        msg = str(e)
        if ("429" in msg) or ("418" in msg):
            _backoff = BACKOFF_START if _backoff == 0 else min(BACKOFF_MAX, _backoff * 2)
            print(f"[rate-limit] backoff={_backoff}s; {symbol}; {msg[:100]}")
        if symbol in _df_cache:
            return _df_cache[symbol]
        return pd.DataFrame(columns=["ts","open","high","low","close","volume","dt"])

# ===================== 지표 =====================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    if df.empty: return pd.Series(dtype="float64")
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df["ema_fast"]  = ema(df["close"], EMA_FAST)
    df["ema_trend"] = ema(df["close"], EMA_TREND)
    df["ema_short"] = ema(df["close"], EMA_SHORT)
    df["ema_long"]  = ema(df["close"], EMA_LONG)
    df["atr"]       = atr(df, ATR_LEN)
    return df

# ===================== Barrier 계산 (Pivot + PrevHL + HTF + HVN) =====================
def find_pivot_levels(df, left=3, right=3, lookback=300):
    highs, lows = [], []
    n = len(df)
    start = max(0, n - lookback)
    for i in range(start + left, n - right):
        hh, ll = df["high"].iloc[i], df["low"].iloc[i]
        if all(hh > df["high"].iloc[i-k-1] for k in range(left)) and \
           all(hh >= df["high"].iloc[i+k+1] for k in range(right)):
            highs.append(float(hh))
        if all(ll < df["low"].iloc[i-k-1] for k in range(left)) and \
           all(ll <= df["low"].iloc[i+k+1] for k in range(right)):
            lows.append(float(ll))
    return sorted(set(highs)), sorted(set(lows))

def prev_day_hl(df):
    if df.empty: return None, None
    ddf = df.set_index(df["dt"]).resample("1D").agg({"high":"max","low":"min"})
    if len(ddf) < 2: return None, None
    prev = ddf.iloc[-2]
    return float(prev["high"]), float(prev["low"])

def htf_levels(symbol, ema_len=200, don_len=55, tf="1h"):
    try:
        df_htf = fetch_ohlcv_throttled(symbol, tf, limit=max(ema_len, don_len)+5)
        df_htf = add_indicators(df_htf).dropna()
        if df_htf.empty: return None, None, None
        ema200_val = float(df_htf["ema_trend"].iloc[-1])
        high_dc = float(df_htf["high"].rolling(don_len).max().iloc[-1])
        low_dc  = float(df_htf["low"].rolling(don_len).min().iloc[-1])
        return ema200_val, high_dc, low_dc
    except: return None, None, None

def approx_hvn_poc(df, bins=60, lookback=400):
    if df.empty: return [], None
    seg = df.iloc[-lookback:].copy()
    lo, hi = float(seg["low"].min()), float(seg["high"].max())
    if hi <= lo: return [], None
    edges = np.linspace(lo, hi, bins+1)
    hist  = np.zeros(bins)
    for _, row in seg.iterrows():
        h, l, v = row["high"], row["low"], row["volume"]
        if h <= l or v <= 0: continue
        a, b = int((l-lo)/(hi-lo)*bins), int((h-lo)/(hi-lo)*bins)
        a, b = max(0,a), min(bins-1,b)
        hist[a:b+1] += v / max(1,b-a+1)
    poc_idx = int(hist.argmax())
    poc_price = float((edges[poc_idx]+edges[poc_idx+1])/2)
    thr = np.percentile(hist[hist>0],85) if hist.sum()>0 else 0
    peaks=[]
    for i in range(1,bins-1):
        if hist[i]>=hist[i-1] and hist[i]>=hist[i+1] and hist[i]>=thr:
            peaks.append(float((edges[i]+edges[i+1])/2))
    return sorted(set(peaks)), poc_price

def pick_barrier(side, entry, df, symbol):
    candidates=[]
    phs,pls=find_pivot_levels(df)
    if side=="LONG": candidates+=[x for x in phs if x>entry]
    else: candidates+=[x for x in pls if x<entry]
    ph,pl=prev_day_hl(df)
    if side=="LONG" and ph and ph>entry: candidates.append(ph)
    if side=="SHORT" and pl and pl<entry: candidates.append(pl)
    ema200,dc_hi,dc_lo=htf_levels(symbol)
    if side=="LONG":
        for lv in [ema200,dc_hi]:
            if lv and lv>entry:candidates.append(lv)
    else:
        for lv in [ema200,dc_lo]:
            if lv and lv<entry:candidates.append(lv)
    peaks,poc=approx_hvn_poc(df)
    if side=="LONG":
        candidates+=[x for x in peaks if x>entry]
        if poc and poc>entry:candidates.append(poc)
    else:
        candidates+=[x for x in peaks if x<entry]
        if poc and poc<entry:candidates.append(poc)
    if not candidates:return None,[]
    if side=="LONG":
        barrier=min(candidates); refs=sorted(candidates)[:3]
    else:
        barrier=max(candidates); refs=sorted(candidates,reverse=True)[:3]
    return float(barrier),[float(x) for x in refs]

# ===================== SL/TP 계산 =====================
def compute_levels(side, entry, atr_v):
    risk=ATR_SL_MULT*atr_v
    if side=="LONG":
        sl, tp = entry-risk, entry+RR*risk
        sl_diff, tp_diff = entry-sl, tp-entry
    else:
        sl, tp = entry+risk, entry-RR*risk
        sl_diff, tp_diff = sl-entry, entry-tp
    sl_pct=(sl_diff/entry)*100 if entry else 0
    tp_pct=(tp_diff/entry)*100 if entry else 0
    return round(sl,2),round(tp,2),round(sl_diff,2),round(tp_diff,2),sl_pct,tp_pct

# ===================== 시그널 로직 =====================
_last_alert_ts={}

def maybe_alert(sym, side, entry, last, rc_info, sig_type):
    key=(sym,side,sig_type)
    now=time.time()
    if key not in _last_alert_ts: _last_alert_ts[key]=0
    if now-_last_alert_ts[key]<COOLDOWN_MIN*60: return
    atr_v=float(last["atr"])
    sl,tp,sl_diff,tp_diff,sl_pct,tp_pct=compute_levels(side,entry,atr_v)
    barrier,refs=pick_barrier(side,entry,last.to_frame().T,sym)
    rr_note=""
    if barrier:
        dist=abs(barrier-entry)
        risk=abs(entry-sl)
        rr=dist/risk if risk>0 else 0
        rr_note=f"장벽: {fmt_price(barrier)} (RR≈{rr:.2f})"
        if rr<1.5: rr_note+=" ❌부족"
        else: rr_note+=" ✅통과"
    dt=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines=[
        f"🔔 <b>{sym}</b> {TIMEFRAME} <b>{side}</b> ({sig_type})",
        f"Entry: <b>{fmt_price(entry)}</b>",
        f"SL: {fmt_price(sl)} (-{fmt_price(sl_diff)}, -{fmt_pct(sl_pct)})",
        f"TP: {fmt_price(tp)} (+{fmt_price(tp_diff)}, +{fmt_pct(tp_pct)})",
    ]
    if rr_note: lines.append(rr_note)
    lines.append(f"<i>{dt}</i>")
    send_telegram("\n".join(lines))
    _last_alert_ts[key]=now

# 리테스트 상태
state={}

def reset_state_for(symbol,trend):
    state[symbol]={"trend":trend,"window_left":RETEST_WINDOW,"waiting_confirm":False,"retest_candle":None}

def ensure_state(symbol,trend):
    if symbol not in state or state[symbol]["trend"]!=trend:
        reset_state_for(symbol,trend)

def process_retest(sym,df):
    last=df.iloc[-1]
    trend="up" if last["close"]>last["ema_trend"] else "down"
    ensure_state(sym,trend)
    st=state[sym]
    if not st["waiting_confirm"] and st["window_left"]>0:
        atr_v=float(last["atr"])
        if abs(last["close"]-last["ema_fast"])<=RETEST_TOL_ATR*atr_v:
            st["retest_candle"]={"high":last["high"],"low":last["low"],"close":last["close"]}
            st["waiting_confirm"]=True
        st["window_left"]-=1
    if st["waiting_confirm"] and st["retest_candle"]:
        rc=st["retest_candle"]
        if trend=="up" and last["close"]>=rc["high"]:
            maybe_alert(sym,"LONG",last["close"],last,rc,"리테스트")
            reset_state_for(sym,"up")
        elif trend=="down" and last["close"]<=rc["low"]:
            maybe_alert(sym,"SHORT",last["close"],last,rc,"리테스트")
            reset_state_for(sym,"down")

def process_golden(sym,df):
    prev,cur=df.iloc[-2],df.iloc[-1]
    if prev["ema_short"]<prev["ema_long"] and cur["ema_short"]>=cur["ema_long"]:
        maybe_alert(sym,"LONG",cur["close"],cur,{},"골든크로스")
    elif prev["ema_short"]>prev["ema_long"] and cur["ema_short"]<=cur["ema_long"]:
        maybe_alert(sym,"SHORT",cur["close"],cur,{},"데드크로스")

def process_symbol(sym):
    df=fetch_ohlcv_throttled(sym,TIMEFRAME,limit=300)
    df=add_indicators(df).dropna()
    if len(df)<max(EMA_TREND,ATR_LEN)+5: return
    process_retest(sym,df)
    process_golden(sym,df)

# ===================== 메인 루프 =====================
def main_loop():
    for sym in SYMBOLS:
        try:
            df0=fetch_ohlcv_throttled(sym,TIMEFRAME,limit=300)
            df0=add_indicators(df0).dropna()
            trend="up" if len(df0)>0 and df0.iloc[-1]["close"]>df0.iloc[-1]["ema_trend"] else "down"
            reset_state_for(sym,trend)
        except: reset_state_for(sym,"up")
    while True:
        for sym in SYMBOLS:
            try: process_symbol(sym); time.sleep(0.4)
            except Exception as e: print(f"[loop:{sym}] {e}")
        time.sleep(POLL_SEC)

# ===================== Flask =====================
app=Flask(__name__)

@app.get("/")
def health():
    return {"status":"ok","exchange":EXCHANGE,"symbols":SYMBOLS,"timeframe":TIMEFRAME}

@app.get("/test")
def test():
    send_telegram("✅ [투자봇] 텔레그램 연결 테스트")
    return "sent",200

_worker_started=False
def _start_worker_once():
    global _worker_started
    if not _worker_started:
        _worker_started=True
        Thread(target=main_loop,daemon=True).start()
_start_worker_once()

if __name__=="__main__":
    port=int(os.getenv("PORT","10000"))
    app.run(host="0.0.0.0",port=port)
