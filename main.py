import os, time, math, json, requests
from datetime import datetime, timezone
from threading import Thread

import numpy as np
import pandas as pd
import ccxt

# ===================== 환경/파라미터 =====================
# 심볼들: 콤마로 구분
SYMBOLS_STR  = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT")
SYMBOLS      = [s.strip() for s in SYMBOLS_STR.split(",") if s.strip()]

TIMEFRAME    = os.getenv("TIMEFRAME", "15m")
LIMIT        = int(os.getenv("LIMIT", "600"))
POLL_SEC     = int(os.getenv("POLL_SEC", "15"))

EMA_SHORT    = int(os.getenv("EMA_SHORT", "10"))
EMA_LONG     = int(os.getenv("EMA_LONG", "20"))
EMA_TREND    = int(os.getenv("EMA_TREND", "200"))
USE_TREND    = os.getenv("USE_TREND", "true").lower() == "true"

VOL_MA_LEN     = int(os.getenv("VOL_MA_LEN", "20"))
NEED_VOL_BOOST = os.getenv("NEED_VOL_BOOST", "true").lower() == "true"

ATR_LEN      = int(os.getenv("ATR_LEN", "14"))

# 간이 HVN(매물대) 필터
REQUIRE_HVN_NEAR = os.getenv("REQUIRE_HVN_NEAR", "false").lower() == "true"
HVN_BINS         = int(os.getenv("HVN_BINS", "60"))
HVN_PEAK_TOPK    = int(os.getenv("HVN_PEAK_TOPK", "5"))
HVN_TOL_ATR      = float(os.getenv("HVN_TOL_ATR", "0.8"))

COOLDOWN_MIN     = int(os.getenv("COOLDOWN_MIN", "60"))  # 동일 방향 재알림 최소 간격(분)

EXCHANGE_NAME    = os.getenv("EXCHANGE", "binance").lower().strip()
# "bybit" 로 바꾸면 bybit 사용, 선물은 binanceusdm 사용 가능

# ===================== 텔레그램 =====================
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")

def send_telegram(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        print("[WARN] Telegram env not set; msg skipped.")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code >= 400:
            print("[TG-ERR]", r.status_code, r.text[:200])
    except Exception as e:
        print("[TG-EXC]", e)

# ===================== 거래소/데이터 =====================
def build_exchange():
    if EXCHANGE_NAME == "bybit":
        ex = ccxt.bybit({"enableRateLimit": True})
        # 선물 사용 시 필요:
        # ex.options = {'defaultType': 'swap'}
    elif EXCHANGE_NAME in ("binanceusdm", "binance_futures", "binance_perp"):
        ex = ccxt.binanceusdm({"enableRateLimit": True})
    else:
        ex = ccxt.binance({"enableRateLimit": True})
    return ex

ex = build_exchange()

def fetch_ohlcv(symbol, timeframe, limit):
    ohlcv = ex.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("UTC")
    return df

# ===================== 지표 =====================
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

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_s"] = ema(df["close"], EMA_SHORT)
    df["ema_l"] = ema(df["close"], EMA_LONG)
    df["ema_t"] = ema(df["close"], EMA_TREND)
    df["vol_ma"] = df["volume"].rolling(VOL_MA_LEN).mean()
    df["atr"]    = atr(df, ATR_LEN)
    return df

# ---- 간이 HVN ----
def volume_profile_hvn(df: pd.DataFrame, bins=60, topk=5):
    closes = df["close"].values
    vols   = df["volume"].values
    if len(closes) < 5:
        return []
    lo, hi = closes.min(), closes.max()
    if lo == hi:
        return [lo]
    hist_vol, edges = np.histogram(closes, bins=bins, range=(lo, hi), weights=vols)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx = np.argsort(hist_vol)[::-1][:topk]
    peaks = centers[idx]
    return sorted(peaks)

def is_near_hvn(current_price: float, peaks: list, tol_atr: float, atr_value: float) -> bool:
    if atr_value <= 0 or not peaks:
        return False
    tol = tol_atr * atr_value
    return any(abs(current_price - p) <= tol for p in peaks)

# ===================== 신호/포맷 =====================
def cross_up(a: pd.Series, b: pd.Series) -> bool:
    return a.iloc[-2] <= b.iloc[-2] and a.iloc[-1] > b.iloc[-1]

def cross_down(a: pd.Series, b: pd.Series) -> bool:
    return a.iloc[-2] >= b.iloc[-2] and a.iloc[-1] < b.iloc[-1]

def build_message(sym: str, side: str, last: pd.Series, atr_v: float, why: str):
    dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    msg = [
        f"🔔 <b>{sym}</b> {TIMEFRAME}  <b>{side}</b>",
        f"가격: <b>{last['close']:.2f}</b>",
        f"EMA10/20: {last['ema_s']:.2f} / {last['ema_l']:.2f}",
        f"EMA200: {last['ema_t']:.2f}",
        f"거래량: {last['volume']:.3f} (평균 {last['vol_ma']:.3f})",
        f"ATR({ATR_LEN}): {atr_v:.2f}",
        f"이유: {why}",
        f"<i>{dt}</i>"
    ]
    return "\n".join(msg)

# 심볼별 쿨다운(최근 알림 시각)
_last_alert_ts_long  = {}
_last_alert_ts_short = {}

def check_signals_for_symbol(sym: str, df: pd.DataFrame):
    global _last_alert_ts_long, _last_alert_ts_short

    last = df.iloc[-1]
    atr_v = float(last["atr"]) if not math.isnan(last["atr"]) else 0.0

    # 필터
    trend_ok_long  = (not USE_TREND) or (last["close"] > last["ema_t"])
    trend_ok_short = (not USE_TREND) or (last["close"] < last["ema_t"])
    vol_ok = (not NEED_VOL_BOOST) or (last["volume"] >= last["vol_ma"])

    hvn_ok = True
    if REQUIRE_HVN_NEAR:
        peaks = volume_profile_hvn(df, bins=HVN_BINS, topk=HVN_PEAK_TOPK)
        hvn_ok = is_near_hvn(float(last["close"]), peaks, HVN_TOL_ATR, atr_v)

    # 교차
    long_cross  = cross_up(df["ema_s"], df["ema_l"])
    short_cross = cross_down(df["ema_s"], df["ema_l"])

    now = time.time()
    cd_sec = COOLDOWN_MIN * 60

    # dict 기본값
    _last_alert_ts_long.setdefault(sym, 0.0)
    _last_alert_ts_short.setdefault(sym, 0.0)

    # 롱
    if long_cross and trend_ok_long and vol_ok and hvn_ok:
        if now - _last_alert_ts_long[sym] >= cd_sec:
            _last_alert_ts_long[sym] = now
            why = ["EMA10 ↑ EMA20"]
            if USE_TREND:      why.append("가격 > EMA200")
            if NEED_VOL_BOOST: why.append("거래량 ≥ 평균")
            if REQUIRE_HVN_NEAR: why.append("HVN 근접")
            send_telegram(build_message(sym, "LONG", last, atr_v, ", ".join(why)))

    # 숏
    if short_cross and trend_ok_short and vol_ok and hvn_ok:
        if now - _last_alert_ts_short[sym] >= cd_sec:
            _last_alert_ts_short[sym] = now
            why = ["EMA10 ↓ EMA20"]
            if USE_TREND:      why.append("가격 < EMA200")
            if NEED_VOL_BOOST: why.append("거래량 ≥ 평균")
            if REQUIRE_HVN_NEAR: why.append("HVN 근접")
            send_telegram(build_message(sym, "SHORT", last, atr_v, ", ".join(why)))

# ===================== 메인 루프(3종 순회) =====================
def main_loop():
    print(f"[bot] start: EXCHANGE={EXCHANGE_NAME}, SYMBOLS={SYMBOLS}, TF={TIMEFRAME}")
    while True:
        try:
            for sym in SYMBOLS:
                try:
                    df = fetch_ohlcv(sym, TIMEFRAME, LIMIT)
                    df = add_indicators(df).dropna()
                    if len(df) >= max(EMA_TREND, VOL_MA_LEN, ATR_LEN) + 5:
                        check_signals_for_symbol(sym, df)
                    # 심볼 사이 살짝 텀
                    time.sleep(0.5)
                except ccxt.BaseError as e:
                    print(f"[ccxt:{sym}] error:", e)
                except Exception as e:
                    print(f"[loop:{sym}] error:", e)
        except Exception as e:
            print("[loop] outer error:", e)
        time.sleep(POLL_SEC)

# ===================== Flask(Web Service) - 맨 아래(방법 A) =====================
from flask import Flask   # 위에서 import 했어도 재선언 무방(같은 이름)
app = Flask(__name__)

@app.get("/")
def health():
    body = {
        "status": "ok",
        "exchange": EXCHANGE_NAME,
        "symbols": SYMBOLS,
        "timeframe": TIMEFRAME,
        "utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return (json.dumps(body), 200, {"Content-Type": "application/json"})

@app.get("/test")
def test():
    send_telegram("✅ [투자봇] 텔레그램 연결 테스트")
    return "sent", 200

# gunicorn import 시 1회만 워커 시작
_worker_started = False
def _start_worker_once():
    global _worker_started
    if not _worker_started:
        _worker_started = True
        Thread(target=main_loop, daemon=True).start()
        # send_telegram("🔔[투자봇] Render(Web Service)에서 시작")

_start_worker_once()

# 로컬 개발용(직접 실행 시)
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    print("Starting dev Flask ...")
    app.run(host="0.0.0.0", port=port)
