import os, time, math, json, requests
from datetime import datetime, timezone
from threading import Thread

import numpy as np
import pandas as pd
import ccxt

from flask import Flask
@app.get("/test")
def test():
    send_telegram("✅ [투자봇] 텔레그램 연결 테스트")
    return "sent", 200

# ===================== 사용자/환경 설정 =====================
SYMBOL       = os.getenv("SYMBOL", "BTC/USDT")   # 심볼
TIMEFRAME    = os.getenv("TIMEFRAME", "15m")     # 15분봉
LIMIT        = int(os.getenv("LIMIT", "600"))    # 최근 캔들 개수(충분히 크게)
POLL_SEC     = int(os.getenv("POLL_SEC", "15"))  # 폴링 주기(초)

EMA_SHORT    = int(os.getenv("EMA_SHORT", "10"))
EMA_LONG     = int(os.getenv("EMA_LONG", "20"))
EMA_TREND    = int(os.getenv("EMA_TREND", "200"))
USE_TREND    = os.getenv("USE_TREND", "true").lower() == "true"

VOL_MA_LEN   = int(os.getenv("VOL_MA_LEN", "20"))
NEED_VOL_BOOST = os.getenv("NEED_VOL_BOOST", "true").lower() == "true"

ATR_LEN      = int(os.getenv("ATR_LEN", "14"))

# 간이 HVN(매물대) 근접 사용 여부 및 파라미터
REQUIRE_HVN_NEAR = os.getenv("REQUIRE_HVN_NEAR", "false").lower() == "true"
HVN_BINS         = int(os.getenv("HVN_BINS", "60"))   # 가격구간 나누는 개수
HVN_PEAK_TOPK    = int(os.getenv("HVN_PEAK_TOPK", "5"))  # 상위 매물대 개수
HVN_TOL_ATR      = float(os.getenv("HVN_TOL_ATR", "0.8")) # 현재가가 매물대와 이 정도*ATR 이내면 근접

# 알림 쿨다운
COOLDOWN_MIN     = int(os.getenv("COOLDOWN_MIN", "60"))  # 같은 방향 재알림 최소 간격(분)

# 거래소 선택 (기본: binance)
EXCHANGE_NAME    = os.getenv("EXCHANGE", "binance").lower().strip()
# "bybit" 로 바꾸고 싶으면 EXCHANGE=bybit, 선물은 ex.options 설정을 추가하세요.

# ===================== 텔레그램 =====================
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7898891479:AAEOdkJEupPj9k_t4YlhqvYJd7Jbsot_5ao")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "8095272059")

def send_telegram(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        print("[WARN] Telegram env not set; msg skipped.")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {
            "chat_id": TG_CHAT,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code >= 400:
            print(f"[ERR] Telegram {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print("[ERR] Telegram error:", e)

# ===================== 거래소/데이터 =====================
def build_exchange():
    if EXCHANGE_NAME == "bybit":
        ex = ccxt.bybit({"enableRateLimit": True})
        # 필요시 선물(퍼프추얼) 기본 타입 설정
        # ex.options = {'defaultType': 'swap'}
    elif EXCHANGE_NAME in ("binanceusdm", "binance_futures", "binance_perp"):
        ex = ccxt.binanceusdm({"enableRateLimit": True})
    else:
        ex = ccxt.binance({"enableRateLimit": True})
    return ex

ex = build_exchange()

def fetch_ohlcv(symbol, timeframe, limit):
    """ccxt OHLCV -> pandas DataFrame"""
    ohlcv = ex.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
    cols = ["ts","open","high","low","close","volume"]
    df = pd.DataFrame(ohlcv, columns=cols)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("UTC")
    return df

# ===================== 지표 계산 =====================
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
    df["atr"] = atr(df, ATR_LEN)
    return df

# 간이 매물대(Volume by Close Price) 계산
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

# ===================== 신호 로직 =====================
_last_alert_ts_long  = 0.0
_last_alert_ts_short = 0.0

def cross_up(a: pd.Series, b: pd.Series) -> bool:
    return a.iloc[-2] <= b.iloc[-2] and a.iloc[-1] > b.iloc[-1]

def cross_down(a: pd.Series, b: pd.Series) -> bool:
    return a.iloc[-2] >= b.iloc[-2] and a.iloc[-1] < b.iloc[-1]

def build_message(side: str, last: pd.Series, atr_v: float, why: str):
    dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    price = last["close"]
    msg = [
        f"🔔 <b>{SYMBOL}</b> {TIMEFRAME}  <b>{side}</b>",
        f"가격: <b>{price:.2f}</b>",
        f"EMA10/20: {last['ema_s']:.2f} / {last['ema_l']:.2f}",
        f"EMA200: {last['ema_t']:.2f}",
        f"거래량: {last['volume']:.3f} (평균 {last['vol_ma']:.3f})",
        f"ATR({ATR_LEN}): {atr_v:.2f}",
        f"이유: {why}",
        f"<i>{dt}</i>"
    ]
    return "\n".join(msg)

def check_signals(df: pd.DataFrame):
    global _last_alert_ts_long, _last_alert_ts_short

    last = df.iloc[-1]
    atr_v = float(last["atr"]) if not math.isnan(last["atr"]) else 0.0

    # 추세 필터
    trend_ok_long  = (not USE_TREND) or (last["close"] > last["ema_t"])
    trend_ok_short = (not USE_TREND) or (last["close"] < last["ema_t"])

    # 거래량 필터
    vol_ok = (not NEED_VOL_BOOST) or (last["volume"] >= last["vol_ma"])

    # HVN 근접
    hvn_ok = True
    if REQUIRE_HVN_NEAR:
        peaks = volume_profile_hvn(df, bins=HVN_BINS, topk=HVN_PEAK_TOPK)
        hvn_ok = is_near_hvn(float(last["close"]), peaks, HVN_TOL_ATR, atr_v)

    # 교차
    long_cross  = cross_up(df["ema_s"], df["ema_l"])
    short_cross = cross_down(df["ema_s"], df["ema_l"])

    now = time.time()
    cd_sec = COOLDOWN_MIN * 60

    # 롱
    if long_cross and trend_ok_long and vol_ok and hvn_ok:
        if now - _last_alert_ts_long >= cd_sec:
            _last_alert_ts_long = now
            why = []
            why.append("EMA10 ↑ EMA20")
            if USE_TREND: why.append("가격 > EMA200")
            if NEED_VOL_BOOST: why.append("거래량 ≥ 평균")
            if REQUIRE_HVN_NEAR: why.append("HVN 근접")
            send_telegram(build_message("LONG", last, atr_v, ", ".join(why)))

    # 숏
    if short_cross and trend_ok_short and vol_ok and hvn_ok:
        if now - _last_alert_ts_short >= cd_sec:
            _last_alert_ts_short = now
            why = []
            why.append("EMA10 ↓ EMA20")
            if USE_TREND: why.append("가격 < EMA200")
            if NEED_VOL_BOOST: why.append("거래량 ≥ 평균")
            if REQUIRE_HVN_NEAR: why.append("HVN 근접")
            send_telegram(build_message("SHORT", last, atr_v, ", ".join(why)))

# ===================== 메인 루프 =====================
def main_loop():
    print(f"[bot] start: EXCHANGE={EXCHANGE_NAME}, SYMBOL={SYMBOL}, TF={TIMEFRAME}")
    while True:
        try:
            df = fetch_ohlcv(SYMBOL, TIMEFRAME, LIMIT)
            df = add_indicators(df).dropna()
            if len(df) >= max(EMA_TREND, VOL_MA_LEN, ATR_LEN) + 5:
                check_signals(df)
        except ccxt.BaseError as e:
            print("[ccxt] error:", e)
        except Exception as e:
            print("[loop] error:", e)
        time.sleep(POLL_SEC)

# ===================== Flask(Web Service) & 구니콘 진입 =====================
app = Flask(__name__)

@app.get("/")
def health():
    # 간단 상태 표시
    body = {
        "status": "ok",
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "exchange": EXCHANGE_NAME,
        "utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return (json.dumps(body), 200, {"Content-Type": "application/json"})

# gunicorn에서 import 시 바로 백그라운드 워커를 1회만 시작
_worker_started = False
def _start_worker_once():
    global _worker_started
    if not _worker_started:
        _worker_started = True
        Thread(target=main_loop, daemon=True).start()
        # 시작 알림이 필요하면 아래 주석 해제
        # send_telegram("🔔[투자봇] Render(Web Service)에서 시작")

_start_worker_once()

# 로컬에서 직접 실행할 때만(개발용)
if __name__ == "__main__":
    print("Starting bot with built-in Flask (dev)...")
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
# (기존 Flask 라우트 바로 아래 아무 데나)
@app.get("/test")
def test():
    send_telegram("✅ [투자봇] 텔레그램 연결 테스트 완료!")
    return "sent", 200
