import os, time, math, json, requests
from datetime import datetime, timezone
from threading import Thread

import numpy as np
import pandas as pd
import ccxt

# ===================== 환경 변수 =====================
SYMBOLS_STR   = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT")
SYMBOLS       = [s.strip() for s in SYMBOLS_STR.split(",") if s.strip()]
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").lower().strip()  # binance | binanceusdm | bybit
TIMEFRAME     = os.getenv("TIMEFRAME", "15m")
LIMIT         = int(os.getenv("LIMIT", "600"))
POLL_SEC      = int(os.getenv("POLL_SEC", "15"))  # 루프 대기

# 지표 파라미터
EMA_FAST  = int(os.getenv("EMA_FAST", "20"))   # 리테스트 대상
EMA_TREND = int(os.getenv("EMA_TREND", "200")) # 추세 필터
ATR_LEN   = int(os.getenv("ATR_LEN", "14"))

# 리테스트 규칙
RETEST_TOL_ATR = float(os.getenv("RETEST_TOL_ATR", "0.3"))  # |Close-EMA20| <= 0.3*ATR
RETEST_WINDOW  = int(os.getenv("RETEST_WINDOW", "30"))      # 추세 확립 후 대기 캔들 수

# SL/TP (A안: ATR 기반)
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))  # SL = entry ± 1.5*ATR
RR          = float(os.getenv("RR", "2.0"))           # TP = entry ± RR*(SL폭)

# 쿨다운(동일 심볼 동일 방향 재알림 최소 간격 / 분)
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "60"))

# 텔레그램
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")

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
    # 소수 둘째 자리까지
    return f"{p:.2f}%"

# ===================== 거래소/데이터 =====================
def build_exchange():
    if EXCHANGE_NAME == "bybit":
        return ccxt.bybit({"enableRateLimit": True})
    elif EXCHANGE_NAME in ("binanceusdm", "binance_futures", "binance_perp"):
        return ccxt.binanceusdm({"enableRateLimit": True})
    else:
        return ccxt.binance({"enableRateLimit": True})

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
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_trend"] = ema(df["close"], EMA_TREND)
    df["atr"] = atr(df, ATR_LEN)
    return df

# ===================== 상태/로직 =====================
# 심볼별 상태 (간단 dict)
state = {}
# 최근 알림 시간 (쿨다운)
_last_alert_ts = {}  # key: (symbol, side)

def reset_state_for(symbol: str, trend: str):
    # trend: "up" or "down"
    state[symbol] = {
        "trend": trend,
        "window_left": RETEST_WINDOW,
        "waiting_confirm": False,
        "retest_candle": None,  # dict: {high, low, close, ts}
    }

def ensure_state(symbol: str, trend_now: str):
    if symbol not in state:
        reset_state_for(symbol, trend_now)
    else:
        if state[symbol]["trend"] != trend_now:
            reset_state_for(symbol, trend_now)

def compute_levels(side: str, entry: float, atr_v: float):
    # ATR 기반 SL/TP 계산 + 가격/퍼센트 차이
    risk = ATR_SL_MULT * atr_v  # 가격 단위
    if side == "LONG":
        sl = entry - risk
        tp = entry + RR * risk
        sl_diff = entry - sl       # 양수
        tp_diff = tp - entry       # 양수
    else:
        sl = entry + risk
        tp = entry - RR * risk
        sl_diff = sl - entry
        tp_diff = entry - tp
    sl_pct = (sl_diff / entry) * 100.0
    tp_pct = (tp_diff / entry) * 100.0
    return round(sl, 2), round(tp, 2), round(sl_diff, 2), round(tp_diff, 2), sl_pct, tp_pct

def build_message(sym: str, side: str, entry: float, sl: float, tp: float,
                  sl_diff: float, tp_diff: float, sl_pct: float, tp_pct: float,
                  last: pd.Series, rc_info: dict):
    dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tol_ratio = abs(last["close"] - last["ema_fast"]) / max(last["atr"], 1e-9) if not math.isnan(last["atr"]) else float("nan")

    lines = [
        f"🔔 <b>{sym}</b> {TIMEFRAME} <b>{side}</b> (확인형 EMA{EMA_FAST} 리테스트 확정)",
        f"Entry: <b>{fmt_price(entry)}</b> USDT",
        f"SL: {fmt_price(sl)} USDT  (-{fmt_price(sl_diff)}, -{fmt_pct(sl_pct)})",
        f"TP: {fmt_price(tp)} USDT  (+{fmt_price(tp_diff)}, +{fmt_pct(tp_pct)})",
        "",
        f"조건: EMA{EMA_TREND} {'상단' if side=='LONG' else '하단'}, EMA{EMA_FAST} 근접(≈ {tol_ratio:.2f}×ATR) 후 다음봉 {'고가' if side=='LONG' else '저가'} 돌파 마감",
        f"현재가: {fmt_price(last['close'])}, EMA{EMA_FAST}: {fmt_price(last['ema_fast'])}, EMA{EMA_TREND}: {fmt_price(last['ema_trend'])}, ATR{ATR_LEN}: {fmt_price(last['atr'])}",
        f"리테스트 봉 (UTC): H={fmt_price(rc_info['high'])}, L={fmt_price(rc_info['low'])}, C={fmt_price(rc_info['close'])}",
        f"<i>{dt}</i>",
    ]
    return "\n".join(lines)

def maybe_alert(sym: str, side: str, entry: float, df: pd.DataFrame, last: pd.Series, rc_info: dict):
    # 쿨다운 체크
    key = (sym, side)
    now = time.time()
    if key not in _last_alert_ts:
        _last_alert_ts[key] = 0.0
    if now - _last_alert_ts[key] < COOLDOWN_MIN * 60:
        return

    atr_v = float(last["atr"]) if not math.isnan(last["atr"]) else 0.0
    sl, tp, sl_diff, tp_diff, sl_pct, tp_pct = compute_levels(side, entry, atr_v)
    msg = build_message(sym, side, entry, sl, tp, sl_diff, tp_diff, sl_pct, tp_pct, last, rc_info)
    send_telegram(msg)
    _last_alert_ts[key] = now

def process_symbol(sym: str):
    df = fetch_ohlcv(sym, TIMEFRAME, LIMIT)
    df = add_indicators(df).dropna()
    if len(df) < max(EMA_TREND, ATR_LEN) + 5:
        return

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 1) 추세판단
    trend_now = "up" if last["close"] > last["ema_trend"] else "down"
    ensure_state(sym, trend_now)

    st = state[sym]

    # 추세 바뀌면 대기 초기화는 ensure_state에서 처리됨.
    # 2) 리테스트 감지 (대기 중 & 윈도우 남아있을 때 & 아직 확인대기 아님)
    if not st["waiting_confirm"] and st["window_left"] > 0:
        # EMA20 근접: |Close - EMA20| <= RETEST_TOL_ATR * ATR
        near = False
        atr_v = float(last["atr"]) if not math.isnan(last["atr"]) else 0.0
        if atr_v > 0:
            near = abs(float(last["close"]) - float(last["ema_fast"])) <= (RETEST_TOL_ATR * atr_v)

        if near:
            st["retest_candle"] = {
                "ts": int(last["ts"]),
                "high": float(last["high"]),
                "low": float(last["low"]),
                "close": float(last["close"]),
            }
            st["waiting_confirm"] = True

        # 윈도우 감소
        st["window_left"] -= 1

    # 3) 확인(다음 봉에서 돌파 마감) & 알림
    if st["waiting_confirm"] and st["retest_candle"] is not None:
        rc = st["retest_candle"]
        # 추세 무효화 체크: 확인 전에 추세 깨지면 리셋
        if trend_now == "up" and last["close"] < last["ema_trend"]:
            reset_state_for(sym, "down")
            return
        if trend_now == "down" and last["close"] > last["ema_trend"]:
            reset_state_for(sym, "up")
            return

        # 확인 조건
        if trend_now == "up":
            confirmed = float(last["close"]) >= rc["high"]
            if confirmed:
                entry = float(last["close"])  # 확인봉 종가
                maybe_alert(sym, "LONG", entry, df, last, rc)
                reset_state_for(sym, "up")
                return
        else:
            confirmed = float(last["close"]) <= rc["low"]
            if confirmed:
                entry = float(last["close"])
                maybe_alert(sym, "SHORT", entry, df, last, rc)
                reset_state_for(sym, "down")
                return

# ===================== 메인 루프 =====================
def main_loop():
    print(f"[bot] start: EXCHANGE={EXCHANGE_NAME}, SYMBOLS={SYMBOLS}, TF={TIMEFRAME}")
    # 초기 상태 세팅
    for sym in SYMBOLS:
        try:
            df0 = fetch_ohlcv(sym, TIMEFRAME, limit= max(EMA_TREND, ATR_LEN) + 5)
            df0 = add_indicators(df0).dropna()
            if len(df0) == 0:
                trend0 = "up"
            else:
                trend0 = "up" if df0.iloc[-1]["close"] > df0.iloc[-1]["ema_trend"] else "down"
            reset_state_for(sym, trend0)
        except Exception as e:
            print(f"[init:{sym}] error:", e)
            reset_state_for(sym, "up")

    while True:
        try:
            for sym in SYMBOLS:
                try:
                    process_symbol(sym)
                    time.sleep(0.4)
                except ccxt.BaseError as e:
                    print(f"[ccxt:{sym}] error:", e)
                except Exception as e:
                    print(f"[loop:{sym}] error:", e)
        except Exception as e:
            print("[loop] outer error:", e)
        time.sleep(POLL_SEC)

# ===================== Flask (파일 맨 아래 - 방법 A) =====================
from flask import Flask
app = Flask(__name__)

@app.get("/")
def health():
    body = {
        "status": "ok",
        "exchange": EXCHANGE_NAME,
        "symbols": SYMBOLS,
        "timeframe": TIMEFRAME,
        "utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "EMA_FAST": EMA_FAST,
            "EMA_TREND": EMA_TREND,
            "ATR_LEN": ATR_LEN,
            "RETEST_TOL_ATR": RETEST_TOL_ATR,
            "RETEST_WINDOW": RETEST_WINDOW,
            "ATR_SL_MULT": ATR_SL_MULT,
            "RR": RR,
            "COOLDOWN_MIN": COOLDOWN_MIN,
        }
    }
    return (json.dumps(body), 200, {"Content-Type": "application/json"})

@app.get("/test")
def test():
    send_telegram("✅ [투자봇] 텔레그램 연결 테스트")
    return "sent", 200

# gunicorn import 시 1회만 백그라운드 워커 시작
_worker_started = False
def _start_worker_once():
    global _worker_started
    if not _worker_started:
        _worker_started = True
        Thread(target=main_loop, daemon=True).start()
        # 원하면 시작 알림:
        # send_telegram("🔔[투자봇] Render(Web Service)에서 시작")

_start_worker_once()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    print("Starting dev Flask ...")
    app.run(host="0.0.0.0", port=port)
