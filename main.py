# main.py
import os, time, math, json, random, requests
from datetime import datetime, timezone
from threading import Thread

import numpy as np
import pandas as pd
import ccxt
from flask import Flask

# ===================== (선택) 로컬 .env 로드 =====================
if not os.getenv("RENDER"):
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass

# ===================== 환경 변수 =====================
EXCHANGE = os.getenv("EXCHANGE", "bybit").strip().lower()
SYMBOLS  = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",") if s.strip()]
TIMEFRAME = os.getenv("TIMEFRAME", "15m")

# 주기/레이트리밋 보호
POLL_SEC      = int(os.getenv("POLL_SEC", "15"))   # 메인 루프 슬립(초)
FETCH_MIN_SEC = int(os.getenv("FETCH_MIN_SEC", "55")) # 같은 심볼 재호출 최소 간격(초)
BACKOFF_START = int(os.getenv("BACKOFF_START", "60"))
BACKOFF_MAX   = int(os.getenv("BACKOFF_MAX", "600"))
JITTER_MAX    = int(os.getenv("JITTER_MAX", "3"))

# 지표 파라미터
EMA_FAST   = int(os.getenv("EMA_FAST", "20"))   # 리테스트 대상 EMA
EMA_TREND  = int(os.getenv("EMA_TREND", "200")) # 추세 필터 EMA
ATR_LEN    = int(os.getenv("ATR_LEN", "14"))

# 리테스트 규칙(A안: 확인형)
RETEST_TOL_ATR = float(os.getenv("RETEST_TOL_ATR", "0.3"))  # |Close-EMA20| <= 0.3*ATR
RETEST_WINDOW  = int(os.getenv("RETEST_WINDOW", "30"))      # 200EMA 돌파 후 ~N캔들 내 리테스트 유효

# SL/TP 기본 계수(최종은 연속형 RR 추천으로 대체될 수 있음)
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))  # SL = entry ± 1.5*ATR

# 쿨다운(동일 심볼/방향)
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "60"))  # 분

# 상위 TF 장벽 계산용
HTF_FOR_BARRIERS = os.getenv("HTF_FOR_BARRIERS", "1h")
HTF_EMAS = [20, 200]

# 텔레그램
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")

# ===================== Flask 먼저 생성 (데코레이터 오류 방지) =====================
app = Flask(__name__)

# ===================== 유틸/알림 =====================
def send_telegram(msg: str):
    if not TG_TOKEN or not TG_CHAT:
        print("[WARN] Telegram not configured.")
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

def fmt_pct(pct: float) -> str:
    return f"{pct:.2f}%"

# ===================== 거래소 =====================
def build_exchange():
    name = EXCHANGE
    if name == "bybit":
        return ccxt.bybit({"enableRateLimit": True})
    elif name in ("binanceusdm", "binance_futures", "binance_perp"):
        return ccxt.binanceusdm({"enableRateLimit": True})
    else:
        return ccxt.binance({"enableRateLimit": True})

ex = build_exchange()

# ===================== 안전 fetch 래퍼/캐시 =====================
_last_fetch = {}     # key: (symbol, timeframe) -> epoch
_df_cache   = {}     # key: (symbol, timeframe) -> df
_backoff    = 0

def fetch_ohlcv_throttled(symbol, timeframe, limit=400):
    """레이트리밋 안전 호출 + 캐시 반환"""
    global _backoff
    key = (symbol, timeframe)
    now = time.time()
    last_t = _last_fetch.get(key, 0)

    if now - last_t < FETCH_MIN_SEC and key in _df_cache:
        return _df_cache[key]

    if JITTER_MAX > 0:
        time.sleep(random.uniform(0, JITTER_MAX))

    if _backoff > 0:
        time.sleep(_backoff)

    try:
        ohlcv = ex.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("UTC")
        _df_cache[key] = df
        _last_fetch[key] = time.time()
        _backoff = max(0, int(_backoff/2))
        return df
    except ccxt.BaseError as e:
        msg = str(e)
        if ("429" in msg) or ("418" in msg) or ("Too much request" in msg) or ("Way too much" in msg):
            _backoff = BACKOFF_START if _backoff == 0 else min(BACKOFF_MAX, _backoff*2)
            print(f"[rate-limit] backoff={_backoff}s {symbol} {timeframe} :: {msg[:120]}")
        else:
            print(f"[ccxt:{symbol}] {msg[:200]}")
        return _df_cache.get(key, pd.DataFrame(columns=["ts","open","high","low","close","volume","dt"]))

# ===================== 지표 =====================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def macd(df: pd.DataFrame, fast=12, slow=26, sig=9):
    ema_fast = ema(df["close"], fast)
    ema_slow = ema(df["close"], slow)
    macd_line = ema_fast - ema_slow
    signal = ema(macd_line, sig)
    hist = macd_line - signal
    return macd_line, signal, hist

def add_indicators(df: pd.DataFrame):
    if df.empty: return df
    df["ema_fast"]  = ema(df["close"], EMA_FAST)
    df["ema_trend"] = ema(df["close"], EMA_TREND)
    df["atr"]       = atr(df, ATR_LEN)
    macd_line, macd_sig, macd_hist = macd(df)
    df["macd"] = macd_line
    df["macd_sig"] = macd_sig
    df["macd_hist"] = macd_hist
    # 거래량 대비 등급용
    df["vol_ma50"] = df["volume"].rolling(50).mean()
    df["vol_ma200"] = df["volume"].rolling(200).mean()
    return df

# ===================== 장벽(저항/지지) 탐지 =====================
def pivot_levels(df: pd.DataFrame, lookback=20):
    """간단 pivot high/low 추출"""
    if len(df) < lookback+2:
        return [], []
    highs = []
    lows = []
    h = df["high"].values
    l = df["low"].values
    for i in range(lookback, len(df)-lookback):
        if h[i] == max(h[i-lookback:i+lookback+1]):
            highs.append(h[i])
        if l[i] == min(l[i-lookback:i+lookback+1]):
            lows.append(l[i])
    return highs[-10:], lows[-10:]  # 최근 것 위주

def round_levels_near(price: float, step: float = None, count: int = 6):
    """반올림 숫자대: 스텝 자동 추정(가격 크기에 비례)"""
    if step is None:
        # 대략적 스텝 추정
        if price >= 50000: step = 1000
        elif price >= 10000: step = 500
        elif price >= 1000: step = 100
        elif price >= 100: step = 10
        elif price >= 10: step = 1
        else: step = 0.1
    base = round(price/step)*step
    return [base + k*step for k in range(-count//2, count//2+1)]

def adr_levels(df: pd.DataFrame, multiples=(1.0, 1.5, 2.0)):
    """최근 N봉 평균 TR(=ATR 대용) 기반 레벨"""
    if df.empty: return []
    tr = atr(df, ATR_LEN)
    if tr.empty or np.isnan(tr.iloc[-1]): return []
    last = df.iloc[-1]
    last_close = float(last["close"])
    trv = float(tr.iloc[-1])
    lvls = []
    for m in multiples:
        lvls += [last_close + m*trv, last_close - m*trv]
    return lvls

def hvn_levels(df: pd.DataFrame, bins=24):
    """간이 HVN: 최근 구간을 가격빈으로 나눠 빈도 최상위 bin 중심"""
    if len(df) < 200: return []
    closes = df["close"].tail(600).values  # 최근 ~600틱
    lo, hi = closes.min(), closes.max()
    if hi <= lo: return []
    hist, edges = np.histogram(closes, bins=bins, range=(lo, hi))
    idxs = np.argsort(hist)[-3:]  # 상위 3 bin
    centers = []
    for i in idxs:
        centers.append((edges[i] + edges[i+1])/2)
    return sorted(centers)

def htf_ema_levels(symbol: str, emalens=HTF_EMAS):
    """상위 TF EMA 레벨 수집 (레이트리밋 보호!)"""
    df = fetch_ohlcv_throttled(symbol, HTF_FOR_BARRIERS, limit=max(emalens)+50)
    if df.empty: return []
    df = add_indicators(df)
    lvls = []
    for ln in emalens:
        s = ema(df["close"], ln)
        if not s.empty and not np.isnan(s.iloc[-1]):
            lvls.append(float(s.iloc[-1]))
    return lvls

def nearest_barriers(df15: pd.DataFrame, symbol: str, price: float):
    """상·하 방향 각각 가장 가까운 장벽들과 거리"""
    highs, lows = pivot_levels(df15, lookback=20)
    rounds = round_levels_near(price)
    adrs   = adr_levels(df15)
    hvns   = hvn_levels(df15)
    htf    = htf_ema_levels(symbol)

    all_lvls = {
        "pivot_high": highs,
        "pivot_low": lows,
        "round": rounds,
        "adr": adrs,
        "h_ema": htf
    }
    # 위/아래 분리
    up, down = [], []
    for key, arr in all_lvls.items():
        for lv in arr:
            if lv >= price: up.append(("UP", key, float(lv), float(lv - price)))
            if lv <= price: down.append(("DN", key, float(lv), float(price - lv)))
    up  = sorted(up, key=lambda x: x[3])[:5]
    down= sorted(down, key=lambda x: x[3])[:5]
    return up, down

# ===================== 신호 로직 =====================
# 심볼 상태(리테스트용)
state = {}  # symbol -> {trend, window_left, waiting_confirm, retest_candle}
_last_alert_ts = {}  # (symbol, side, tag) -> epoch

def ensure_state(symbol: str, trend_now: str):
    st = state.get(symbol)
    if (st is None) or (st["trend"] != trend_now):
        state[symbol] = {
            "trend": trend_now,
            "window_left": RETEST_WINDOW,
            "waiting_confirm": False,
            "retest_candle": None
        }

def rr_recommendation(side: str, atr_v: float, up_barriers, down_barriers,
                      htf_align_score: float, volume_tier: str, macd_momentum: float):
    """
    연속형 RR 추천 (1.3~2.5 범위) : 스코어 기반
    - 장벽 거리 넉넉/순풍/모멘텀 강 → RR↑
    - 장벽 가깝/역풍/모멘텀 약 → RR↓
    """
    if atr_v <= 0 or math.isnan(atr_v):
        atr_v = 1.0
    # 장벽 거리 점수(익절 방향이 더 멀수록 가점)
    if side == "LONG":
        up_dist  = up_barriers[0][3] if up_barriers else atr_v
        dn_dist  = down_barriers[0][3] if down_barriers else atr_v
    else:
        up_dist  = down_barriers[0][3] if down_barriers else atr_v
        dn_dist  = up_barriers[0][3] if up_barriers else atr_v
    # ATR 대비 거리 비율
    tp_room = up_dist/atr_v
    sl_wall = dn_dist/atr_v

    score = 0.0
    # 익절 공간
    score += np.clip(tp_room, 0, 3) * 0.7
    # 손절까지 완충
    score += np.clip(sl_wall, 0, 3) * 0.3
    # 상위TF 정렬
    score += htf_align_score * 0.6
    # 거래량 등급
    vol_map = {"A": 0.6, "B": 0.3, "C": 0.0}
    score += vol_map.get(volume_tier, 0.0)
    # MACD 모멘텀
    score += np.clip(macd_momentum, -2, 2) * 0.2

    # 점수→RR 매핑 (대략 1.3 ~ 2.5 범위)
    rr = 1.3 + (np.clip(score, 0, 5) / 5.0) * (2.5 - 1.3)
    return float(round(rr, 2))

def compute_levels(side: str, entry: float, atr_v: float, rr: float):
    risk = ATR_SL_MULT * atr_v
    if side == "LONG":
        sl = entry - risk
        tp = entry + rr * risk
        sl_diff = entry - sl
        tp_diff = tp - entry
    else:
        sl = entry + risk
        tp = entry - rr * risk
        sl_diff = sl - entry
        tp_diff = entry - tp
    sl_pct = (sl_diff/entry)*100.0 if entry else 0.0
    tp_pct = (tp_diff/entry)*100.0 if entry else 0.0
    # 소수 둘째자리 표기
    return round(sl,2), round(tp,2), round(sl_diff,2), round(tp_diff,2), round(sl_pct,2), round(tp_pct,2)

def volume_tier_of(df_last: pd.Series):
    """A/B/C 등급"""
    v  = float(df_last["volume"])
    v50 = float(df_last.get("vol_ma50", np.nan))
    v200= float(df_last.get("vol_ma200", np.nan))
    ratio = 0.0
    if v50 and not math.isnan(v50) and v50>0:
        ratio = v / v50
    # 느슨 등급
    if ratio >= 1.5: return "A"
    if ratio >= 1.1: return "B"
    return "C"

def htf_align(df15_last: pd.Series):
    """상위TF 정렬 간단 점수: 15m close vs 200EMA, macd_hist 기초"""
    score = 0.0
    # 15m 기준 추세(200EMA)
    if float(df15_last["close"]) > float(df15_last["ema_trend"]): score += 0.5
    else: score -= 0.2
    # 15m macd_hist
    mh = float(df15_last.get("macd_hist", 0.0))
    score += np.clip(mh, -1, 1) * 0.5
    return float(np.clip(score, -1, 1))

def golden_cross_signal(df: pd.DataFrame):
    """EMA10/EMA20 + MACD 히스토그램 기준 골든/데드 크로스 탐지"""
    if len(df) < 30: return None
    ema10 = ema(df["close"], 10)
    ema20 = ema(df["close"], 20)
    macd_line, sig, hist = macd(df)
    cross_up = (ema10.iloc[-2] <= ema20.iloc[-2]) and (ema10.iloc[-1] > ema20.iloc[-1])
    cross_dn = (ema10.iloc[-2] >= ema20.iloc[-2]) and (ema10.iloc[-1] < ema20.iloc[-1])
    # 모멘텀 동행
    if cross_up and hist.iloc[-1] > 0:
        return "GOLDEN"
    if cross_dn and hist.iloc[-1] < 0:
        return "DEAD"
    return None

def maybe_alert(sym: str, side: str, tag: str, entry1: float, entry2: float,
                rr: float, last: pd.Series, rc_info: dict,
                up_barriers, down_barriers):
    # 쿨다운
    key = (sym, side, tag)
    now = time.time()
    if now - _last_alert_ts.get(key, 0.0) < COOLDOWN_MIN*60:
        return
    atr_v = float(last.get("atr", 0.0))
    sl1,tp1,sl1d,tp1d,sl1p,tp1p = compute_levels(side, entry1, atr_v, rr)
    sl2,tp2,sl2d,tp2d,sl2p,tp2p = compute_levels(side, entry2, atr_v, rr)

    # 장벽 요약(최단 거리 1~2개)
    def barrier_lines(arr, prefix):
        lines=[]
        for i, itm in enumerate(arr[:2], start=1):
            _dir, typ, lvl, dist = itm
            lines.append(f"{prefix}{i}. {typ.upper()} @ {fmt_price(lvl)} (Δ {fmt_price(dist)})")
        return lines

    vtier = volume_tier_of(last)
    mh = float(last.get("macd_hist", 0.0))

    lines = [
        f"🔔 <b>{sym}</b> {TIMEFRAME} <b>{side}</b> [{tag}]",
        f"추천 RR: <b>{rr:.2f}</b>",
        "",
        f"① <u>확정 진입</u> (확인봉 종가): {fmt_price(entry1)}",
        f"   SL {fmt_price(sl1)} (-{fmt_price(sl1d)}, -{fmt_pct(sl1p)}) / TP {fmt_price(tp1)} (+{fmt_price(tp1d)}, +{fmt_pct(tp1p)})",
        "",
        f"② <u>되돌림 진입</u> (EMA{EMA_FAST} 근처): {fmt_price(entry2)}",
        f"   SL {fmt_price(sl2)} (-{fmt_price(sl2d)}, -{fmt_pct(sl2p)}) / TP {fmt_price(tp2)} (+{fmt_price(tp2d)}, +{fmt_pct(tp2p)})",
        "",
        f"장벽↑(익절방향):"] + barrier_lines(up_barriers, "• ") + [
        f"장벽↓(손절방향):"] + barrier_lines(down_barriers, "• ") + [
        "",
        f"컨텍스트: Vol tier={vtier}, MACD_hist={mh:.3f}, ATR{ATR_LEN}={fmt_price(float(last.get('atr',0.0)))}",
        f"<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</i>"
    ]
    send_telegram("\n".join(lines))
    _last_alert_ts[key] = now

def process_symbol(sym: str):
    df = fetch_ohlcv_throttled(sym, TIMEFRAME, limit=400)
    if df.empty: return
    df = add_indicators(df).dropna()
    if len(df) < max(EMA_TREND, ATR_LEN) + 10: return
    last = df.iloc[-1]

    # 추세확인 (리테스트 흐름)
    trend_now = "up" if last["close"] > last["ema_trend"] else "down"
    ensure_state(sym, trend_now)
    st = state[sym]

    # 골든/데드 크로스 체크 (보조 시그널)
    gc = golden_cross_signal(df)  # "GOLDEN" or "DEAD" or None

    # 리테스트: EMA_FAST 근접 → 다음봉 확인
    atr_v = float(last["atr"]) if not np.isnan(last["atr"]) else 0.0
    near = False
    if atr_v > 0:
        near = abs(float(last["close"]) - float(last["ema_fast"])) <= (RETEST_TOL_ATR * atr_v)

    # 200EMA 상단(up)일 때만 롱 리테스트 후보, 하단(down)일 때만 숏
    if not st["waiting_confirm"] and st["window_left"] > 0 and near:
        st["retest_candle"] = {
            "ts": int(last["ts"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "close": float(last["close"]),
            "ema_fast": float(last["ema_fast"])
        }
        st["waiting_confirm"] = True

    # 윈도우 진행
    if st["window_left"] > 0:
        st["window_left"] -= 1

    # 확인 단계(이번 봉에서 리테스트 봉의 고/저 돌파 마감)
    if st["waiting_confirm"] and st["retest_candle"] is not None:
        rc = st["retest_candle"]
        # 추세 무효화 시 리셋
        if trend_now == "up" and last["close"] < last["ema_trend"]:
            ensure_state(sym, "down")
            return
        if trend_now == "down" and last["close"] > last["ema_trend"]:
            ensure_state(sym, "up")
            return

        confirmed = False
        side = None
        if trend_now == "up":
            confirmed = float(last["close"]) >= rc["high"]
            side = "LONG"
        else:
            confirmed = float(last["close"]) <= rc["low"]
            side = "SHORT"

        if confirmed:
            # 진입가 2가지
            entry_confirm = float(last["close"])
            entry_pullbk  = float(last.get("ema_fast", rc["ema_fast"]))  # 되돌림 목표

            # 장벽 계산(15m 기준 + HTF EMA)
            up_b, dn_b = nearest_barriers(df, sym, float(last["close"]))

            # 컨텍스트 점수/등급
            vtier = volume_tier_of(last)
            hscore = htf_align(last)
            mh = float(last.get("macd_hist", 0.0))
            # 연속형 RR 추천
            rr = rr_recommendation(side, atr_v, up_b, dn_b, hscore, vtier, mh)

            tag = "리테스트"
            # 보조: 골든/데드가 같이 방금 발생했다면 태그에 표기
            if (side=="LONG" and gc=="GOLDEN") or (side=="SHORT" and gc=="DEAD"):
                tag += "+크로스동행"

            maybe_alert(sym, side, tag, entry_confirm, entry_pullbk, rr, last, rc, up_b, dn_b)
            # 상태 리셋(동일 추세 지속)
            ensure_state(sym, trend_now)
            return

    # 별도: 골든/데드 크로스 자체 시그널 (거래량 등급 느슨 적용 + ATR 환경 보조)
    if gc in ("GOLDEN", "DEAD"):
        side = "LONG" if gc=="GOLDEN" else "SHORT"
        # 추세 반대면 신뢰 낮음 → 그대로 알리되 RR 낮게 갈 것
        entry_confirm = float(last["close"])
        entry_pullbk  = float(last["ema_fast"])
        up_b, dn_b = nearest_barriers(df, sym, entry_confirm)
        vtier = volume_tier_of(last)
        hscore = htf_align(last)
        mh = float(last.get("macd_hist", 0.0))
        rr = rr_recommendation(side, float(last.get("atr",0.0)), up_b, dn_b, hscore, vtier, mh)
        maybe_alert(sym, side, "골든크로스" if side=="LONG" else "데드크로스",
                    entry_confirm, entry_pullbk, rr, last, {"high": last["high"],"low": last["low"],"close": last["close"]},
                    up_b, dn_b)

# ===================== 메인 루프 =====================
def main_loop():
    print(f"[bot] start: EXCHANGE={EXCHANGE}, SYMBOLS={SYMBOLS}, TF={TIMEFRAME}")
    # 초기 상태 세팅
    for sym in SYMBOLS:
        try:
            df0 = fetch_ohlcv_throttled(sym, TIMEFRAME, limit=max(EMA_TREND, ATR_LEN)+20)
            df0 = add_indicators(df0).dropna()
            if len(df0)==0:
                trend0 = "up"
            else:
                trend0 = "up" if df0.iloc[-1]["close"] > df0.iloc[-1]["ema_trend"] else "down"
            ensure_state(sym, trend0)
        except Exception as e:
            print(f"[init:{sym}] {str(e)[:200]}")
            ensure_state(sym, "up")

    while True:
        try:
            for sym in SYMBOLS:
                try:
                    process_symbol(sym)
                    time.sleep(0.4)  # 심볼 간 소폭 쉬기
                except Exception as e:
                    print(f"[loop:{sym}] {str(e)[:200]}")
        except Exception as e:
            print("[loop:outer]", str(e)[:200])
        time.sleep(POLL_SEC)

# ===================== Flask 라우트 =====================
@app.get("/")
def health():
    body = {
        "status": "ok",
        "exchange": EXCHANGE,
        "symbols": SYMBOLS,
        "timeframe": TIMEFRAME,
        "utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "EMA_FAST": EMA_FAST, "EMA_TREND": EMA_TREND, "ATR_LEN": ATR_LEN,
            "RETEST_TOL_ATR": RETEST_TOL_ATR, "RETEST_WINDOW": RETEST_WINDOW,
            "ATR_SL_MULT": ATR_SL_MULT, "COOLDOWN_MIN": COOLDOWN_MIN,
            "FETCH_MIN_SEC": FETCH_MIN_SEC, "BACKOFF_START": BACKOFF_START,
            "BACKOFF_MAX": BACKOFF_MAX, "JITTER_MAX": JITTER_MAX, "POLL_SEC": POLL_SEC,
            "HTF": HTF_FOR_BARRIERS, "HTF_EMAS": HTF_EMAS
        }
    }
    return (json.dumps(body), 200, {"Content-Type": "application/json"})

@app.get("/test")
def test():
    send_telegram("✅ [투자봇] 텔레그램 연결 테스트")
    return "sent", 200

# ===================== gunicorn 임포트 시 1회 워커 기동 =====================
_worker_started = False
def _start_worker_once():
    global _worker_started
    if not _worker_started:
        _worker_started = True
        Thread(target=main_loop, daemon=True).start()

_start_worker_once()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    print("Starting dev Flask ...")
    app.run(host="0.0.0.0", port=port)
