# main.py
# Live Signal Notifier — BTC / SOL / XRP (15m)
# - ENTRY: BB breakout + ENG(+PIN) + MACD slope + RSI + Volume + Trend(EMA50>200)
# - EXIT S3: TP1 +1.0R → 70% take → SL to BE → remainder trails by BB mid
# - Alerts (모바일 포함 동일 포맷): 풀버전 텍스트 + 2줄 사유 / 진행이벤트
# - State persists to ./state_positions.json (포지션/SL/TP1 여부)

import os, json, time, math, statistics, traceback
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import ccxt
import requests

# =========================
# Config
# =========================
CFG = {
    "EXCHANGE": os.getenv("EXCHANGE", "okx"),
    "SYMBOLS": ["BTC/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT"],
    "TIMEFRAME": "15m",
    "HTF_TIMEFRAME": "1h",
    "FETCH_LIMIT": 1200,       # 최근 약 12~13일 (15m)
    "SQUEEZE_LOOKBACK": 600,   # pctl 계산 구간
    "EMA_FAST": 50, "EMA_SLOW": 200,
    "BB_PERIOD": 20, "BB_STD": 2.0,
    "ATR_PERIOD": 14,
    "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
    "RSI_PERIOD": 14,
    "INTRABAR_ORDER": "conservative",  # TP/SL 동시 충족 시 SL 우선
    "SL_METHOD": "ATR",
    # S3 partial
    "TP1_R": 1.0, "TP1_WEIGHT": 0.7,   # 70% 익절
    # Notifications
    "STAR_SYMBOLS": {"XRP/USDT:USDT", "SOL/USDT:USDT"},  # 헤더에 ⭐
    "SEND_INTERVAL_SEC": 20,    # 폴링 (권장 20~30초)
    "STATE_FILE": "./state_positions.json",
    # Notifiers (텔레그램/디스코드 둘다 비어있으면 콘솔로만 출력)
    "TELEGRAM_BOT_TOKEN": os.getenv("TG_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID": os.getenv("TG_CHAT_ID", ""),
    "DISCORD_WEBHOOK": os.getenv("DISCORD_WEBHOOK", ""),
}

# 심볼별 오버라이드 (백테스트 좋은 설정 복원)
CFG_SYMBOL = {
    # BTC: 중립(거래수 살짝↑)
    "BTC/USDT:USDT": {
        "SQUEEZE_PCTL": 35, "ATR_MULT": 1.5,
        "CONFIRM_MODE": "ENG+PIN", "COOLDOWN_BARS": 15,
        "USE_HTF": True, "USE_VOL_FILTER": True, "VOL_BOOST": 1.2,
        "USE_RSI_FILTER": True
    },
    # SOL: 보수(휩소 강함)
    "SOL/USDT:USDT": {
        "SQUEEZE_PCTL": 20, "ATR_MULT": 2.0,
        "CONFIRM_MODE": "ENG", "COOLDOWN_BARS": 40,
        "USE_HTF": True, "USE_VOL_FILTER": True, "VOL_BOOST": 1.5,
        "USE_RSI_FILTER": True
    },
    # XRP: 에이스(신호수↑)
    "XRP/USDT:USDT": {
        "SQUEEZE_PCTL": 35, "ATR_MULT": 1.5,
        "CONFIRM_MODE": "ENG+PIN", "COOLDOWN_BARS": 15,
        "USE_HTF": True, "USE_VOL_FILTER": True, "VOL_BOOST": 1.1,
        "USE_RSI_FILTER": True
    },
}

# =========================
# Notifier
# =========================
class Notifier:
    def __init__(self, tg_token: str, tg_chat: str, discord_hook: str):
        self.tg_token = tg_token
        self.tg_chat = tg_chat
        self.discord_hook = discord_hook

    def send(self, text: str):
        sent = False
        if self.tg_token and self.tg_chat:
            try:
                url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
                payload = {"chat_id": self.tg_chat, "text": text}
                requests.post(url, json=payload, timeout=10)
                sent = True
            except Exception:
                print("[telegram error]", traceback.format_exc())
        if self.discord_hook:
            try:
                requests.post(self.discord_hook, json={"content": text}, timeout=10)
                sent = True
            except Exception:
                print("[discord error]", traceback.format_exc())
        if not sent:
            print(text)

notifier = Notifier(CFG["TELEGRAM_BOT_TOKEN"], CFG["TELEGRAM_CHAT_ID"], CFG["DISCORD_WEBHOOK"])

# =========================
# Helpers & Indicators
# =========================
def tz_kst(ts_utc: pd.Timestamp) -> str:
    if ts_utc.tzinfo is None: ts_utc = ts_utc.tz_localize(timezone.utc)
    return (ts_utc.astimezone(timezone(timedelta(hours=9)))).strftime("%Y-%m-%d %H:%M")

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def macd(s: pd.Series, f=12, sl=26, sig=9):
    line = ema(s, f) - ema(s, sl)
    signal = ema(line, sig)
    return line, signal, line - signal

def bollinger(close: pd.Series, n=20, k=2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    return mid + k*std, mid, mid - k*std

def rsi(series: pd.Series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(n).mean()
    roll_down = down.rolling(n).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    cp = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - cp).abs()
    c = (df["low"] - cp).abs()
    return pd.concat([a,b,c], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return true_range(df).rolling(n).mean()

def percent_rank(arr: np.ndarray, v: float) -> float:
    if len(arr) == 0 or np.isnan(v): return np.nan
    return float((arr <= v).sum()) / len(arr) * 100.0

def detect_bull_engulf(po, pc, co, cc):
    return (pc < po) and (cc > co) and (cc >= po) and (co <= pc)

def detect_bear_engulf(po, pc, co, cc):
    return (pc > po) and (cc < co) and (cc <= po) and (co >= pc)

def detect_pinbar(o,h,l,c, bullish=True, body_maxr=0.35, tail_minr=0.6):
    rng = h - l
    if rng <= 0: return False
    body = abs(c - o)
    if body > rng * body_maxr: return False
    upper = h - max(o, c)
    lower = min(o, c) - l
    return (lower >= rng*tail_minr) if bullish else (upper >= rng*tail_minr)

def breakout_ok(row, side):
    return row.close > row.bb_upper if side=="long" else row.close < row.bb_lower

def squeeze_ok(row, th):
    return (not math.isnan(row.bb_pctl)) and (row.bb_pctl <= th)

def macd_ok(prev, curr, side):
    return (curr.macd_hist > 0 and curr.macd_hist > prev.macd_hist) if side=="long" else \
           (curr.macd_hist < 0 and curr.macd_hist < prev.macd_hist)

def rsi_ok(curr, side, use=True):
    if not use: return True
    return curr.rsi > 50 if side=="long" else curr.rsi < 50

def vol_ok(curr, use, boost, med):
    if not use: return True
    if math.isnan(med) or med <= 0: return True
    return curr.volume >= med * boost

def trend_flags(row):
    return row.ema_fast > row.ema_slow, row.ema_fast < row.ema_slow

def get_sl(entry, row_prev, df, i_prev, is_long, method, atr_mult):
    if method.upper() == "ATR":
        a = float(row_prev.atr)
        if np.isnan(a) or a <= 0:
            # fallback: swing
            seg = df.iloc[max(0, i_prev-20):i_prev+1]
            return float(seg.low.min()) if is_long else float(seg.high.max())
        return entry - atr_mult*a if is_long else entry + atr_mult*a
    # fallback
    return float(row_prev.bb_mid)

# =========================
# Exchange
# =========================
def load_ex(name: str):
    name = name.lower()
    if name == "okx":     return ccxt.okx({"enableRateLimit": True})
    if name == "bybit":   return ccxt.bybit({"enableRateLimit": True})
    if name == "binance": return ccxt.binance({"enableRateLimit": True, "options":{"defaultType":"future"}})
    raise ValueError("Unsupported EXCHANGE")

def fetch_ohlcv(ex, sym, tf, limit) -> Optional[pd.DataFrame]:
    raw = ex.fetch_ohlcv(sym, timeframe=tf, limit=limit)
    if not raw: return None
    df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

# =========================
# Features
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    up, mid, lo = bollinger(df["close"], CFG["BB_PERIOD"], CFG["BB_STD"])
    bb_w = (up - lo) / mid
    efast = ema(df["close"], CFG["EMA_FAST"])
    eslow = ema(df["close"], CFG["EMA_SLOW"])
    m_line, m_sig, m_hist = macd(df["close"], CFG["MACD_FAST"], CFG["MACD_SLOW"], CFG["MACD_SIGNAL"])
    A = atr(df, CFG["ATR_PERIOD"])
    RSI = rsi(df["close"], CFG["RSI_PERIOD"])

    out = df.copy()
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = up, mid, lo
    out["bb_w"] = bb_w
    out["ema_fast"], out["ema_slow"] = efast, eslow
    out["macd_line"], out["macd_sig"], out["macd_hist"] = m_line, m_sig, m_hist
    out["atr"] = A
    out["rsi"] = RSI

    # rolling percentile (no lookahead)
    out["bb_pctl"] = np.nan
    Np = CFG["SQUEEZE_LOOKBACK"]
    if len(out) > Np + 5:
        arr = out["bb_w"].values
        for i in range(Np, len(out)):
            hist = arr[i-Np:i]
            v = arr[i]
            out.iat[i, out.columns.get_loc("bb_pctl")] = percent_rank(hist, v)

    # volume median
    out["vol_med20"] = out["volume"].rolling(20).median()
    return out

def fetch_htf_alignment(ex, sym: str) -> Tuple[bool, bool]:
    try:
        hdf = fetch_ohlcv(ex, sym, CFG["HTF_TIMEFRAME"], 400)
        if hdf is None or len(hdf) < CFG["EMA_SLOW"] + 5:
            return True, True
        f = ema(hdf["close"], CFG["EMA_FAST"])
        s = ema(hdf["close"], CFG["EMA_SLOW"])
        return bool(f.iloc[-2] > s.iloc[-2]), bool(f.iloc[-2] < s.iloc[-2])
    except Exception:
        return True, True

# =========================
# Position State
# =========================
@dataclass
class Position:
    symbol: str
    side: str            # LONG / SHORT
    entry: float
    sl: float
    tp1_price: float
    took_tp1: bool
    size_remain: float   # 1.0 at entry, 0.3 after TP1
    ts_entry: str
    reason: str

def load_state() -> Dict[str, dict]:
    fp = CFG["STATE_FILE"]
    if not os.path.exists(fp): return {}
    try:
        with open(fp, "r") as f: return json.load(f)
    except Exception:
        return {}

def save_state(st: Dict[str, dict]):
    try:
        with open(CFG["STATE_FILE"], "w") as f:
            json.dump(st, f, indent=2)
    except Exception:
        print("[state save error]")

# =========================
# Alert formatting
# =========================
def fmt_percent(x: float, digits=1) -> str:
    return f"{x:.{digits}f}%"

def fmt_price(x: float) -> str:
    # 간단 포맷 (거래소 틱 규칙은 주문시 반올림 적용 권장)
    if x >= 1000: return f"{x:,.0f}"
    if x >= 100:  return f"{x:,.2f}"
    if x >= 1:    return f"{x:,.4f}"
    return f"{x:.6f}"

def entry_alert(symbol, side, tf, entry, sl, tp1_price, sl_pct, pctl, vol_boost, htf_trend, confirm_tag, risk_pct=0.75):
    star = "⭐" if symbol in CFG["STAR_SYMBOLS"] else ""
    head = f"[ENTRY{star}] {tf} | {symbol} | {side.capitalize()}"
    body1 = f"Entry {fmt_price(entry)} | SL {fmt_price(sl)} (ATR×{CFG_SYMBOL[symbol]['ATR_MULT']}, ΔSL {fmt_percent(sl_pct)}) | TP1 +1.0R @ {fmt_price(tp1_price)} (≈+{fmt_percent(sl_pct)})"
    filt = f"Risk {risk_pct:.2f}% | HTF{'↑' if htf_trend=='up' else '↓' if htf_trend=='down' else '-'} | BB squeeze p{int(pctl)} | Vol>{vol_boost}×med"
    text = f"{head}\n{body1}\n{filt}"
    return text

def reason_alert(bu, reason_notes):
    return f"Reason: {bu}\nNotes: {reason_notes}"

def tp1_alert(symbol, side, tf, entry, new_sl, realized_r):
    head = f"[TP1] {tf} | {symbol} {side.capitalize()} | +{realized_r:.1f}R realized"
    body = f"SL → BE({fmt_price(entry)})로 이동, 잔여 30% BB mid 트레일"
    return f"{head}\n{body}"

def exit_alert(symbol, side, reason, price, total_r=None, entry=None):
    if reason == "BE_stop":
        return f"[EXIT] {symbol} {side.capitalize()} | BE_stop at {fmt_price(entry)} | Total +0.7R"
    if reason == "MidCross":
        return f"[EXIT] {symbol} {side.capitalize()} | MidCross | Total {total_r:+.1f}R"
    if reason == "SL":
        return f"[EXIT] {symbol} {side.capitalize()} | SL {fmt_price(price)} | -1.0R"
    # fallback
    return f"[EXIT] {symbol} {side.capitalize()} | {reason}"

# =========================
# Core signal loop
# =========================
def run_loop():
    ex = load_ex(CFG["EXCHANGE"])
    state = load_state()

    while True:
        try:
            for sym in CFG["SYMBOLS"]:
                params = CFG_SYMBOL[sym]
                df = fetch_ohlcv(ex, sym, CFG["TIMEFRAME"], CFG["FETCH_LIMIT"])
                if df is None or len(df) < 300: 
                    print(f"[skip] {sym} not enough data")
                    continue
                df = build_features(df)

                # HTF trend
                htf_up, htf_down = (True, True)
                if params.get("USE_HTF", True):
                    htf_up, htf_down = fetch_htf_alignment(ex, sym)
                htf_trend = "up" if htf_up else ("down" if htf_down else "-")

                # 마지막 2개 바 기준: 신호바=직전, 엔트리바=현재
                i = len(df) - 1
                prev = df.iloc[i-1]
                prev2 = df.iloc[i-2]
                curr = df.iloc[i]    # 엔트리 가격으로 현재 봉 open 사용

                # 트렌드(15m)
                t_long, t_short = trend_flags(prev)

                # 확인(ENG 또는 ENG+PIN)
                bull_eng = detect_bull_engulf(prev2.open, prev2.close, prev.open, prev.close)
                bear_eng = detect_bear_engulf(prev2.open, prev2.close, prev.open, prev.close)
                confirm_mode = params["CONFIRM_MODE"]
                if confirm_mode == "ENG":
                    conf_long = bull_eng
                    conf_short = bear_eng
                    confirm_tag = "ENG"
                else:
                    bull_pin = detect_pinbar(prev.open, prev.high, prev.low, prev.close, True)
                    bear_pin = detect_pinbar(prev.open, prev.high, prev.low, prev.close, False)
                    conf_long = bull_eng or bull_pin
                    conf_short = bear_eng or bear_pin
                    confirm_tag = "ENG+PIN" if (bull_pin or bear_pin) else "ENG"

                # 필터
                base_long = squeeze_ok(prev, params["SQUEEZE_PCTL"]) and breakout_ok(prev, "long") \
                            and conf_long and macd_ok(prev2, prev, "long") \
                            and rsi_ok(prev, "long", params.get("USE_RSI_FILTER", True)) \
                            and vol_ok(prev, params.get("USE_VOL_FILTER", True), params["VOL_BOOST"], prev.vol_med20)
                base_short = squeeze_ok(prev, params["SQUEEZE_PCTL"]) and breakout_ok(prev, "short") \
                            and conf_short and macd_ok(prev2, prev, "short") \
                            and rsi_ok(prev, "short", params.get("USE_RSI_FILTER", True)) \
                            and vol_ok(prev, params.get("USE_VOL_FILTER", True), params["VOL_BOOST"], prev.vol_med20)

                long_sig = base_long and t_long
                short_sig = base_short and t_short

                if params.get("USE_HTF", True):
                    long_sig = long_sig and htf_up
                    short_sig = short_sig and htf_down

                # 쿨다운: 동일 심볼 최근 엔트리로부터 bars 제한
                key = f"{sym}"
                srec = state.get(key, {})
                last_idx = srec.get("last_entry_idx", -1)
                cooldown = params["COOLDOWN_BARS"]
                can_long = long_sig and ((i - 1) - last_idx >= cooldown)
                can_short = short_sig and ((i - 1) - last_idx >= cooldown)

                # 포지션 유무
                pos: Optional[Position] = None
                if srec.get("position"):
                    pos = Position(**srec["position"])

                # ========== ENTRY ==========
                if (can_long or can_short) and pos is None:
                    side = "LONG" if can_long else "SHORT"
                    entry = float(curr.open)
                    sl = get_sl(entry, prev, df, i-1, is_long=(side=="LONG"),
                                method=CFG["SL_METHOD"], atr_mult=params["ATR_MULT"])
                    risk = (entry - sl) if side=="LONG" else (sl - entry)
                    if risk > 0 and np.isfinite(risk):
                        sl_pct = abs(risk/entry) * 100.0
                        tp1_price = entry + risk if side=="LONG" else entry - risk
                        # Alerts
                        bu = ("BBU 돌파" if side=="LONG" else "BBL 돌파") + (" + Bull Engulfing" if (side=="LONG" and bull_eng) else " + Bear Engulfing" if (side=="SHORT" and bear_eng) else "")
                        if confirm_mode == "ENG+PIN":
                            if side=="LONG" and detect_pinbar(prev.open, prev.high, prev.low, prev.close, True): bu += " + Pinbar"
                            if side=="SHORT" and detect_pinbar(prev.open, prev.high, prev.low, prev.close, False): bu += " + Pinbar"
                        bu += (" + MACD(hist↑)" if side=="LONG" else " + MACD(hist↓)")
                        bu += (" + RSI>50" if side=="LONG" else " + RSI<50")
                        reason_notes = f"쿨다운 {cooldown}바, HTF {htf_trend}, confirm {confirm_tag}"

                        ent_txt = entry_alert(sym, side.lower(), CFG["TIMEFRAME"],
                                              entry, sl, tp1_price, sl_pct,
                                              params["SQUEEZE_PCTL"], params["VOL_BOOST"],
                                              htf_trend, confirm_tag, risk_pct=0.75)
                        notifier.send(ent_txt)
                        notifier.send(reason_alert(bu, reason_notes))

                        # Save position
                        state[key] = {
                            "last_entry_idx": i-1,
                            "position": asdict(Position(
                                symbol=sym, side=side, entry=entry, sl=sl,
                                tp1_price=tp1_price, took_tp1=False, size_remain=1.0,
                                ts_entry=tz_kst(curr.name) if hasattr(curr, "name") else tz_kst(df.iloc[i]["ts"]),
                                reason=bu
                            ))
                        }
                        save_state(state)
                        continue

                # ========== POSITION MANAGEMENT ==========
                if pos is not None:
                    # 현재/최근 바들로 TP1/BE/MidCross/SL 체크
                    # 엔트리 이후 모든 바 스캔 대신, 최신 바 기준으로 순차 처리
                    # 최신 바
                    bar = df.iloc[i]
                    side = pos.side
                    entry = pos.entry
                    curr_sl = pos.sl
                    tp1 = pos.tp1_price
                    # 먼저 TP1/SL 동시: intrabar 우선순위 = SL(보수)
                    tp_hit = (bar.high >= tp1) if side=="LONG" else (bar.low <= tp1)
                    sl_hit = (bar.low <= curr_sl) if side=="LONG" else (bar.high >= curr_sl)

                    # 아직 TP1 안했으면
                    if not pos.took_tp1:
                        if tp_hit and not sl_hit:
                            # TP1 체결
                            pos.took_tp1 = True
                            pos.size_remain = 0.3
                            pos.sl = entry  # SL → BE
                            notifier.send(tp1_alert(pos.symbol, pos.side.lower(), CFG["TIMEFRAME"], entry, pos.sl, CFG["TP1_WEIGHT"]*CFG["TP1_R"]))
                            # 저장
                            state[key]["position"] = asdict(pos)
                            save_state(state)
                            continue
                        elif sl_hit and not tp_hit:
                            # 손절
                            notifier.send(exit_alert(pos.symbol, pos.side.lower(), "SL", price=curr_sl, entry=entry))
                            state[key]["position"] = None
                            save_state(state)
                            continue
                        # 동시 히트는 정책상 SL 우선(보수)
                        elif tp_hit and sl_hit:
                            notifier.send(exit_alert(pos.symbol, pos.side.lower(), "SL", price=curr_sl, entry=entry))
                            state[key]["position"] = None
                            save_state(state)
                            continue
                    else:
                        # TP1 이후: BE stop 또는 BB mid cross
                        be_hit = (bar.low <= pos.sl) if side=="LONG" else (bar.high >= pos.sl)
                        if be_hit:
                            notifier.send(exit_alert(pos.symbol, pos.side.lower(), "BE_stop", price=pos.sl, entry=entry))
                            state[key]["position"] = None
                            save_state(state)
                            continue
                        # BB mid cross (종가 기준)
                        mid = float(bar.bb_mid)
                        if (side=="LONG" and bar.close < mid) or (side=="SHORT" and bar.close > mid):
                            # 총 R 계산: 0.7R + (0.3 * rem_r)
                            risk = (entry - (entry - (pos.tp1_price - entry))) if side=="LONG" else ((entry + (entry - pos.tp1_price)) - entry)
                            # 위 risk 계산은 의미 없이 복잡하므로 재계산:
                            risk = abs(pos.tp1_price - entry)  # = 1.0R 폭
                            rem_r = ((bar.close - entry) / risk) if side=="LONG" else ((entry - bar.close) / risk)
                            total_r = CFG["TP1_WEIGHT"]*CFG["TP1_R"] + (1.0 - CFG["TP1_WEIGHT"]) * rem_r
                            notifier.send(exit_alert(pos.symbol, pos.side.lower(), "MidCross", price=bar.close, total_r=total_r, entry=entry))
                            state[key]["position"] = None
                            save_state(state)
                            continue

            time.sleep(CFG["SEND_INTERVAL_SEC"])
        except Exception as e:
            print("[loop error]", e)
            print(traceback.format_exc())
            time.sleep(CFG["SEND_INTERVAL_SEC"])
            continue

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    run_loop()