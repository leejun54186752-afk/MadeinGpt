main.py
# bot.py
import os, time, math, json, requests
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt

# ====== 환경설정 ======
SYMBOL   = 'BTC/USDT'   # Bybit 기준 심볼명 (ccxt 표기)
TIMEFRAME= '15m'
LIMIT    = 600          # 최근 봉 개수(매물대 계산용 여유 포함)
POLL_SEC = 15           # 폴링 주기 (초). 15초 권장 (웹소켓 없이 가볍게)

# 전략 파라미터
EMA_SHORT = 10
EMA_LONG  = 20
EMA_TREND = 200
USE_TREND = True

RSI_LEN   = 14
RSI_LONG_MAX  = 35   # 롱 허용 RSI ≤
RSI_SHORT_MIN = 65   # 숏 허용 RSI ≥

VOL_MA_LEN   = 20
NEED_VOL_BOOST = True  # 평균 이상 거래량만

ATR_LEN  = 14
STOP_ATR = 2.0
TAKE_ATR = 3.0

# 매물대(HVN) 근사
VP_LOOKBACK_BARS = 300   # 최근 N봉 범위에서
VP_BINS          = 40    # 가격 버킷 수
HVN_COUNT        = 3     # 상위 매물대 개수
HVN_TOL_ATR      = 0.5   # HVN 근접 허용(ATR x)
REQUIRE_HVN_NEAR = True  # HVN 근접일 때만 신호 허용

# 쿨다운(분)
COOLDOWN_MIN = 240

# 알림/트레이딩
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID   = os.getenv('TELEGRAM_CHAT_ID', '')
ENABLE_TRADING     = False  # 자동매매 켜기: True (테스트넷/모의로 충분히 검증 후!)

# ====== 유틸 ======
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(),
                    (high-prev_close).abs(),
                    (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def volume_profile_hvn(prices, volumes, lookback=300, bins=40, topk=3):
    # 최근 구간 자르기
    prices = prices[-lookback:]
    volumes = volumes[-lookback:]

    pmin, pmax = prices.min(), prices.max()
    if pmax <= pmin:
        return []

    hist, edges = np.histogram(prices, bins=bins, range=(pmin, pmax), weights=volumes)
    # 상위 k개 인덱스
    idxs = hist.argsort()[::-1][:topk]
    # 버킷 중앙가격
    centers = (edges[:-1] + edges[1:]) / 2.0
    hvn_prices = centers[idxs]
    hvn_prices = sorted(hvn_prices)
    return hvn_prices

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] 텔레그램 환경변수 없음. 메시지:", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        print("Telegram error:", e)

def now_kst():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

# ====== 데이터 소스 (Bybit via ccxt) ======
ex = ccxt.bybit({"enableRateLimit": True})
# 선물 마켓을 명시하고 싶으면: ex.options = {'defaultType': 'future'}

def fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=LIMIT):
    # ccxt 표준: [timestamp, open, high, low, close, volume]
    o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

# ====== 전략 엔진 ======
STATE_PATH = "state.json"
def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH, "r"))
    return {"last_long_ts": 0, "last_short_ts": 0}

def save_state(s):
    json.dump(s, open(STATE_PATH, "w"))

def main_loop():
    state = load_state()
    cooldown_sec = COOLDOWN_MIN * 60

    while True:
        try:
            df = fetch_ohlcv()
            if len(df) < max(EMA_TREND, VP_LOOKBACK_BARS) + 5:
                print("데이터 부족. 재시도.")
                time.sleep(POLL_SEC); continue

            # 지표
            df['ema10'] = ema(df['close'], EMA_SHORT)
            df['ema20'] = ema(df['close'], EMA_LONG)
            df['ema200']= ema(df['close'], EMA_TREND)
            df['rsi']   = rsi(df['close'], RSI_LEN)
            df['atr']   = atr(df, ATR_LEN)
            df['vol_ma']= df['volume'].rolling(VOL_MA_LEN).mean()

            last = df.iloc[-1]
            prev = df.iloc[-2]

            # 크로스 (종가 기준)
            golden = (prev['ema10'] <= prev['ema20']) and (last['ema10'] > last['ema20'])
            death  = (prev['ema10'] >= prev['ema20']) and (last['ema10'] < last['ema20'])

            trend_long_ok  = (not USE_TREND) or (last['close'] > last['ema200'])
            trend_short_ok = (not USE_TREND) or (last['close'] < last['ema200'])
            vol_ok = (not NEED_VOL_BOOST) or (last['volume'] >= last['vol_ma'])

            long_base  = golden and (last['rsi'] <= RSI_LONG_MAX)  and vol_ok and trend_long_ok
            short_base = death  and (last['rsi'] >= RSI_SHORT_MIN) and vol_ok and trend_short_ok

            # HVN 계산 (근사)
            hvns = volume_profile_hvn(df['close'].values, df['volume'].values,
                                      lookback=VP_LOOKBACK_BARS, bins=VP_BINS, topk=HVN_COUNT)
            near_hvn = False
            if len(hvns) > 0 and not math.isnan(last['atr']):
                for p in hvns:
                    if abs(last['close'] - p) <= HVN_TOL_ATR * last['atr']:
                        near_hvn = True
                        break

            long_ok  = long_base  and (near_hvn if REQUIRE_HVN_NEAR else True)
            short_ok = short_base and (near_hvn if REQUIRE_HVN_NEAR else True)

            # 쿨다운 체크 (바가 확정된 직후만 보도록 ts 사용)
            ts = int(df['ts'].iloc[-1].timestamp())
            text_common = (f"[{now_kst()}]\n"
                           f"{SYMBOL} {TIMEFRAME}\n"
                           f"가격: {last['close']:.2f}\n"
                           f"EMA10/20: {last['ema10']:.2f}/{last['ema20']:.2f}\n"
                           f"RSI: {last['rsi']:.2f} | ATR: {last['atr']:.2f}\n"
                           f"HVN: {', '.join([f'{p:.2f}' for p in sorted(hvns)])}\n")

            # 롱 신호
            if long_ok and (ts - state["last_long_ts"] >= cooldown_sec):
                stop = last['close'] - STOP_ATR * last['atr']
                take = last['close'] + TAKE_ATR * last['atr']
                msg = (text_common +
                      f"신호: LONG (합의 + HVN근접)\n"
                       "권장: 분할·손절고정\n"
                      f"손절/익절: {stop:.2f} / {take:.2f}\n"
                       "(연구용, 투자권유 아님)")
                send_telegram(msg)
                state["last_long_ts"] = ts
                save_state(state)

                if ENABLE_TRADING:
                    place_order(side='buy', price=float(last['close']),
                                stop=float(stop), take=float(take))

            # 숏 신호
            if short_ok and (ts - state["last_short_ts"] >= cooldown_sec):
                stop = last['close'] + STOP_ATR * last['atr']
                take = last['close'] - TAKE_ATR * last['atr']
                msg = (text_common +
                      f"신호: SHORT (합의 + HVN근접)\n"
                       "권장: 분할·손절고정\n"
                      f"손절/익절: {stop:.2f} / {take:.2f}\n"
                       "(연구용, 투자권유 아님)")
                send_telegram(msg)
                state["last_short_ts"] = ts
                save_state(state)

        except Exception as e:
            print("loop error:", e)

        time.sleep(POLL_SEC)

# ====== (옵션) 자동매매 스텁 ======
def place_order(side, price, stop, take):
    """
    TODO: Bybit 주문 로직.
    - 권장: Bybit 공식 라이브러리(pybit) 또는 ccxt의 create_order 사용
    - 필수: 격리모드/레버리지 설정, idempotency 키, 일일 손실 한도, 중복 주문 방지
    """
    print(f"[DRY RUN] {side.upper()} @ {price} | SL {stop} | TP {take}")
    # 예시 (나중에 활성화):
    # order = ex.create_order(symbol=SYMBOL, type='market', side='buy', amount=qty)
    # ex.create_order(symbol=SYMBOL, type='stop_market', side='sell', amount=qty,
    #                 params={'stopLossPrice': stop, 'takeProfitPrice': take})

if __name__ == "__main__":
    print("Starting bot...")
    send_telegram("🔔[투자봇] 시작했습니다. 알림 연결 OK")
    main_loop()
    # 파일 상단에 추가
from threading import Thread
from flask import Flask
import os

# ... (기존 코드 그대로) ...

app = Flask(__name__)

@app.get("/")
def health():
    return "OK", 200

def run_bot():
    # 시작 테스트 메시지를 보내고 싶으면 주석 해제
    # send_telegram("🔔[투자봇] Render(Web Service)에서 시작")
    main_loop()

if __name__ == "__main__":
    print("Starting bot as Web Service...")
    t = Thread(target=run_bot, daemon=True)
    t.start()

    port = int(os.getenv("PORT", "10000"))   # Render가 PORT를 주입해 줍니다
    app.run(host="0.0.0.0", port=port)
    