import requests
from datetime import datetime
import torch
import numpy as np
from ppo import PPO
import pandas as pd
import os
import indicator

# 1분, 5분, 30분, 1시간 등의 데이터(현재 데이터)를 거래소로부터 가져온다.
def getCurrentData(symbol, interval='5m', limit=None):
    url = "https://api.binance.com/api/v3/klines"
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Base asset volume', 'Number of trades',\
                'Taker buy volume', 'Taker buy base asset volume', 'Ignore']
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": None,
        "endTime": None,
        "limit": limit
    }

    res = requests.get(url, params=params)
    value = res.json()

    df = pd.DataFrame(value, columns=columns)
    
    df['Open'] = df['Open'].astype('float')
    df['High'] = df['High'].astype('float')
    df['Low'] = df['Low'].astype('float')
    df['Close'] = df['Close'].astype('float')

    df['Open time'] = df['Open time'].astype('int')
    df['Open time'] = df['Open time'].apply(lambda x : datetime.fromtimestamp(x/1000))
    df['Close time'] = df['Close time'].astype('int')
    df['Close time'] = df['Close time'].apply(lambda x : datetime.fromtimestamp(x/1000))
    df = df.set_index('Open time')

    return df, df['Close'].iloc[-1]

def preprocess(ticker='BTCUSDT'):
    data_6h, _ = getCurrentData(ticker, '6h', limit=210)
    data_30m, price = getCurrentData(ticker, '30m', limit=40)
    
    data = pd.DataFrame()
    data['Open'] = data_30m['Open']
    data['Close'] = data_30m['Close']
    data['High'] = data_30m['High']
    data['Low'] = data_30m['Low']
    data['Volume'] = data_30m['Volume']

    """하이킨 아시 캔들 계산"""
    # Heikin Ashi 캔들 계산
    data['ha_close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    data['ha_open'] = (data['Open'].shift(1) + data['Close'].shift(1)) / 2
    data['ha_high'] = data[['High', 'Open', 'Close']].max(axis=1)
    data['ha_low'] = data[['Low', 'Open', 'Close']].min(axis=1)
    
    # 캔들 특성 계산
    data['ha_body'] = abs(data['ha_close'] - data['ha_open'])
    data['ha_lower_wick'] = np.minimum(data['ha_open'], data['ha_close']) - data['ha_low']
    data['ha_upper_wick'] = data['ha_high'] - np.maximum(data['ha_open'], data['ha_close'])
    
    # 하이킨 아시 신호 생성 (1: 상승, 0: 중립, -1: 하락)
    data['ha_signal'] = 0
    data.loc[(data['ha_close'] > data['ha_open']) & 
             (data['ha_lower_wick'] < 1e-6) & 
             (data['ha_body'] > 0.5), 'ha_signal'] = 1
    data.loc[(data['ha_close'] < data['ha_open']) & 
             (data['ha_upper_wick'] < 1e-6) & 
             (data['ha_body'] > 0.5), 'ha_signal'] = -1


    data['ha_high_diff'] = data['ha_high'] - data['ha_high'].shift(1)
    data['ha_low_diff'] = data['ha_low'] - data['ha_low'].shift(1)
    data['ha_body_diff'] = data['ha_body'] - data['ha_body'].shift(1)
    
    """6시간봉 200 EMA 및 신호 계산"""
    # 가정:
    # data_6h: 6시간봉 데이터가 포함된 Pandas DataFrame. DatetimeIndex를 가지며 'Close' 컬럼이 있어야 합니다.
    # data: 최종 결과를 저장할 원본 Pandas DataFrame (예: 30분봉 데이터). DatetimeIndex를 가져야 합니다.
    # pandas 라이브러리는 pd로 import 되어 있어야 합니다. (예: import pandas as pd)

    # 1. 6시간봉 데이터 기준으로 200-period EMA 계산
    #    adjust=False는 금융 데이터에서 EMA를 계산할 때 일반적인 설정입니다.
    ema_200_on_6h = data_6h['Close'].ewm(span=200, adjust=False).mean()

    # 2. 6시간봉 데이터 기준으로 신호 생성
    #    신호는 data_6h의 인덱스를 기준으로 계산됩니다.
    #    초기값은 0 (중립)으로 설정합니다.
    signal_on_6h = pd.Series(0, index=data_6h.index, dtype=int)
    
    # 'Close' 가격이 EMA 위에 있으면 신호는 1
    signal_on_6h.loc[data_6h['Close'] > ema_200_on_6h] = 1
    # 'Close' 가격이 EMA 아래에 있으면 신호는 -1
    signal_on_6h.loc[data_6h['Close'] < ema_200_on_6h] = -1

    # 3. 계산된 6시간봉 EMA 값과 신호를 원본 DataFrame('data')의 인덱스에 맞게 매핑합니다.
    #    reindex와 ffill (forward-fill)을 사용하여, 6시간봉의 EMA 값과 신호가
    #    다음 6시간봉 값이 나올 때까지 'data' DataFrame의 해당 기간 동안 유지되도록 합니다.
    data['ema_200'] = ema_200_on_6h.reindex(data.index, method='ffill')
    data['ema_200_signal'] = signal_on_6h.reindex(data.index, method='ffill')
    
    # 'data' DataFrame의 시작 부분에서 EMA/신호 계산에 필요한 6시간봉 데이터가 부족하여
    # reindex 후에도 NaN 값이 남아있을 수 있습니다. 이 경우 신호를 0으로 채우고 정수형으로 변환합니다.
    data['ema_200_signal'] = data['ema_200_signal'].fillna(0).astype(int)
    
    # print(data[['Close', 'ema_200', 'ema_200_signal']].tail()) # 디버깅 용도로 필요시 주석 해제하여 사용

    """Stochastic RSI 계산"""
    # RSI 계산
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic RSI 계산
    data['stoch_rsi'] = (data['RSI'] - data['RSI'].rolling(14).min()) / \
                        (data['RSI'].rolling(14).max() - data['RSI'].rolling(14).min())
    
    # Stochastic RSI 신호 생성 (1: 과매수, 0: 중립, -1: 과매도)
    data['stoch_signal'] = 0
    data.loc[data['stoch_rsi'] < 0.2, 'stoch_signal'] = -1  # 과매도
    data.loc[data['stoch_rsi'] > 0.8, 'stoch_signal'] = 1   # 과매수

    """볼린저 밴드 계산"""
    bollinger_bands = indicator.Bollinger(data, window=20, num_std_dev=2)
    data['bb_middle'] = bollinger_bands['bb_middle']
    data['bb_upper'] = bollinger_bands['bb_upper']
    data['bb_lower'] = bollinger_bands['bb_lower']
    data['bb_width'] = bollinger_bands['bb_width']
    data['bb_width_change'] = bollinger_bands['bb_width_change']

    """MACD 계산"""
    MACD = indicator.MACD(data, cross=False)
    data['MACD'] = MACD['Histogram']
    data['MACD_Signal'] = MACD['Signal Line']
    data['Cross Signal'] = MACD['Cross Signal']
    data['Divergence Signal'] = MACD['Divergence Signal']
    data['Trade Signal'] = MACD['Trade Signal']

    # 필요한 컬럼만 선택
    data = data[['Open', 'Close', 'High', 'Low', 'Volume',
                'ha_close', 'ha_open', 'ha_high', 'ha_low',
                'ha_body', 'ha_lower_wick', 'ha_upper_wick',
                'ha_signal', 'ha_high_diff', 'ha_low_diff', 'ha_body_diff',
                'ema_200', 'ema_200_signal',
                'stoch_rsi', 'stoch_signal',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_width_change',
                'MACD', 'MACD_Signal', 'Cross Signal', 'Divergence Signal', 'Trade Signal']]
    
    # NaN 값 제거
    data = data.dropna()
    data = data.iloc[-1]
    state = np.array(data, dtype=np.float32)
    return state, price

def ai(ticker='BTCUSDT'):
    try:
        state, price = preprocess(ticker=ticker)

        file_path = load_file('./model')
        checkpoint = load_checkpoint(file_path)

        
        # 이전 학습 상태 확인
        model_name = checkpoint.get('model_name', 'ppo')
        state_dim = checkpoint.get('state_dim', 12)
        action_dim = checkpoint.get('action_dim', 1)
        gamma = checkpoint.get('gamma', 0.99)
        epsilon = checkpoint.get('epsilon', 0.2)
        epochs = checkpoint.get('epochs', 10)
        
        # 옵티마이저에서 학습률 가져오기
        optimizer_state = checkpoint['optimizer_state_dict']
        lr_actor = optimizer_state['param_groups'][0]['lr']  # actor의 학습률
        lr_critic = optimizer_state['param_groups'][3]['lr']  # critic의 학습률

             
        ppo_agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            model_name=model_name,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            epsilon=epsilon,
            epochs=epochs
        )

        # 모델 가중치 로드
        ppo_agent.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        optimizer_state['param_groups'][0]['lr'] = lr_actor  # actor 학습률
        optimizer_state['param_groups'][3]['lr'] = lr_critic  # critic 학습률
        ppo_agent.optimizer.load_state_dict(optimizer_state)

        
        action, value, log_prob = ppo_agent.select_action(state)

        return action, price

    except Exception as e:
        print(f"\n에러 발생: {str(e)}")
        raise e

def load_file(directory):
    """현재 디렉터리에서 가장 먼저 발견된 .pth 파일 반환"""
    for file in os.listdir(directory):
        if file.endswith(".pth"):
            file_path = os.path.join(directory, file)
            return file_path
            
    return None  # 이 부분은 실행되지 않겠지만, 예외 처리를 위해 남김

def load_checkpoint(file_path):
    if torch.cuda.is_available():
        return torch.load(file_path, map_location=torch.device('cuda'))
    else:
        return torch.load(file_path, map_location=torch.device('cpu'))