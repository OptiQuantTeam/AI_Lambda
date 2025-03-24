import requests
from datetime import datetime
import torch
from ppo import PPO
from indicator import RSI, EMA, CHG, StochasticRSI, MACD
import pandas as pd

# 1분, 5분, 30분, 1시간 등의 데이터(현재 데이터)를 거래소로부터 가져온다.
def getCurrentData(symbol, interval='1m', limit=None):
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
    df['Volume'] = df['Volume'].astype('float')
    df['RSI'] = RSI(df)
    df['EMAF'] = EMA(df, window=10)

    df['Open time'] = df['Open time'].astype('int')
    df['Open time'] = df['Open time'].apply(lambda x : datetime.fromtimestamp(x/1000))
    df['Close time'] = df['Close time'].astype('int')
    df['Close time'] = df['Close time'].apply(lambda x : datetime.fromtimestamp(x/1000))
    df = df.set_index('Open time')
    #print(df)
    df['CHG'] = CHG(df)
    df['stocRSI'] = StochasticRSI(df)
    df['MACD'] = MACD(df)
    df = df[['Open','Close','Volume','CHG','stocRSI','MACD']]

    return df



def preprocess_data(df):
    """데이터 전처리: 결측치 처리 및 정규화"""
    # 필요한 컬럼만 선택
    df = df[['Open', 'Close', 'Volume', 'CHG', 'stocRSI', 'MACD']]
    
    # 결측치 처리
    #df = df.fillna(method='ffill')  # 앞의 값으로 채우기
    df = df.ffill()
    df = df.bfill()
    #df = df.fillna(method='bfill')  # 뒤의 값으로 채우기

    # 이상치 제거 (극단값 제거)
    for column in ['Open', 'Close', 'Volume', 'CHG']:
        q1 = df[column].quantile(0.01)
        q3 = df[column].quantile(0.99)
        df[column] = df[column].clip(q1, q3)
    
    # 정규화
    '''
    for column in ['Open', 'Close', 'Volume', 'CHG']:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column] - mean) / (std + 1e-8)
    '''
    # stocRSI와 MACD는 이미 정규화된 형태이므로 극단값만 처리
    df['stocRSI'] = df['stocRSI'].clip(0, 100)
    df['MACD'] = df['MACD'].clip(-10, 10)  # 적절한 범위로 조정
    
    return df.iloc[-1]



def load_checkpoint(file_path):
    if torch.cuda.is_available():
        return torch.load(file_path, map_location=torch.device('cuda'))
    else:
        return torch.load(file_path, map_location=torch.device('cpu'))

def test(file_path):
    try:
        state = getCurrentData("BTCUSDT", "1h", limit=12)
        state = preprocess_data(state)  

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

        return action

    except Exception as e:
        print(f"\n에러 발생: {str(e)}")
        raise e