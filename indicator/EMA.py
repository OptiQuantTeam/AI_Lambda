import pandas as pd

def EMA(data, window):

    # EMA 계산
    ema = data['Close'].ewm(span=window, adjust=False).mean()
    return ema

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # EMA 계산
    ema_result = EMA(data, window=10)
    print(ema_result)
