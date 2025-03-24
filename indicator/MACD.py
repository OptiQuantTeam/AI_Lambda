import pandas as pd

def MACD(data, short_window=12, long_window=26, signal_window=9, cross=True):
    # 짧은 기간과 긴 기간의 지수 이동 평균 계산
    ema_short = data['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_window, adjust=False).mean()
    
    # MACD Line 계산
    macd_line = ema_short - ema_long
    
    # Signal Line 계산
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    
    # Histogram 계산
    macd_histogram = round(macd_line - signal_line, 2)
    
    # 결과 반환
    if cross:
        return macd_histogram
    else:
        return pd.DataFrame({
            'MACD Line': macd_line,
            'Signal Line': signal_line,
            'Histogram': macd_histogram
        }, index=data.index, dtype=float)

# 예제 데이터 사용
if __name__ == "__main__":
    # 종가 데이터 생성
    data = pd.read_csv('/workspace/data/raw/BTCUSDT/BTCUSDT-1d-2018.csv', index_col=0)
    
    # MACD 계산
    macd_result = MACD(data, cross=False)
    print(macd_result)