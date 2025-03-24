import torch
import os
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_file(directory):
    """현재 디렉터리에서 가장 먼저 발견된 .pth 파일 반환"""
    for file in os.listdir(directory):
        if file.endswith(".pth"):
            file_path = os.path.join(directory, file)
            return file_path
            
    return None  # 이 부분은 실행되지 않겠지만, 예외 처리를 위해 남김

def lambda_handler(event, context):
    
    try:
        file_path = load_file('./')
        
        if file_path is None:
            return {
                "statusCode":400,
                "message":"파일을 찾을 수 없습니다."
            }
        
        
        action = utils.test(file_path)
        
        if action > 0:
            side = "BUY"
        elif action < 0:
            side = "SELL"
        else:
            side = "HOLD"
    
        return {
            "statusCode":200,
            "side":side
        }
    except Exception as e:
        return {
            "statusCode":500,
            "message":str(e)
        }


if __name__ == '__main__':
    event={}
    context={}
    print(lambda_handler(event, context))
