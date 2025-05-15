import utils


def lambda_handler(event, context):
    
    try:
        symbol = 'BTCUSDT'
        trade = "futures"

        action, price = utils.ai(symbol)

        if action == 1:
            side = "BUY"
        elif action == -1:
            side = "SELL"
        else:
            side = "HOLD"
        
        return {
            "statusCode":200,
            "info":{
                "price":price,
                "symbol":symbol,
                "side":side,
                "positionSide":"BOTH",
                "trade":trade
            }
        }
    
    except Exception as e:
        print(e)
        return {
            "statusCode":500,
            "info":str(e)
        }


if __name__ == '__main__':
    event={}
    context={}
    print(lambda_handler(event, context))