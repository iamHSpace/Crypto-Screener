import requests
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle

# market = 'BTCUSDT'
# tick_interval = '1m'

# url = 'https://api.binance.com/api/v1/klines?symbol='+market+'&interval='+tick_interval
# df = pd.read_json(url)

# # desired output
# # [
# #   [
# #     1499040000000,      // Open time
# #     "0.01634790",       // Open
# #     "0.80000000",       // High
# #     "0.01575800",       // Low
# #     "0.01577100",       // Close
# #     "148976.11427815",  // Volume
# #     1499644799999,      // Close time
# #     "2434.19055334",    // Quote asset volume
# #     308,                // Number of trades
# #     "1756.87402397",    // Taker buy base asset volume
# #     "28.46694368",      // Taker buy quote asset volume
# #     "17928899.62484339" // Ignore
# #   ]
# # ]

# df.columns = [ "date","open","high","low","clos"eg,"volume",
# 	"close time","quote asset volume","number of trades","taker buy base asset volume",
# 	"Taker buy quote asset volume","ignore"]
# df['date'] =  pd.to_datetime(df['date'],dayfirst=True, unit = 'ms')

# df.set_index('date',inplace=True)
# print(df.tail())

def Cryptodata2(symbol,tick_interval='1m'):
	url = 'https://api.binance.com/api/v1/klines?symbol='+symbol+'&interval='+tick_interval
	df = pd.read_json(url)
	df.columns = [ "date","open","high","low","close","volume",
	"close time","quote asset volume","number of trades","taker buy base asset volume",
	"Taker buy quote asset volume","ignore"]
	df['date'] =  pd.to_datetime(df['date'],dayfirst=True, unit = 'ms')
	df.set_index('date',inplace=True)
	del df['ignore']
	return df
def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/(down+0.001)
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/(down+0.001)
        rsi[i] = 100. - 100./(1.+rs)

    return rsi
def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas
def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a
def percent_B(df, n=14):
	Price=df.close
	Upper_Band = df.close.rolling(window=20).mean() + 2*(df.close.rolling(window=20).std())
	Lower_Band = df.close.rolling(window=20).mean() - 2*(df.close.rolling(window=20).std())
	b = (Price - Lower_Band)/(Upper_Band - Lower_Band)
	return b*100
def computeMACD(x, slow=26, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslo
def FinalRsi(Data,n=14):
	close=Data.close
	rsi=rsiFunc(close,n)
	rsiFrame=pd.DataFrame(rsi)
	rsiFrame.reset_index()
	date=pd.DataFrame(Data.index)
	date.reset_index()
	newrsi=rsiFrame.join(date)
	newrsi.set_index('date',inplace=True)
	return newrsi
def FinalMovingAverage(Data, window):
	close=Data.close
	avg=movingaverage(close,window)
	avgFrame=pd.DataFrame(avg)
	avgFrame.reset_index()
	date=pd.DataFrame(Data.index)
	date.reset_index()
	newavg=avgFrame.join(date)
	newavg.set_index('date',inplace=True)
	return newavg
def FinalExpMovingAverage(Data,window):
	close=Data.close
	avg=ExpMovingAverage(close,window)
	avgFrame=pd.DataFrame(avg)
	avgFrame.reset_index()
	date=pd.DataFrame(Data.index)
	date.reset_index()
	newavg=avgFrame.join(date)
	newavg.set_index('date',inplace=True)
	return newavg
def makeData(df, nrsi=14, avg_window=10, expavg_window=10, nPercent_B=14):
	cc= FinalRsi(df,n=nrsi)
	df['rsi'] = cc
	del cc
	cc = df.close.rolling(avg_window).mean()
	df['avg'] = cc 
	del cc
	cc =FinalExpMovingAverage(df,expavg_window)
	df['expavg']=cc
	del cc
	cc= percent_B(df,nPercent_B)
	df['percent_B'] = cc
	del cc
	return df 
# def liveCoin(coin,exchange="USDT"):
# 	url = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'
# 	df = pd.read_json(url)
# 	df.columns = ["symbol","price"]
# 	return df

altcoins2=['BTC','ETH','XRP','BCHABC','EOS','BCHSV','TRX','BNB','NEO',
	'LTC','PAX','TUSD','ADA','XLM','ONT','USDC','IOTA','VET','ICX',
	'QTUM','NULS']
actcoin_data = {}

for altcoin in altcoins2:
    coinpair = '{}USDT'.format(altcoin)
    crypto_price_df = Cryptodata2(coinpair)
    crypto_price_df['weekday']=crypto_price_df.index.weekday
    crypto_price_df['change']=crypto_price_df.close.diff()
    actcoin_data[altcoin] = makeData(crypto_price_df)		
c_rsi = pd.DataFrame()
c_close = pd.DataFrame()
c_avg = pd.DataFrame()
c_high = pd.DataFrame()
c_low = pd.DataFrame()
c_open = pd.DataFrame()
c_change = pd.DataFrame()
c_expavg = pd.DataFrame()
c_percent_B = pd.DataFrame()


for altcoin in altcoins2:
	c_rsi[altcoin] = actcoin_data[altcoin].rsi
	c_close[altcoin] = actcoin_data[altcoin].close
	c_avg[altcoin] = actcoin_data[altcoin].avg
	c_high[altcoin] = actcoin_data[altcoin].high
	c_low[altcoin] = actcoin_data[altcoin].low
	c_open[altcoin]= actcoin_data[altcoin].open
	c_change[altcoin]= actcoin_data[altcoin].change
	c_expavg[altcoin]= actcoin_data[altcoin].expavg
	c_percent_B[altcoin]= actcoin_data[altcoin].percent_B

print("What you want? \nType 1 or 2")
print("1 - Live Data")
print("2 - Historical Data")
z1 = int(input(">"))

if z1 == 1:
	# Live Data
	print("Enter your coin among following choices:")
	print("BTC,ETH,XRP,BCHABC,EOS,BCHSV,TRX,BNB,NEO,LTC,PAX,TUSD,ADA,XLM,ONT,USDC,IOTA,VET,ICX,QTUM,NULS")
	inCoin=input(">")
	inCoin = inCoin + "USDT"
	print(Cryptodata2(inCoin).tail(1))

elif z1 == 2:
	print("Choose by entering number:")
	print("1 - By Coin")
	print("2 - By Indicator")
	z2 = int(input(">"))

	if z2 == 1:		
		print("Enter your coin among following choices:")
		print("BTC,ETH,XRP,BCHABC,EOS,BCHSV,TRX,BNB,NEO,LTC,PAX,TUSD,ADA,XLM,ONT,USDC,IOTA,VET,ICX,QTUM,NULS")
		inCoin = input(">")
		inCoin = inCoin + "USDT"
		print("Frequency of data? Enter one among following:")
		print("1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M")
		freq = input(">")
		print(Cryptodata2(inCoin,freq))



	elif z2 == 2:
		print("Enter you Indicator code among following:")
		print("close,rsi,per_b,m_avg,exp_m_avg")
		indi = input(">")
		# print("Frequency of data? Enter one among following:")
		# print("1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M")
		# freq = input(">")




		if indi == "close":
			print(c_close)
		elif indi == "rsi":
			print(c_rsi)
		elif indi == "per_b":
			print(c_percent_B)
		elif indi == "m_avg":
			print(c_avg)
		elif indi == "exp_m_avg":
			print(c_expavg)

		
		else:
			print("Try again from start. It's shitty, I know.")

	else:
		print("Try agin from start. Shitty Code you know.")	

else:
	print("Try again from Start.")