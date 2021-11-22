# https://pandas-datareader.readthedocs.io/en/latest/readers/naver.html?highlight=naver


import pandas_datareader.naver as web_naver


df  = web_naver.NaverDailyReader(symbols='005930', start='20201111', end='20211114', adjust_price=True)

df = df.read()
print(df)