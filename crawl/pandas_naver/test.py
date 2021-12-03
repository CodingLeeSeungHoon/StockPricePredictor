# https://pandas-datareader.readthedocs.io/en/latest/readers/naver.html?highlight=naver


import pandas_datareader.naver as web_naver

def naver_reader(co_date, start, end):
    df = web_naver.NaverDailyReader(symbols=co_date, start=start, end=end, adjust_price=True)
    df = df.read()
    print(df)
if __name__ == "__main__":
    df = web_naver.NaverDailyReader(symbols='005930', start='20210915', end='20210923', adjust_price=True)

    df = df.read()
    print(df)