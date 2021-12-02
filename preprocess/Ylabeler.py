import pandas_datareader.naver as stock_api
import pandas as pd

class Ylabeler:
    """ this is class for y data labeling. Stock price to y data """
    def __init__(self):
        pass

    def _get_co_code_with_co_name(self, co_name):
        loaded_csv = pd.read_csv("../Data/data_1213_20211114.csv", engine='python')
        sorted_data = loaded_csv[['단축코드', '상장주식수', '한글 종목약명']].sort_values('한글 종목약명')
        return int(sorted_data.loc[sorted_data['한글 종목약명'] == co_name]['단축코드'])

    def _get_stock_price_with_co_code_and_date(self, co_code, start_date, end_date):
        """
        get stock price value with company code by naver pandas api
        :param co_code: String
        :return: integer
        """
        df = stock_api.NaverDailyReader(symbols=co_code, start=start_date, end=end_date, adjust_price=True)
        df = df.read().tolist()
        print(df)
        pass

    @staticmethod
    def _convert_price_to_y(self, price):
        """
        convert stock price to y data
        :param price: Integer
        :return: Double
        """
        pass


if __name__ == '__main__':
    ylabeler = Ylabeler()
    ylabeler._get_co_code_with_co_name()