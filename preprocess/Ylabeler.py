import pandas_datareader.naver as stock_api
import pandas as pd
import Preprocessor

from datetime import datetime
from dateutil.relativedelta import relativedelta

class Ylabeler:
    """ this is class for y data labeling. Stock price to y data """
    def __init__(self, date, co_list):
        self.date = date
        self.co_list = co_list
        pass

    def _get_co_code_with_co_name(self, co_name):
        loaded_csv = pd.read_csv("../Data/data_1213_20211114.csv", engine='python')
        sorted_data = loaded_csv[['단축코드', '상장주식수', '한글 종목약명']].sort_values('한글 종목약명')
        return '{0:06d}'.format(int(sorted_data.loc[sorted_data['한글 종목약명'] == co_name]['단축코드']))

    def _get_stock_price_with_co_code_and_date(self, co_code, start_date, end_date):
        """
        get stock price value with company code by naver pandas api
        :param co_code: String
        :return: integer
        """
        df = stock_api.NaverDailyReader(symbols=co_code, start=start_date, end=end_date, adjust_price=True)
        df = df.read()
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

    def __date_interval(self, date):
        """
        3일 기준으로 날짜를 return
        """
        split_date = date.split('.')
        year = int(split_date[0])
        month = int('{0:02d}'.format(int(split_date[1])))
        day = int('{0:02d}'.format(int(split_date[2])))
        temp_date = datetime(year, month, day)
        after_three_day = str((temp_date + relativedelta(days=2)).date())
        temp_date = str(temp_date.date())
        return ''.join(temp_date.split('-')), ''.join(after_three_day.split('-'))

    def _ylaber_step(self):
        """
        this is function ylaber step
        1) get data from preprocessor
        2) each data get co_code 
        3) get stock price (naver api)
        4) convert price to y
        """
        for i in range(len(self.date)):
            if self.co_list[i] != False:
                co_code = self._get_co_code_with_co_name(co_list[i][0])
                start_date, after_three_day = self.__date_interval(self.date[i])
                print(start_date, after_three_day)
                self._get_stock_price_with_co_code_and_date(co_code, start_date, after_three_day)
            
        pass


if __name__ == '__main__':
    pre = Preprocessor.Preprocessor()
    date, co_list = pre._preprocess_step()
    ylabeler = Ylabeler(date, co_list)
    ylabeler._ylaber_step()
