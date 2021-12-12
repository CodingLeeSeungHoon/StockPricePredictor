import sys
from os import path
sys.path.append(path.dirname(__file__))


import pandas_datareader.naver as stock_api
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta
from requests.models import encode_multipart_formdata

class Ylabeler:
    """ this is class for y data labeling. Stock price to y data """
    def __init__(self):
        pass

    def _get_national_holiday(self):
        '''
        공휴일을 얻어옵니다.
        self.national_holiday리스트에 공휴일을 저장.
        '''
        self.national_holiday = []
        national_holiday_df = pd.read_excel('../Data/national_holiday.xlsx')
        national_holiday_lst = national_holiday_df.loc[:, ['년', '월', '일']].values.tolist()
        
        for holiyday in national_holiday_lst:
            year = int(holiyday[0])
            month = int('{0:02d}'.format(int(holiyday[1])))
            day = int('{0:02d}'.format(int(holiyday[2])))
            self.national_holiday.append(str((datetime(year, month, day).date())))
        
    def __get_three_day(self, date):
        '''
        특정 날짜 기준으로 3번의 장 이후 날짜를 반환합니다.
        공휴일과 주말을 제외한 날짜를 반환.
        '''
        day = 0 # 3번의 장이 지났는지 확인하기 위해
        after = 1 #date 기준 after 이후
        while day < 2:
            temp_date = date + relativedelta(days=after)
            if temp_date not in self.national_holiday and temp_date.weekday() in range(5):
                day += 1
            after += 1
        return str(temp_date)

    def _get_co_code_with_co_name(self, co_name):
        '''
        기업명을 통해 naver_api에 필요한 기업 코드를 받아옵니다.
        '''
        loaded_csv = pd.read_csv("../Data/data_1213_20211114.csv", engine='python')
        sorted_data = loaded_csv[['단축코드', '상장주식수', '한글 종목약명']].sort_values('한글 종목약명')
        code = sorted_data[sorted_data['한글 종목약명'] == co_name]['단축코드'].values[0]
        # 문자열 6자리 밑이면 앞에 0으로 채워준다.
        return code.zfill(6)

    def _get_stock_price_with_co_code_and_date(self, co_code, start_date, end_date):
        """
        get stock price value with company code by naver pandas api
        :param co_code, start_date, end_date: String
        :return: list
        """
        df = stock_api.NaverDailyReader(symbols=co_code, start=start_date, end=end_date, adjust_price=True)
        df = df.read()
        closing_price = df['Close'].values
        stock_price = []
        for price in closing_price:
            stock_price.append(int(price))
        return stock_price
        

    @staticmethod
    def _convert_price_to_y_for_reg(price_lst):
        """
        convert stock price to y data for regression
        :param price_lst: list
        :return: Double
        """
        numerator = sum(price_lst) // 3 - price_lst[0]
        y_percent_data = round((numerator / price_lst[0]) * 100, 3)
        return y_percent_data

    @staticmethod
    def convert_to_cls(percent):
        if percent > 1:
            return 2 # 상승
        elif percent >= -1:
            return 1 # 보합
        else:
            return 0 # 하락

    @staticmethod
    def _convert_price_to_y_for_cls(price_lst):
        """
        convert stock price to y data for classification
        :param price_lst: list
        :return: 0, 1, 2 (0 = 하락 / 1 = 보합 / 2 = 상승)
        """
        numerator = sum(price_lst) // 3 - price_lst[0]
        y_percent_data = round((numerator / price_lst[0]) * 100, 3)
        return Ylabeler.convert_to_cls(y_percent_data)


    def __date_interval(self, date):
        """
        데이터를 탐색할 시작 날짜와 끝 날짜를 반환합니다.
        형식에 맞춰줘야 하기 때문에 월과 일을 두자리로 맞춰줍니다.
        """
        split_date = date.split('-')
        year = int(split_date[0])
        month = int('{0:02d}'.format(int(split_date[1])))
        day = int('{0:02d}'.format(int(split_date[2])))
        temp_date = datetime(year, month, day)
        after_three_day = self.__get_three_day(temp_date.date())
        temp_date = str(temp_date.date())
        start, end = ''.join(temp_date.split('-')), ''.join(after_three_day.split('-'))
        return start, end 

    def _ylaber_step(self, type, date, co_list):
        """
        this is function ylaber step
        1) get data from preprocessor
        2) each data get co_code 
        3) get stock price (naver api)
        4) convert price to y
        """
        self._get_national_holiday()
        y_data = []

        for i in range(len(date)):
            if co_list[i] != False:
                co_code = self._get_co_code_with_co_name(co_list[i][0])
                start_date, after_three_day = self.__date_interval(date[i])
                stock_price_lst = self._get_stock_price_with_co_code_and_date(co_code, start_date, after_three_day)
                #print(stock_price_lst)
                if type == "reg": # 회귀
                    if len(stock_price_lst) == 3:
                        stock_data = self._convert_price_to_y_for_reg(stock_price_lst)
                        y_data.append(stock_data)
                    else:
                        co_list[i] = False
                    #print(stock_data)
                else: # 분류
                    if len(stock_price_lst) == 3:
                        stock_data = self._convert_price_to_y_for_cls(stock_price_lst)
                        print(f'{i} : {stock_data}')
                        y_data.append(stock_data)
                    else:
                        co_list[i] = False
        return y_data


if __name__ == '__main__':
    ylabeler = Ylabeler()