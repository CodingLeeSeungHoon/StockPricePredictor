import sys
from os import path

sys.path.append(path.dirname(__file__))

import pandas_datareader.naver as stock_api
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta
from requests.models import encode_multipart_formdata


class Ylabeler:
    """ 
    this is class for y data labeling. Stock price to y data
    """

    def __init__(self):
        pass

    def _get_national_holiday(self):
        '''
        self.national_holiday리스트에 공휴일만 따로 저장합니다.
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
        date를 기준으로 공휴일과 주말을 제외한 
        3일 이후 날짜를 반환합니다.
        return : str
        '''
        day = 0  
        after = 1 
        while day < 2:
            temp_date = date + relativedelta(days=after)
            if str(temp_date) not in self.national_holiday:
                if temp_date.weekday() in range(5):
                    day += 1
            after += 1
        return str(temp_date)

    def _get_co_code_with_co_name(self, co_name):
        '''
        기업명을 통해 pandas_naver_api 에 필요한 기업 코드를 받아옵니다.
        기업 코드는 6자리로 맞춰줍니다.
        return str
        '''
        loaded_csv = pd.read_csv("../Data/data_1213_20211114.csv", engine='python')
        sorted_data = loaded_csv[['단축코드', '상장주식수', '한글 종목약명']].sort_values('한글 종목약명')
        code = sorted_data[sorted_data['한글 종목약명'] == co_name]['단축코드'].values[0]
        
        return code.zfill(6)

    def _get_stock_price_with_co_code_and_date(self, co_code, start_date, end_date):
        """
        naver pandas api를 이용하여 주식 가격을 받아옵니다.
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
        """
        변동 퍼센트 기준을 1%로 잡음.
        """
        if percent > 1:
            return 2  # 상승
        elif percent >= -1:
            return 1  # 보합
        else:
            return 0  # 하락

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
        1) preprocessor로 데이터를 받는다.
        2) 기업명을 바탕으로 기업 코드를 받아온다.
        3) 공시 날짜를 기반으로 3번의 장 이후 날짜를 받는다.
        4) 3일치 주식 가격을 받아온다.
        5) cls, reg 별로 다른 y_data를 만든다.
        :param price_lst: type(cls, reg), 공시 날짜, 기업명 리스트
        :return list
        """
        self._get_national_holiday()
        y_data = []
        total_true_count = 0
        false_count = 0

        for i in range(len(date)):
            if co_list[i] != False:
                co_code = self._get_co_code_with_co_name(co_list[i][0])
                start_date, after_three_day = self.__date_interval(date[i])
                stock_price_lst = self._get_stock_price_with_co_code_and_date(co_code, start_date, after_three_day)
                # print("길이",len(stock_price_lst))
                # print(stock_price_lst)
                if type == "reg":  # 회귀
                    if len(stock_price_lst) == 3:
                        stock_data = self._convert_price_to_y_for_reg(stock_price_lst)
                        if i % 1000 == 0:
                            print(">> {}번째 계산 결과..".format(i), "변화율 =", stock_data, "%")
                        total_true_count = total_true_count + 1
                        y_data.append(stock_data)
                    else:
                        co_list[i] = False
                        total_true_count = total_true_count + 1
                    # print(stock_data)
                else:  # 분류
                    if len(stock_price_lst) == 3:
                        stock_data = self._convert_price_to_y_for_cls(stock_price_lst)
                        if i % 1000 == 0:
                            print(">> {}번째 계산 결과..".format(i), "class =", stock_data)
                        total_true_count = total_true_count + 1
                        y_data.append(stock_data)
                    else:
                        co_list[i] = False
                        total_true_count = total_true_count + 1
            else:
                false_count = false_count+1

        #print(">> 총 읽은 데이터 수 =",  total_true_count+false_count, "실제 계산된 데이터 수 =", total_true_count, "계산되지않은 데이터 수 =", false_count)
        return y_data


if __name__ == '__main__':
    ylabeler = Ylabeler()