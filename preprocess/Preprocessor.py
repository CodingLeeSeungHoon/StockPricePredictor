import numpy as np
import pandas as pd
import re
import nltk

from nltk.tokenize import word_tokenize as wt
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize

import konlpy
from konlpy.tag import Kkma
from konlpy.tag import Hannanum
from konlpy.tag import Okt
from konlpy.tag import Komoran

from nltk.stem import WordNetLemmatizer as lemma
from nltk.stem import PorterStemmer as stemming
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import json
#from preprocess import Stopwords, Vectorization, Ylabeler
import Stopwords, Vectorization, Ylabeler

class Preprocessor:
    def __init__(self, path="../Data/news_datatest.csv"):
        # raw_csv = pandas instance
        self.raw_csv = self.__get_csv(path)
        # company name list from csv
        self.co_name_list = set(self.__get_company_names())

        """ initializing tools for preprocess (stopwords, vectorization, y_labeling) """
        self.stopwords = Stopwords
        self.vectorization = Vectorization
        self.y_labeler = Ylabeler

        """ output = CSV file with X and Y """
        self.output = None

    @staticmethod
    def __get_csv(path):
        """
        get raw data csv
        :param path: String type
        :return: pandas instance
        """
        return pd.read_csv(path, engine='python')

    def _test_open_csv(self):
        """
        test code for raw csv data
        :return:
        """
        print(self.raw_csv)

    def __get_company_names(self):
        """
        get company name from csv
        :return: list
        """
        loaded_csv = pd.read_csv("../Data/data_1213_20211114.csv", engine='python')
        sorted_data = loaded_csv[['단축코드', '상장주식수', '한글 종목약명']].sort_values('한글 종목약명')
        return sorted_data['한글 종목약명'].tolist()

    def __get_token_and_company_name(self):
        """
        공시 1차적으로 약한 토큰화 진행, 회사 이름 뽑아내기
        회사 이름 없거나 2개 이상으로 뽑히면 False로 처리
        :return: 공시 딕셔너리, 공시에서 언급된 회사이름 리스트
        
        뉴스에 맞는 날짜도 가져와야 해서 날짜 딕셔너리 추가.
        2021.10.10, 13
        """
        # print(self.raw_csv['title'])
        title_dict = {}
        date_dict = []
        raw_csv_to_co_name = []
        
        for idx, row in self.raw_csv.iterrows():
            title_dict[str(idx)] = word_tokenize(row['title'])
            date_dict.append(word_tokenize(row['date'])[0])

        for t in title_dict.values():
            raw_csv_to_co_name.append(list(set(t) & self.co_name_list) if len(set(t) & self.co_name_list) == 1 else False)

        return title_dict, date_dict, raw_csv_to_co_name

    def __get_processed_csv(self):
        """
        get csv after preprocessed
        :return:
        """
        pass
    
    def _preprocess_step(self):
        """
        this is function preprocess step
        1) raw csv ->
        2) eliminate symbol stopword ->
        3) get y labeled data from pandas naver api ->
        4) return preprocessed csv
        :return:
        """
        tokenized_dict, date_dict, csv_co_list = self.__get_token_and_company_name()
        return date_dict, csv_co_list
        pass


if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor._preprocess_step()

    