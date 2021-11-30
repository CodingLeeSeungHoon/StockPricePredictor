import numpy as np
import pandas as pd
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
from preprocess import Stopwords, Vectorization, Ylabeler


class Preprocessor:
    def __init__(self, path="../Data/news_datatest.csv"):
        # raw_csv = pandas instance
        self.raw_csv = self.__get_csv(path)
        # company name list from csv
        self.co_name_list = None

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

    def __init_stop_word_set(self):
        """
        init stop word set
        :return:
        """
        pass

    def __get_processed_csv(self):
        """
        get csv after preprocessed
        :return:
        """
        pass


    def __get_stock_price_with_co_code(self, co_code):
        """
        get stock price value with company code by naver pandas api
        :param co_code: String
        :return: integer
        """
        pass

    @staticmethod
    def __convert_price_to_y(self, price):
        """
        convert stock price to y data
        :param price: Integer
        :return: Double
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
        pass


if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor._test_open_csv()
