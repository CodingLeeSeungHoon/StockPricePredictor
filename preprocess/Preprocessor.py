import sys
from os import path
sys.path.append(path.dirname(__file__))

import numpy as np
import pandas as pd

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

from Stopwords import Stopwords
from Vectorization import Vectorization
from Ylabeler import Ylabeler

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score

class Preprocessor:
    def __init__(self, path="../Data/news_datatest.csv"):
        # raw_csv = pandas instance
        self.raw_csv = self.__get_csv(path)
        # company name list from csv
        self.co_name_list = set(self.__get_company_names())

        """ initializing tools for preprocess (stopwords, vectorization, y_labeling) """
        self.stopwords = Stopwords()
        self.vectorization = Vectorization()
        self.y_labeler = Ylabeler()

        """ output = CSV file with X and Y """
        self.output = None

    @staticmethod
    def __get_csv(path):
        """
        csv로 부터 데이터를 받아온다.
        :param path: String type
        :return: pandas instance
        """
        return pd.read_csv(path, engine='python')

    def _test_open_csv(self):
        """
        csv데이터 출력.
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

    def _get_token_and_company_name(self):
        """
        공시 1차적으로 약한 토큰화 진행, 회사 이름 뽑아내기
        회사 이름 없거나 2개 이상으로 뽑히면 False로 처리
        :return: 공시 딕셔너리, 날짜 리스트, 공시에서 언급된 회사이름 리스트

        뉴스에 맞는 날짜도 가져와야 해서 날짜 딕셔너리 추가.
        2021.10.10, 13
        """
        print(">> 총 데이터 수 = {}".format(len(self.raw_csv)))
        # 기업명을 얻기 위한 딕셔너리
        title_dict_for_getting_company = {}

        # 최종적으로 replace와 stopwords 제거된 문장을 얻는 dictionary
        title_dict = {}

        # 날짜
        date_list = []

        # csv로부터 구한 회사 이름
        company_name = []

        # 기업명을 title_dict_for_getting_company에 추가.
        # 날짜를 date_list에 추가
        for idx, row in self.raw_csv.iterrows():
            if idx % 1000 == 0:
                print(">> {}번째 기업명 추출 작업 실행 완료..".format(idx))
            title_dict_for_getting_company[str(idx)] = word_tokenize(row['title'])
            date_list.append(word_tokenize(row['date'])[0])

        # 기업명이 안뽑아 지거나 2개 이상 뽑히는 경우 False로 처리
        for t in title_dict_for_getting_company.values():
            company_name.append(list(set(t) & self.co_name_list) if len(set(t) & self.co_name_list) == 1 else False)

        # False로 처리한 기업들 제거 후 불용어 처리
        for idx, row in self.raw_csv.iterrows():
            if idx % 1000 == 0:
                print(">> {}번째 기업명 제거 작업 실행 완료..".format(idx))
            if company_name[idx] != False:
                row['title'] = row['title'].replace(company_name[idx][0], '')

            title_dict[str(idx)] = self.stopwords._get_delete_stopwords(self.stopwords._get_replace_stopwords(row['title']))

        print(">> result")
        for i in range(3):
            print(title_dict[str(i)],  date_list[i],  company_name[i])

        return title_dict, date_list, company_name

    def __get_processed_csv(self):
        """
        get csv after preprocessed
        :return:
        """
        pass

    def _preprocess_step_reg(self):
        """
        this is function preprocess step
        1) raw csv ->
        2) eliminate symbol stopword ->
        3) get y labeled data from pandas naver api ->
        4) return preprocessed csv
        :return:
        """
        tokenized_dict, date_dict, csv_co_list = self._get_token_and_company_name()
        y_data = self.y_labeler._ylaber_step("reg", date_dict, csv_co_list)

        not_coname_count = 0

        for idx, co in enumerate(csv_co_list):
            if co == False:
                not_coname_count=not_coname_count+1
                del tokenized_dict[str(idx)]
        print(">> Y값이 도출되지 않은 공시 데이터 개수 =", not_coname_count)

        print(">> 유의미한 X 데이터의 개수 =", len(tokenized_dict), '유의미한 Y 데이터의 개수 =', len(y_data))

        lst = list(tokenized_dict.values())

        # -----------------------------데이터 불러오기

        x_y_data = pd.DataFrame({'X': lst, 'Y': y_data})

        print("Dataframe")
        print(x_y_data.head(3))

        result = list(x_y_data['X'])

        NewsModel = Word2Vec(sg=0, size=300, window=1, min_count=1, workers=-1)
        NewsModel.build_vocab(result)  # 단어 생성.
        NewsModel.train(result, total_examples=len(result), epochs=100)  # 학습

        # -----------------Word2vec으로부터 임베딩된 결과 추출 dict() 형태

        embedding_results = dict()

        for token in NewsModel.wv.vocab.keys():
            embedding_results[token] = NewsModel.wv[token]

        vocab_size = len(NewsModel.wv.vocab.keys()) + 1
        embedding_matrix = np.zeros((vocab_size, NewsModel.wv.vector_size))

        # ------------------------------- 단어의 정수화
        word_to_id = dict()
        id_to_word = dict()

        all_words = NewsModel.wv.vocab.keys()

        for word in all_words:
            if word not in word_to_id:
                new_id = len(word_to_id) + 1
                word_to_id[word] = new_id
                id_to_word[new_id] = word

        # print('\nword_to_id and id_to_word')
        # print(word_to_id)
        # print(id_to_word)


        # ----------------------- 정수화 된 단어 매트릭스에 임베딩 결과로 저장
        for word, i in word_to_id.items():
            embedding_matrix[i, :] = embedding_results[word]

        # -------------------CNN 기반 데이터 분석
        encoded_result = []

        for sent in result:
            temp_id = []
            for word in sent:
                temp_id.append(word_to_id[word])
            encoded_result.append(temp_id)


        max_len = max(len(encoded_title) for encoded_title in encoded_result)
        min_len = min(len(encoded_title) for encoded_title in encoded_result)

        # -----------------------------------padding
        padded_encoded_result = pad_sequences(encoded_result, padding='post')  # 앞 부분에다가 값을 넣어줌 maxlen 파라미터로 조정가능

        print('\n인코딩 결과 확인')
        print(encoded_result[0])
        print(padded_encoded_result[0].shape, padded_encoded_result[0])

        # -------------------------------------train, test 분리
        indices = np.arange(len(padded_encoded_result))
        Y = x_y_data['Y']
        indices_train, indicies_test = train_test_split(indices, test_size=0.2, shuffle=True,
                                                        random_state=0)

        train_X = padded_encoded_result[indices_train]
        train_Y = Y.iloc[indices_train]

        test_X = padded_encoded_result[indicies_test]
        test_Y = Y.iloc[indicies_test]

        print('\nWord2Vec info')
        print("vocab size=", vocab_size, "vector size=", NewsModel.wv.vector_size, "max len=", max_len, "min len=",
              min_len)

        # -------------------------------cnn 모델 설계

        embedding_layer = Embedding(vocab_size, NewsModel.wv.vector_size, weights=[embedding_matrix],
                                    input_length=max_len, trainable=False)  # word2vec 임베딩 결과 학습되어있음 true로 하면 파인튜닝


        CNN = Sequential()
        CNN.add(embedding_layer)
        CNN.add(Conv1D(filters=50, kernel_size=1, activation='relu'))  # 너비가 고정되어있으므로 1D kernel_size 가 높이
        CNN.add(GlobalMaxPool1D())
        CNN.add(Flatten())
        # CNN.add(Dense(30, activation='relu'))
        # CNN.add(Dense(10, activation='relu'))
        CNN.add(Dense(1))
        print(CNN.summary())

        # -----------------------------cnn 기반 학습
        epoch_num = 300
        CNN.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        history = CNN.fit(x=train_X, y=train_Y, epochs=epoch_num, verbose=0,
                          batch_size=32,
                          validation_data=(test_X, test_Y))

        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        epochs = range(1, epoch_num+1)
        plt.plot(epochs, loss_train, 'b-o', label='Training loss')
        plt.plot(epochs, loss_val, 'r-o', label='Valid loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

        y_pred = CNN.predict(test_X)

        r2 = r2_score(test_Y, y_pred)
        print("r_square score=", r2)


    def _preprocess_step_cls(self):
        """
        this is function preprocess step
        1) raw csv ->
        2) eliminate symbol stopword ->
        3) get y labeled data from pandas naver api ->
        4) return preprocessed csv
        :return:
        """
        tokenized_dict, date_dict, csv_co_list = self._get_token_and_company_name()

        y_data = self.y_labeler._ylaber_step("cls", date_dict, csv_co_list)

        not_coname_count = 0

        for idx, co in enumerate(csv_co_list):
            if co == False:
                not_coname_count = not_coname_count + 1
                del tokenized_dict[str(idx)]
        print(">> Y값이 도출되지 않은 공시 데이터 개수 =", not_coname_count)

        print(">> 유의미한 X 데이터의 개수 =", len(tokenized_dict), '유의미한 Y 데이터의 개수 =', len(y_data))


        print('하락의 개수:', y_data.count(0), '\n보합의 개수:', y_data.count(1), '\n상승의 개수:', y_data.count(2))


        lst = list(tokenized_dict.values())

        # -----------------------------데이터 불러오기

        x_y_data = pd.DataFrame({'X': lst, 'Y': y_data})

        zero_x_y_data = x_y_data[x_y_data['Y'] == 0]
        one_x_y_data = x_y_data[x_y_data['Y'] == 1]
        two_x_y_data = x_y_data[x_y_data['Y'] == 2]

        min_count = min(y_data.count(0), y_data.count(1), y_data.count(2))

        zero_x_y_data = zero_x_y_data.sample(frac=min_count / y_data.count(0), random_state=0)
        one_x_y_data = one_x_y_data.sample(frac=min_count / y_data.count(1), random_state=0)
        two_x_y_data = two_x_y_data.sample(frac=min_count / y_data.count(2), random_state=0)

        x_y_data = pd.concat([zero_x_y_data, one_x_y_data, two_x_y_data])
        print(">> 데이터 불균형 해결 이후 ")
        print(x_y_data['Y'].value_counts())

        print("Dataframe")
        print(x_y_data.head(3))

        result = list(x_y_data['X'])

        NewsModel = Word2Vec(sg=1, size=300, window=1, min_count=1, workers=-1)
        NewsModel.build_vocab(result)  # 단어 생성.
        NewsModel.train(result, total_examples=len(result), epochs=100)  # 학습

        # ----------------- Word2vec 으로부터 임베딩된 결과 추출 dict() 형태

        embedding_results = dict()

        for token in NewsModel.wv.vocab.keys():
            embedding_results[token] = NewsModel.wv[token]

        vocab_size = len(NewsModel.wv.vocab.keys()) + 1
        embedding_matrix = np.zeros((vocab_size, NewsModel.wv.vector_size))

        # ------------------------------- 단어의 정수화
        word_to_id = dict()
        id_to_word = dict()

        all_words = NewsModel.wv.vocab.keys()

        for word in all_words:
            if word not in word_to_id:
                new_id = len(word_to_id) + 1
                word_to_id[word] = new_id
                id_to_word[new_id] = word

        # print('\nword_to_id and id_to_word')
        # print(word_to_id)
        # print(id_to_word)

        # ----------------------- 정수화 된 단어 매트릭스에 임베딩 결과로 저장
        for word, i in word_to_id.items():
            embedding_matrix[i, :] = embedding_results[word]

        # -------------------CNN 기반 데이터 분석
        encoded_result = []

        for sent in result:
            temp_id = []
            for word in sent:
                temp_id.append(word_to_id[word])
            encoded_result.append(temp_id)



        max_len = max(len(encoded_email) for encoded_email in encoded_result)
        min_len = min(len(encoded_email) for encoded_email in encoded_result)
        # -----------------------------------padding
        padded_encoded_result = pad_sequences(encoded_result, padding='post')  # 앞 부분에다가 값을 넣어줌 maxlen 파라미터로 조정가능

        print('\n인코딩 결과 확인')
        print(encoded_result[0])
        print(padded_encoded_result[0].shape, padded_encoded_result[0])

        # -------------------------------------train, test 분리
        indices = np.arange(len(padded_encoded_result))
        Y = x_y_data['Y']
        indices_train, indicies_test = train_test_split(indices, test_size=0.2, shuffle=True, random_state=0,stratify=Y)


        train_X = padded_encoded_result[indices_train]
        train_Y = Y.iloc[indices_train]

        test_X = padded_encoded_result[indicies_test]
        test_Y = Y.iloc[indicies_test]

        print('\nWord2Vec info')
        print("vocab size=", vocab_size, "vector size=", NewsModel.wv.vector_size, "max len=", max_len, "min len=",
              min_len)

        #------------------------------- cnn 모델 설계

        embedding_layer = Embedding(vocab_size, NewsModel.wv.vector_size, weights=[embedding_matrix],
                                    input_length=max_len, trainable=False)  # word2vec 임베딩 결과 학습되어있음 true로 하면 파인튜닝

        """
        CNN 모델의 최적 파라미터를 학습해보기 위하여 파라미터가 될 수 있는 값들을 
        반복문으로 실행하여 최적의 파라미터를 도출해본 코드
        
        # filter_list = [10, 30, 50]
        # kernel_list = [1, 2]
        # conv_activation_list = ['relu', 'sigmoid']
        # optimizer_list = ['adam', 'sgd']
        # batch_size_list = [32, 64]
        # info_list = []
        # result_list = []
        # class_result_list = []
      
        # n = 0
        # for f in filter_list:
        #     for k in kernel_list:
        #         for c in conv_activation_list:
        #             for o in optimizer_list:
        #                 for b in batch_size_list:
        """

        CNN = Sequential()
        CNN.add(embedding_layer)
        CNN.add(Conv1D(filters=50, kernel_size=1, activation='relu'))  # 너비가 고정되어있으므로 1D kernel_size 가 높이
        CNN.add(GlobalMaxPool1D())
        CNN.add(Flatten())
        CNN.add(Dense(3, activation='softmax'))
        print(CNN.summary())

        # -----------------------------cnn 기반 학습
        epoch_num=100
        CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = CNN.fit(x=train_X, y=to_categorical(np.array(train_Y)), epochs=epoch_num, verbose=0, batch_size=32,
                          validation_data=(test_X, to_categorical(np.array(test_Y))))

        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        epochs = range(1, epoch_num+1)
        plt.plot(epochs, loss_train, 'b-o', label='Training loss')
        plt.plot(epochs, loss_val, 'r-o', label='Valid loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

        y_pred = CNN.predict(test_X)
        y_pred = np.argmax(y_pred, axis=1)

        print(y_pred, type(y_pred))

        print("0 =",np.count_nonzero(y_pred == 0),"1 =",np.count_nonzero(y_pred == 1),"2 =",np.count_nonzero(y_pred == 2))
        print(confusion_matrix(test_Y, y_pred))
        print(classification_report(test_Y, y_pred, target_names=['0', '1', '2']))

        """   
        CNN 모델의 최적 파라미터의 결과를 확인해보기 위하여 반복문으로 실행 결과를 저장하여 출력한 코드
         
        # info_list.append([f,k,c,o,b])
        # result_list.append(confusion_matrix(test_Y, predy))
        # class_result_list.append(classification_report(test_Y, predy, target_names=['0', '1', '2']))
        # print(confusion_matrix(test_Y, predy))
        # print(classification_report(test_Y, predy, target_names=['0', '1', '2']))
        #                     n=n+1
        #                     print(n,"번째 완료")
        # for a in range(len(result_list)):
        #     print("\n",info_list[a],"\n",result_list[a],"\n",class_result_list[a])
        """

if __name__ == '__main__':
    preprocessor = Preprocessor()
    preprocessor._preprocess_step_cls()
    #preprocessor._preprocess_step_reg()

# 로그 재 구성, 클래스 재구성, 필요없는거 지우기, 하이퍼 파라미터 재정립, 각 step 결과 print 해보기