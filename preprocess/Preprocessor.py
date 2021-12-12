import sys
from os import path
sys.path.append(path.dirname(__file__))


from nltk.text import TokenSearcher
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

#from preprocess import Stopwords, Vectorization, Ylabeler
from Stopwords import Stopwords
from Vectorization import Vectorization
from Ylabeler import Ylabeler

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

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

    def _get_token_and_company_name(self):
        """
        공시 1차적으로 약한 토큰화 진행, 회사 이름 뽑아내기
        회사 이름 없거나 2개 이상으로 뽑히면 False로 처리
        :return: 공시 딕셔너리, 공시에서 언급된 회사이름 리스트
        
        뉴스에 맞는 날짜도 가져와야 해서 날짜 딕셔너리 추가.
        2021.10.10, 13
        """
        # print(self.raw_csv['title'])

        # 기업명을 얻기 위한 딕셔너리
        title_dict_for_getting_company = {}

        # 최종적으로 replace와 stopwords 제거된 문장을 얻는 dictionary
        title_dict = {}

        # 날짜 
        date_list = []

        # csv로부터 구한 회사 이름
        company_name = []
        
        for idx, row in self.raw_csv.iterrows():
            print(idx, "번째 기업명 추출")
            title_dict_for_getting_company[str(idx)] = word_tokenize(row['title'])
            date_list.append(word_tokenize(row['date'])[0])

        for t in title_dict_for_getting_company.values():
            company_name.append(list(set(t) & self.co_name_list) if len(set(t) & self.co_name_list) == 1 else False)

        for idx, row in self.raw_csv.iterrows():
            # 기업명 뺴기
            print(idx,"번째 기업명빼기")
            if company_name[idx] != False:
                row['title'] = row['title'].replace(company_name[idx][0], '')

            title_dict[str(idx)] = self.stopwords._get_delete_stopwords(self.stopwords._get_replace_stopwords(row['title']))
        return title_dict, date_list, company_name

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
        tokenized_dict, date_dict, csv_co_list = self._get_token_and_company_name()

        y_data = self.y_labeler._ylaber_step("cls", date_dict, csv_co_list)

        for idx, co in enumerate(csv_co_list):
            if co == False:
                del tokenized_dict[str(idx)]
         
        print(len(tokenized_dict))
        print(len(y_data))
        print(f'하락개수: {y_data.count(0)}')
        print(f'보합개수: {y_data.count(1)}')
        print(f'상승개수: {y_data.count(2)}')

        # with open('temp_y.txt', 'w', encoding='utf-8') as f:
        #     for data in y_data:
        #         f.write(str(data)+'\n')

        # df = pd.DataFrame(tokenized_dict.values())
        # print(df)
        lst = list(tokenized_dict.values())
        #model = Word2Vec(sg=0, size=100, window=3, min_count=1, workers=4)
        #model.build_vocab(sentences=lst)
        #model.train(sentences=lst, total_examples=len(lst), epochs=10)

        #print(model.wv.vectors.shape)


        # model_result = model.wv.most_similar("증가", topn=10)
        # print(model_result)
 
        # for t in tokenized_dict.values():
        #     print(t) 

        # -----------------------------데이터 불러오기

        import tensorflow
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

        x_y_data = pd.DataFrame({'X': lst, 'Y': y_data})

        result = list(x_y_data['X'])
        TransferedModel = Word2Vec(size=300, min_count=1, workers=-1)
        TransferedModel.build_vocab(result)  # 단어 생성.
        TransferedModel.train(result, total_examples=len(result), epochs=100)  # 학습

        # -----------------Word2vec으로부터 임베딩된 결과 추출 dict() 형태

        embedding_results = dict()

        for token in TransferedModel.wv.vocab.keys():
            embedding_results[token] = TransferedModel.wv[token]

        vocab_size = len(TransferedModel.wv.vocab.keys()) + 1
        embedding_matrix = np.zeros((vocab_size, TransferedModel.wv.vector_size))

        # ------------------------------- 단어의 정수화
        word_to_id = dict()
        id_to_word = dict()

        all_words = TransferedModel.wv.vocab.keys()

        for word in all_words:
            if word not in word_to_id:
                new_id = len(word_to_id) + 1
                word_to_id[word] = new_id
                id_to_word[new_id] = word

        # print(word_to_id['분기'])

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

        # print(encoded_result[3],encoded_result[4],encoded_result[8],encoded_result[9])

        max_len = max(len(encoded_email) for encoded_email in encoded_result)

        # -----------------------------------padding
        padded_encoded_result = pad_sequences(encoded_result, padding='post')  # 앞 부분에다가 값을 넣어줌 maxlen 파라미터로 조정가능
        # print(padded_encoded_result[0].shape,padded_encoded_result[0])

        # -------------------------------------train, test 분리
        indices = np.arange(len(padded_encoded_result))
        Y = x_y_data['Y']
        indices_train, indicies_test = train_test_split(indices, test_size=0.2, shuffle=True, random_state=0,stratify=Y)
        #print(indices_train,len(indices_train),indicies_test,len(indicies_test))

        train_X = padded_encoded_result[indices_train]
        train_Y = Y.iloc[indices_train]

        test_X = padded_encoded_result[indicies_test]
        test_Y = Y.iloc[indicies_test]
        #print(vocab_size,TransferedModel.wv.vector_size)

        #-------------------------------cnn 모델 설계
        embedding_layer = Embedding(vocab_size, TransferedModel.wv.vector_size, weights=[embedding_matrix],
                                    input_length=max_len, trainable=False)  # word2vec 임베딩 결과 학습되어있음 true로 하면 파인튜닝

        CNN = Sequential()
        CNN.add(embedding_layer)
        CNN.add(Conv1D(filters=50, kernel_size=1, activation='relu'))  # 너비가 고정되어있으므로 1D kernel_size 가 높이
        CNN.add(GlobalMaxPool1D())
        CNN.add(Flatten())
        CNN.add(Dense(3, activation='softmax'))
        print(CNN.summary())

        # -----------------------------cnn 기반 학습
        CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = CNN.fit(x=train_X, y=to_categorical(np.array(train_Y)), epochs=100, verbose=1, batch_size=32,
                          validation_data=(test_X, to_categorical(np.array(test_Y))))

        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        epochs = range(1, 101)
        plt.plot(epochs, loss_train, 'b-o', label='Training loss')
        plt.plot(epochs, loss_val, 'r-o', label='Valid loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

        predy = CNN.predict(test_X)
        predy = np.argmax(predy, axis=1)

        print(predy, type(predy))

        print("0 =",np.count_nonzero(predy == 1),"1 =",np.count_nonzero(predy == 1),"2 =",np.count_nonzero(predy == 2))
        print(confusion_matrix(test_Y, predy))
        print(classification_report(test_Y, predy, target_names=['0', '1', '2']))


if __name__ == '__main__':
    preprocessor = Preprocessor()
    #preprocessor._get_token_and_company_name()
    preprocessor._preprocess_step()

    