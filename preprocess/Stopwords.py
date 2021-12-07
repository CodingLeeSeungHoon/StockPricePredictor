from konlpy.tag import Kkma
import re

class Stopwords:


    def __init__(self):
        self.delete_stopwords = []
        self.replace_stopwords = []

        self.stopwords_file = open("stopwords.txt", 'r', encoding='UTF8')
        self.stopwords = []
        self.stopwords_file_text = self.stopwords_file.readlines()
        for words in self.stopwords_file_text:
            words = words.strip()
            self.stopwords.append(words)

    def _get_delete_stopwords(self, title_str):
        """
        title_str이 들어와서 stopwords를 제거하고
        토큰화를 해서
        제거한 문장 토큰 리스트 형태로 반환.
        """
        result_tokens = []
        kkma = Kkma()
        title_tokens = kkma.morphs(title_str)
        for i in title_tokens:
            if i not in self.stopwords:
                result_tokens.append(i)
        # print(result_tokens)
        return result_tokens

    def _get_replace_stopwords(self, title_str):
        """
        title_str을 읽어 들여서
        replace 될 단어를 찾아 replace를 해주고 해당 문장을 반환해준다.
        @return : string
        """
        
        replace_dict = {'比': '대비', '↓': '감소', '↑': '증가', 'Q': '분기', '中': '중국', '日': '일본', '美': '미국', '韓': '한국',
                            '北': '북한', '$': '달러', '￥': '엔', '€': '유로'}
        replace_table = title_str.maketrans(replace_dict)
        title_str = re.sub('[0-9]', '', title_str)
        title_str = title_str.translate(replace_table)
        return title_str
