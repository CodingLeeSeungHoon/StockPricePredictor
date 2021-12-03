class Stopwords:
    def __init__(self):
        self.delete_stopwords = []
        self.replace_stopwords = []

        """ init stopwords list """
        self._init_delete_stopwords()
        self._init_replace_stopwords()

        stopwords_file = open("stopwords.txt", 'r', encoding='UTF8')
        stopwords = []
        stopwords_file_text = stopwords_file.readlines()
        for words in stopwords_file_text:
            words = words.strip()
            stopwords.append(words)

    def _init_delete_stopwords(self, title_str):
        result_tokens = []
        title_tokens = kkma.morphs(title_str)
        for i in title_tokens:
            if i not in self.stopwords:
                result_tokens.append(i)
        return result_tokens

    def _init_replace_stopwords(self, title_str):
        replace_dict = {'比': '대비', '↓': '감소', '↑': '증가', 'Q': '분기', '中': '중국', '日': '일본', '美': '미국', '韓': '한국',
                        '北': '북한', '$': '달러', '￥': '엔', '€': '유로'}
        replace_table = title_str.maketrans(replace_dict)
        title_str = re.sub('[0-9]', '', title_str)
        title_str = title_str.translate(replace_table)
        return title_str
