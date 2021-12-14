from crawling.crawl_naver import Crawl
from datetime import datetime, timedelta

def get_date(date):
  """
  :return : list
  2021/10/07 이전 공시 정보를 크롤링합니다.
  받고자 하는 날짜들을 date_lst로 반환합니다.
  """
  date_lst = []
  first_date = datetime(2021, 10, 7)
  for i in range(date):
    before_day = first_date - timedelta(i)
    date_lst.append(before_day.strftime("%Y%m%d"))
  
  return date_lst


def crawl_stock_data():
  """
  get_date() 파라미터로 크롤링 하고 싶은 일 수를 쓰면 됩니다.
  news_crawl 함수는 뉴스 > 공시 메모 부분을 크롤링합니다.
  """
  c = Crawl()
  date_lst = get_date(5)

  for date in date_lst:
    c.news_crawl(date)
