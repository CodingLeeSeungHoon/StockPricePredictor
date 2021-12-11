from crawling.crawl_naver import Crawl
from datetime import datetime, timedelta

# 원하는 일수 만큼의 날을 date_lst에 저장.
def get_date(date):
  date_lst = []
  first_date = datetime(2021, 12, 7) # 21/12/7 이전 데이터를 받아옴.
  for i in range(date):
    before_day = first_date - timedelta(i)
    date_lst.append(before_day.strftime("%Y%m%d"))
  
  return date_lst


def crawl_stock_data():
  c = Crawl()
  # 받고자 하는 뉴스의 일 수를 get_date함수의 매개변수로 넣어준다.
  date_lst = get_date(500)

  for date in date_lst:
    c.news_crawl(date)
  
  # 투자정보 > 공시정보 부분. 이번 프로젝트에는 필요 없어짐.
  # start_date = date_lst[-1]
  # c.market_notice(start_date)