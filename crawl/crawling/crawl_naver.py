# crawling data from naver finacnce
import requests
import json
from bs4 import BeautifulSoup


class Crawl:
  def __init__(self):
    self.base_url = "https://finance.naver.com/news/news_list.naver"
    self.market_notice_url = "https://finance.naver.com/news/market_notice.naver"

  def news_crawl(self, date):
    '''
    뉴스포커스 > 공시 메모 크롤링
    '''
    news_data = self.openJson()

    disclosure = "?mode=LSS3D&section_id=101&section_id2=258&section_id3=406"
    url = self.base_url + disclosure

    page_num = 1
    before_title = ''

    while True:
      page_url = url + f"&date={date}&page={page_num}"
      html = requests.get(page_url)

      if html.status_code == 200:
        html = html.text
        soup = BeautifulSoup(html, 'html.parser')

        newsList = soup.find('ul', {'class': 'realtimeNewsList'})
        subject = newsList.find_all(class_="articleSubject")
        summary = newsList.find_all(class_="articleSummary")

        if len(subject) == 0:
          break

        for i in range(len(subject)):
          wdate = summary[i].find("span", class_="wdate").text
          if int(wdate[5:7] + wdate[8:10]) < int(date[4:]):
            break

          title = subject[i].find("a").text
          if before_title != title:
            news_data.append({"title": title, "date":wdate})

          before_title = title

      page_num += 1

    self.writeJson(news_data)
  
  def market_notice(self, date):
    '''
    투자정보 > 공시정보 크롤링.
    '''
    news_data = self.openJson()

    page = 1
    while True:
      page_url = self.market_notice_url + f"?&page={page}"
      html = requests.get(page_url)

      if html.status_code == 200:
        html = html.text
        soup = BeautifulSoup(html, 'html.parser')

        disclosureList = soup.find_all('tr')
        for disclosure in disclosureList:
          try:
            title = disclosure.find("td", class_="publicSubject").text
            title = title.strip("\n")

            wdate = disclosure.find("td", class_="wdate").text
            w = wdate.replace('.','')
            w = '20' + w[:6]
            if int(w) < int(date):
              break
            wdate = wdate.replace('.','-')
            news_data.append({'title':title, 'date':'20'+wdate })
          except (AttributeError, ModuleNotFoundError):
            pass

      if int(w) < int(date):
        break
      page += 1
    
    self.writeJson(news_data)


  def writeJson(self, data):
    with open("newsdata/news_data.json", "w", encoding='UTF-8') as json_file:
      json.dump(data, json_file, indent=4, ensure_ascii=False)

  def openJson(self):
    with open("newsdata/news_data.json", 'rt', encoding='UTF-8') as f:
      json_data = json.load(f)
    return json_data
