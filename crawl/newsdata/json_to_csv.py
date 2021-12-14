import csv
import json

if __name__ == '__main__':
  """
  크롤링한 json데이터를 new_data.csv로 변환한다.
  """
  with open("news_data.json", 'rt', encoding='UTF-8', newline="") as input_file, \
    open("news_data.csv", "w", encoding='UTF-8', newline="") as output_file:

    data = json.load(input_file)
    f = csv.writer(output_file)

    f.writerow(['title', 'date'])

    for datum in data:
      f.writerow([datum["title"], datum["date"]])
    
    
