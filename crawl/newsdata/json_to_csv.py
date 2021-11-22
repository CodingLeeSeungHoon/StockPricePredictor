import csv
import json


with open("news_data.json", 'rt', encoding='UTF-8', newline="") as input_file, \
  open("news_data.csv", "w", encoding='UTF-8', newline="") as output_file:

  data = json.load(input_file)
  print(data)
  f = csv.writer(output_file)

  f.writerow(['title', 'date'])

  for datum in data:
    f.writerow([datum["title"], datum["date"]])
  
    
