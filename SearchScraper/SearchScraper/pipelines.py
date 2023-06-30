# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import csv
from scrapy.exporters import CsvItemExporter
import pandas as pd
import re
from scrapy.exceptions import DropItem
import pymysql
import sys
sys.path.append('/home/ubuntu/workspace')
from DB.insert_select_db import Database
from DB.realtime_insert_data_processor import RealTimeDataProcessor
from transformers import pipeline
from collections import Counter
from nltk.tokenize import sent_tokenize
import ast

class DBPipeline:
    def open_spider(self, spider):
        # with open("/home/ubuntu/workspace/DB/config.txt", "r") as file:
        #     exec(file.read())
        configs = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': 'password',
            'database': 'finalpjt',
            'charset': 'utf8mb4',
        }
        self.db = Database(configs)
        self.processor = RealTimeDataProcessor(self.db)

        self.classifier = pipeline('sentiment-analysis', model='snunlp/KR-FinBert-SC')
        self.df = pd.DataFrame(columns=['company_id', 'company_name', 'title', 'content', 'url', 'date', 'sentiment', 'score'])

    def close_spider(self, spider):
        
        print(self.df.info())
        print(self.df.head())
        
        if len(self.df) > 0:
            self.processor.process_and_insert_news_analysis_data(self.df,
                required_columns = ['company_id','company_name','title','content','url','date','sentiment','score'],
                table_name='news_analysis')

    def process_item(self, item, spider):
        if any(v is None for v in item.values()):
            raise DropItem("Missing value in %s" % item)

        if len(item.get('content', '')) < 100:
            raise DropItem("Content length is less than 100 characters")

        if item['company_name'] not in item['title']:
            raise DropItem('Item dropped: company_name not found in title')

        contents = re.sub(r'[^가-힣a-zA-Z0-9 ]', ' ', item['content'])
        contents = re.sub(r' +', ' ', contents)
        sentences = [sent[:512] for sent in sent_tokenize(contents)]
        result = self.classifier(sentences)
        label_counts = Counter(d['label'] for d in result if d['score'] > 0.5)
        if label_counts == Counter():
            item['sentiment'] = 'neutral'
            item['score'] = 0
        elif label_counts.most_common(1) == label_counts.most_common(2):
            most_label = label_counts.most_common(1)[0][0]
            item['sentiment'] = most_label
            item['score'] = 0 if most_label=='neutral' else 1
        elif label_counts.most_common(2)[0][1] == label_counts.most_common(2)[1][1]:
            item['sentiment'] = 'neutral'
            item['score'] = 0
        else:
            most_label = label_counts.most_common(1)[0][0]
            most_label_count = label_counts[most_label]
            item['sentiment'] = most_label
            item['score'] = 0 if most_label=='neutral' else np.round(most_label_count/sum(label_counts.values()), 3)

        item_dict = dict(item)
        self.df = self.df.append(item_dict, ignore_index=True)

        return item


class SearchscraperPipeline:
    def open_spider(self, spider):
        self.files = {}

    def close_spider(self, spider):
        for exporter in self.files.values():
            exporter.finish_exporting()
            exporter.file.close()

    def process_item(self, item, spider):
        keyword = item['keyword']

        # 데이터가 비어있을때 제외
        if any(v is None for v in item.values()):
            raise DropItem("Missing value in %s" % item)

        # 데이터가 100자 미만일때 제외
        if len(item.get('content', '')) < 100:
            raise DropItem("Content length is less than 100 characters")

        if keyword not in self.files:
            f = open(f'{keyword}data.csv', 'wb')
            exporter = CsvItemExporter(f)
            exporter.start_exporting()
            self.files[keyword] = exporter
        exporter = self.files[keyword]
        exporter.export_item(item)
