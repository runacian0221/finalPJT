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
