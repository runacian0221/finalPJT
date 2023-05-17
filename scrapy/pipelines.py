# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import csv
from scrapy.exporters import CsvItemExporter

class SearchscraperPipeline:
    def open_spider(self, spider):
        self.files = {}

    def close_spider(self, spider):
        for exporter in self.files.values():
            exporter.finish_exporting()
            exporter.file.close()

    def process_item(self, item, spider):
        keyword = item['keyword']
        if keyword not in self.files:
            f = open(f'{keyword}data.csv', 'wb')
            exporter = CsvItemExporter(f)
            exporter.start_exporting()
            self.files[keyword] = exporter
        exporter = self.files[keyword]
        exporter.export_item(item)
        return item
