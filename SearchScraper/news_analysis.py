import os

command = "scrapy crawl search_spider_daily"

output_file = "/home/ubuntu/workspace/SearchScraper/analysis_log.txt"

os.system(f"{command} > {output_file}")
