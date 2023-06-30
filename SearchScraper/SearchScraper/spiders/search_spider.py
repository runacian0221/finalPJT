import scrapy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from SearchScraper.items import SearchscraperItem
from SearchScraper.cleanser import cleansing
import re

class SearchSpiderSpider(scrapy.Spider):
    name = "search_spider"
    allowed_domains = ["search.naver.com", "news.naver.com"]
    start_urls = ["http://search.naver.com/"]
    # 자꾸 m.serch.naver.com으로 redirection 되길래 추가
    handle_httpstatus_list = [302, 403]

    keywords = ['셀트리온','삼성생명','현대차','포스코']
    keyword_generator = (keyword for keyword in keywords)
    keyword = next(keyword_generator, None)
    
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2019, 12, 31)
    

    URL_FORMAT = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={}&sort=0&photo=0&field=0&pd=3&ds={}&de={}&start={}'

    def start_requests(self):
        dates = pd.date_range(end=self.end_date, start=self.start_date).strftime('%Y.%m.%d').tolist()
        for keyword in self.keywords:
            for date in dates:
                target_url = self.URL_FORMAT.format(
                                keyword, 
                                date, date,
                                1
                                )
                yield scrapy.Request(url=target_url, callback=self.parse_url, meta={'page':1, 'keyword':keyword})

    def parse_url(self, response):
        # 검색결과 없다는 칸이 있으면 중지
        if response.xpath('//*[@class="api_noresult_wrap"]').extract() != []:
            self.keyword = next(self.keyword_generator, None)
            if self.keyword is not None:
                yield scrapy.Request(url=self.start_urls[0], callback=self.start_requests)
            return
        
        urls = response.xpath('//*[@class="bx"]/div/div/div[1]/div[2]/a[2]/@href').extract()

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, meta={**response.meta})

        with open('./test_url_list.txt', 'a') as f:
            for url in urls:
                f.write(url+'\n')

        page = response.meta.pop('page') + 1
        # 400페이지 넘긴다면 중지
        if page > 400:
            self.keyword = next(self.keyword_generator, None)
            if self.keyword is not None:
                yield scrapy.Request(url=self.start_urls[0], callback=self.start_requests)
            return
        
        target_url = re.sub('start\=\d+', f'start={10*(page-1)+1}', response.url)
        yield scrapy.Request(url=target_url, callback=self.parse_url, meta={**response.meta, 'page':page})

    def parse(self, response):
        item = SearchscraperItem()
        item['keyword'] = response.meta['keyword']
        item['title'] = response.xpath('//*[@id="title_area"]/span/text()').extract()
        item['content'] = cleansing(' '.join(response.xpath('//*[@id="dic_area"]//text()').extract()))
        item['writed_at'] = response.css('.media_end_head_info_datestamp_time::attr(data-date-time)').get()
        item['url'] = response.url
        yield item
