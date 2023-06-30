import scrapy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from SearchScraper.items import SearchscraperItem
from SearchScraper.cleanser import cleansing
import re
import pymysql


class SearchSpiderSpider(scrapy.Spider):
    name = "search_spider_daily"
    allowed_domains = ["search.naver.com", "news.naver.com"]
    start_urls = ["http://search.naver.com/"]
    # 자꾸 m.serch.naver.com으로 redirection 되길래 추가
    handle_httpstatus_list = [302, 403]

    keywords = ['삼성전자','현대차','삼성생명','셀트리온','포스코']
    keyword_generator = (keyword for keyword in keywords)
    keyword = next(keyword_generator, None)
    
    stock_ids = {'삼성전자': 1, '현대차': 2, '삼성생명': 3, '셀트리온': 4, '포스코': 5}

    URL_FORMAT = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={}&sort=0&photo=0&field=0&pd=3&ds={}&de={}&start={}'

    def start_requests(self):
        nowdate = datetime.now().date()
        # 최근 크롤링 날짜 조회 -> 없으면 오늘부터
        try:
            with open('./latest_date', 'r') as f:
                lastdate = pd.to_datetime(f.read().strip()).date()
        except:
            lastdate = nowdate
            pass
            
        # 크롤링 날짜 업데이트
        with open('./latest_date', 'w') as f:
            f.write(nowdate.strftime('%Y%m%d'))

        # 마지막 날짜부터 오늘 날짜까지 크롤링하기 위해 dates 생성
        dates = pd.date_range(end=nowdate, start=lastdate).strftime('%Y%m%d').tolist()

        previous_url = {}
        # 서브카테고리별 가장 최근에 가져온 url 조회
        try:
            with open('./latest_url.txt', 'r') as f:
                for line in f.readlines():
                    previous_url[line.split(' ')[0]] = line.split(' ')[-1].strip()
                    print(previous_url)
        except:
            pass    
        # 파일 초기화
        with open('./latest_url.txt', 'w') as f:
            f.write('')


        for keyword in self.keywords:
            previous_url.setdefault(keyword, '')
            for date in dates:
                target_url = self.URL_FORMAT.format(
                                keyword, 
                                date, date,
                                1
                                )
                yield scrapy.Request(url=target_url, callback=self.parse_url, meta={
                    'page':1, 'keyword':keyword, 'previous_url':previous_url[keyword], 'latest_url':'', 'stop':0
                    })

    def parse_url(self, response):
        if response.meta.pop('stop'):
            self.keyword = next(self.keyword_generator, None)
            if self.keyword is not None:
                yield scrapy.Request(url=self.start_urls[0], callback=self.start_requests)
            return

        urls = response.xpath('//*[@class="bx"]/div/div/div[1]/div[2]/a[2]/@href').extract()
        stop = response.meta['previous_url'] in urls

        if response.meta['latest_url'] == '' and len(urls) >= 1:
            response.meta['latest_url'] = urls[0]
            with open('./latest_url.txt', 'a') as f:
                f.write(response.meta['keyword']+' '+urls[0]+'\n')

        # 검색결과 없다는 칸이 있으면 중지
        if response.xpath('//*[@class="api_noresult_wrap"]').extract() != []:
            self.keyword = next(self.keyword_generator, None)
            if self.keyword is not None:
                yield scrapy.Request(url=self.start_urls[0], callback=self.start_requests)
            return
        
        for url in urls:
            if url == response.meta['previous_url']:
                self.keyword = next(self.keyword_generator, None)
                if self.keyword is not None:
                    yield scrapy.Request(url=self.start_urls[0], callback=self.start_requests)
                return
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
        yield scrapy.Request(url=target_url, callback=self.parse_url, meta={**response.meta, 'page':page, 'stop':stop})

    def parse(self, response):
        item = SearchscraperItem()
        item['company_id'] = self.stock_ids[response.meta['keyword']]
        item['company_name'] = response.meta['keyword']
        item['title'] = response.xpath('//*[@id="title_area"]/span/text()').extract()
        item['content'] = cleansing(' '.join(response.xpath('//*[@id="dic_area"]//text()').extract()))
        item['date'] = response.css('.media_end_head_info_datestamp_time::attr(data-date-time)').get()
        item['url'] = response.url
        item['sentiment'] = 'unknown'
        item['score'] = -1        
        yield item
