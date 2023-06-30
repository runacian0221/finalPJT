# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class SearchscraperItem(scrapy.Item):
    # define the fields for your item here like:
    company_id = scrapy.Field()
    company_name = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    url = scrapy.Field()
    date = scrapy.Field()
    sentiment = scrapy.Field()
    score = scrapy.Field()
    pass
