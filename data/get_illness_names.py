# -*- coding: utf-8 -*-
# =====================
# get all illness names
# =====================
import scrapy
class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    start_urls = [
        'https://timesofindia.indiatimes.com/life-style/health-fitness/health-a-z'
    ]
    def parse(self, response):
        for quote in response.css('div.alphaContentContainer').xpath('div'):
            for lis in quote.css('ul.diseas_listing').xpath('li'):
                link = lis.xpath('a/@href').get()
                yield {
                    'content': lis.xpath('a/span/text()').get(),
                    'link':link,
                    'dire':[x for x in link.split("/") if x][3]
                }
