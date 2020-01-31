# ============================
# get all pages from subpages
# ============================
import json
import os
import shutil
from pathlib import Path

dire_list = []
url_list = []
with open('sublink.json') as json_file:
    ill_list = json.load(json_file)
    for illness in ill_list:
        url_list += ["https://timesofindia.indiatimes.com/"+ele for ele in illness['sublinklist']]

import scrapy

class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    start_urls = url_list

    def parse(self, response):
        temp = [x for x in response.url.split("/") if x]
        if len(temp)>5 and os.path.exists('./timesofindia/'+ temp[5]):

            filename = './timesofindia/'+temp[5] +"/" +"_".join(temp[5:]).split(".")[0]+ '.txt'
            if Path(filename).exists():
                pass
            else:
                content_list  = response.css('div.tabcontainer').xpath('.//text()').extract()
                with open(filename, 'a+') as f:
                     _ = [f.write(ele+'\n') for ele in content_list]
