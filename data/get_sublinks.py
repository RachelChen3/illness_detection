# =================================
# json remove the duplicated items
# =================================
import json
dire_list = []
with open('timesofindia.json','r+') as json_file:
    ill_list = json.load(json_file)
    json_file.seek(0)
    unique = {each['content'] : each for each in ill_list }.values()
    json.dump(list(unique), json_file)
    json_file.truncate()

# ============================================
# get the main page and substract the sublinks
# ============================================
import json
import os
from pathlib import Path

dire_list = []
url_list = []
with open('timesofindia.json') as json_file:
    ill_list = json.load(json_file)
    for illness in ill_list:
        dire_list.append(illness['dire'])
        url_list.append("https://timesofindia.indiatimes.com/"+illness['link'])
ill_dire_list = list(set(dire_list))

import scrapy

class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    start_urls = url_list

    def parse(self, response):
        temp = [x for x in response.url.split("/") if x]
        if len(temp)>5 and temp[5] in ill_dire_list:
            # create directory
            if not os.path.exists('./timesofindia/'+ temp[5]):
                os.makedirs('./timesofindia/'+ temp[5])
            filename = './timesofindia/'+temp[5] +"/" +"_".join(temp[5:]).split(".")[0]+ '.txt'
            if Path(filename).exists():
                pass
            else:
                content_list  = response.css('div.tabcontainer').xpath('.//text()').extract()
                with open(filename, 'a+') as f:
                     _ = [f.write(ele+'\n') for ele in content_list]

                sub_link_list = response.css('ul.ulshadow').xpath('li/@data-url').getall()
                yield  {
                    'sublinklist': sub_link_list
                }
