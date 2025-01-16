from scrapy.crawler import CrawlerProcess
from scrapy import Spider

class BranchesSpider(Spider):
    name = "branches"
    allowed_domains = ["service.standardchartered.com.tw"]
    start_urls = [
        "https://service.standardchartered.com.tw/location_finder/branch_table/Index"
    ]

    def parse(self, response):
        rows = response.xpath('//table[contains(@class, "branch_table")]/tbody/tr')
        for row in rows:
            yield {
                "branch_name": row.xpath('td[1]/text()').get(default="").strip(),
                "address": row.xpath('td[2]/text()').get(default="").strip(),
                "phone": row.xpath('td[3]/text()').get(default="").strip(),
            }

process = CrawlerProcess({
    'FEED_FORMAT': 'json',
    'FEED_URI': 'branches.json',
})

process.crawl(BranchesSpider)
process.start()
