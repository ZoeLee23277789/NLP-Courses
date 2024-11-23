import scrapy
from scrapy.crawler import CrawlerProcess

class ProxyTestSpider(scrapy.Spider):
    name = "proxy_test"

    def start_requests(self):
        yield scrapy.Request(
            url="http://httpbin.org/ip",
            callback=self.parse,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"}
        )

    def parse(self, response):
        self.logger.info(f"Proxy response: {response.text}")

if __name__ == "__main__":
    process = CrawlerProcess(settings={
        "SCRAPEOPS_API_KEY": "3eafda05-2641-4d20-896c-153390b9b9b0",  # 替換成你的 API 金鑰
        "SCRAPEOPS_PROXY_ENABLED": True,
        "DOWNLOADER_MIDDLEWARES": {
            'scrapeops_scrapy.middleware.proxy.ScrapeOpsProxyMiddleware': 725,
        },
        "LOG_LEVEL": "INFO",
    })

    process.crawl(ProxyTestSpider)
    process.start()
