from scrapy_selenium import SeleniumRequest
from scrapy.spiders import Spider


class IndeedJobSpider(Spider):
    name = "indeed_jobs"

    def start_requests(self):
        # Indeed 搜索页 URL
        url = "https://www.indeed.com/jobs?q=software+engineer&l=California"

        # 使用 SeleniumRequest 发起请求
        yield SeleniumRequest(
            url=url,
            callback=self.parse,
            wait_time=3,  # 等待页面加载时间
            screenshot=False,  # 可设置为 True 调试，获取截图
        )

    def parse(self, response):
        # 使用 Selenium 抓取返回的内容
        self.logger.info(f"Response URL: {response.url}")
        
        # 提取职位标题
        job_titles = response.css('.jobTitle > span::text').getall()
        for title in job_titles:
            yield {"Job Title": title}

        # 示例：打印部分 HTML 内容
        self.logger.info(response.text[:500])
