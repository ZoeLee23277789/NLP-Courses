# Scrapy settings for indeed_scraper project

BOT_NAME = "indeed_scraper"

SPIDER_MODULES = ["indeed_scraper.spiders"]
NEWSPIDER_MODULE = "indeed_scraper.spiders"

# 使用真實的用戶代理來模擬瀏覽器
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0'

# 防止被檢測為爬蟲
DOWNLOAD_DELAY = 2
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 10
RANDOMIZE_DOWNLOAD_DELAY = True

# Selenium Middleware 設置
DOWNLOADER_MIDDLEWARES = {
    "scrapy_selenium.SeleniumMiddleware": 800,
}

# Selenium 驅動配置
SELENIUM_DRIVER_NAME = "chrome"
SELENIUM_DRIVER_EXECUTABLE_PATH = r"C:\Program Files\Google\Chrome\Application\chromedriver.exe"  # 修改為你的 ChromeDriver 路徑
SELENIUM_DRIVER_ARGUMENTS = ["--headless", "--no-sandbox", "--disable-gpu"]  # 無頭模式運行

ROBOTSTXT_OBEY = False
COOKIES_ENABLED = False
FEED_EXPORT_ENCODING = "utf-8"

DEFAULT_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}
