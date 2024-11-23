# Scrapy settings for indeed project
BOT_NAME = 'indeed'

SPIDER_MODULES = ['indeed.spiders']
NEWSPIDER_MODULE = 'indeed.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# ScrapeOps API Key
SCRAPEOPS_API_KEY = '3eafda05-2641-4d20-896c-153390b9b9b0'  # 替換為你的 ScrapeOps API 金鑰

# Enable ScrapeOps Proxy
SCRAPEOPS_PROXY_ENABLED = True

# Add In The ScrapeOps Monitoring Extension
EXTENSIONS = {
    'scrapeops_scrapy.extension.ScrapeOpsMonitor': 500,
}

DOWNLOADER_MIDDLEWARES = {
    # ScrapeOps Monitor
    'scrapeops_scrapy.middleware.retry.RetryMiddleware': 550,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,

    # Proxy Middleware
    'scrapeops_scrapy.middleware.proxy.ScrapeOpsProxyMiddleware': 725,
}

# Max Concurrency On ScrapeOps Proxy Free Plan is 1 thread
CONCURRENT_REQUESTS = 1

# Set Logging Level
LOG_LEVEL = "INFO"
