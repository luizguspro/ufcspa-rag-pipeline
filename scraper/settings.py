"""
Configurações do Scrapy para o projeto UFCSPA Spider.
"""

# Nome do bot
BOT_NAME = 'ufcspa_scraper'

# Módulos do spider
SPIDER_MODULES = ['scraper']
NEWSPIDER_MODULE = 'scraper'

# Obedecer robots.txt
ROBOTSTXT_OBEY = True

# Configurações de concorrência
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 8
DOWNLOAD_DELAY = 0.5

# Configurações de User-Agent
USER_AGENT = 'ufcspa_scraper (+http://www.yourdomain.com)'

# Configurações de cookies
COOKIES_ENABLED = False

# Configurações de middleware
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
    'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
}

# Configurações de pipeline
ITEM_PIPELINES = {
    # Adicione pipelines customizados aqui se necessário
}

# Configurações de retry
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Configurações de timeout
DOWNLOAD_TIMEOUT = 30

# Configurações de cache HTTP (desabilitado por padrão)
HTTPCACHE_ENABLED = False
HTTPCACHE_EXPIRATION_SECS = 0
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [503, 500, 400, 403, 404]

# Configurações de log
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(levelname)s: %(message)s'

# Configurações de largura de banda
DOWNLOAD_DELAY = 0.5
RANDOMIZE_DOWNLOAD_DELAY = True
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 8.0
AUTOTHROTTLE_DEBUG = False

# Respeitar meta tags
ROBOTSTXT_OBEY = True

# Configurações de memória
CONCURRENT_ITEMS = 100

# Configurações de DNS
DNSCACHE_ENABLED = True
DNSCACHE_SIZE = 10000
DNS_TIMEOUT = 60