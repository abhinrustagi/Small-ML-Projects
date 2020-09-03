import scrapy

class QuotesSpider(scrapy.Spider):
    # Name of the Spider should be in single quotes
    name = "quotesjs"
    
    # To make GET request
    def start_request(self):
        
        #Write URL in single quotes as well
        urls = ['http://quotes.toscrape.com/page/1/',
               'http://quotes.toscrape.com/page/2/']
        
        # Generator Function
        for url in urls:
            yield scrapy.Request(url=url, callback = self.parse)
    
    def parse(self, response):
        
        for q in response.css('div.quote'):
            text = q.css('span.text::text').get()
            author = q.css('small.author::text').get()
            tags = q.css('a.tag::text').getall()
            
            yield {
                'text' : text,
                'author' : author,
                'tags' : tags
            }