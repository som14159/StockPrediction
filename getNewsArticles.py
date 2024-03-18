from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key='d7f4f06cb623498587f38f5107c19bf5')
all_articles = newsapi.get_everything(q='Apple', language='en', sort_by = 'publishedAt')

no_of_articles = 0

with open('NewsArticles.txt', 'w') as file:
    for article in all_articles['articles']:
        file.write('Title: ' + article['title'] + '\n')
        if article['description'] is not None:
            file.write('Description: ' + article['description'] + '\n')
        else:
            file.write('Description: N/A\n')
        if article['content'] is not None:
            file.write('Content: ' + article['content'] + '\n')
        else:
            file.write('Content: N/A\n')
        file.write('Source: ' + article['source']['name'] + '\n')
        file.write('Published At: ' + article['publishedAt'] + '\n')
        file.write('URL: ' + article['url'] + '\n\n')
        no_of_articles += 1

print(no_of_articles," articles collected.")

with open('NewsContent.txt', 'w') as file:
    for article in all_articles['articles']:
        file.write('Title: ' + article['title'] + '\n')
        if article['description'] is not None:
            file.write('Description: ' + article['description'] + '\n')
        if article['content'] is not None:
            file.write('Content: ' + article['content'] + '\n')
