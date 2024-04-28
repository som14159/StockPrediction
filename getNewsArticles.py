from newsapi import NewsApiClient
import json

newsapi = NewsApiClient(api_key='d7f4f06cb623498587f38f5107c19bf5')
all_articles = newsapi.get_everything(q='Apple', language='en', sort_by = 'publishedAt')

no_of_articles = 0

with open('NewsArticles.txt', 'w') as file:
    articles_data = []
    for article in all_articles['articles']:
        article_data = {
            'title': article['title'],
            'date': article['publishedAt'],
            'description': article['description'] if article['description'] is not None else 'N/A',
            'content': article['content'] if article['content'] is not None else 'N/A'
        }
        articles_data.append(article_data)
    file_path = "articles.json"
    with open(file_path, "w") as json_file:
        json.dump(articles_data, json_file, indent=4)

