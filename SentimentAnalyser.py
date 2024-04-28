from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from datetime import datetime

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    score = (sentiment_scores['compound'] + 1) * 2.5
    return score



with open('articles.json', 'r') as file:
    data = json.load(file)

current_date = datetime.today()

total_score = 0
relevance_factor = 0.85
freq_sum = 0

for obj in data:
    title = obj['title']
    publishedAt = obj['date']
    description = obj['description']
    content = obj['content']
    date = datetime.strptime(publishedAt[:10], "%Y-%m-%d")
    score = analyze_sentiment(title+description+content) * (relevance_factor ** (current_date - date).days)
    freq = (relevance_factor ** (current_date - date).days)
    freq_sum += freq
    print("Score:",score)
    total_score += score

total_score = (1+total_score)/(1+freq_sum)
print("Total Score:",total_score)
s = {'Score:': total_score}
with open("prediction.json",'a') as json_file:
    json.dump(s,json_file,indent = 4)

