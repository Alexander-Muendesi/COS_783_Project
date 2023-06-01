import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Uncomment the line below of the first run then you can comment it out
# nltk.download('vader_lexicon')





def analyze_sentiment(sentence):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(sentence)
    
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


sentence = input("Enter a sentence: ")
sentiment = analyze_sentiment(sentence)
print("Sentiment:", sentiment)
