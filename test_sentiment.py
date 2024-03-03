# coding: utf-8
import nltk
from mrjob.job import MRJob
from mrjob.step import MRStep
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import re
sentiAna=SentimentIntensityAnalyzer()

nltk.download('vader_lexicon')


class TwitterTest(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_tweets,
                   reducer=self.reducer_tweets),
            MRStep(reducer=self.reducer_percent_tweets)
             
        ]

    
    def clean_tweets(self,tweet):
        pattern = r',-?\d+$'# Remove the matched pattern from the text
        modified_text = re.sub(pattern, '', tweet)
        return modified_text

    def sentiment_analyzer(self,tweet):
        sentiment=sentiAna.polarity_scores(tweet)
        compound=sentiment['compound']
        if compound >= 0.05:
            return "positive"
        elif compound<= -0.05:
            return "negative"
        else:
            return "neutral"
    

    def mapper_tweets(self,key,tweets):
        tweets=self.clean_tweets(tweets)
        sentiment=self.sentiment_analyzer(tweets)
        yield sentiment , 1

    def reducer_tweets(self,key,value):
        yield "total counts",(key, sum(value))

    def reducer_percent_tweets(self,key,values):
        sentiment_scores=dict(values)
        total_tweets=sum(sentiment_scores.values())
        for sentiment,count in sentiment_scores.items():
            percent=(count/total_tweets)*100
            yield sentiment,round(percent,ndigits=2)
        

if __name__ == '__main__':
    TwitterTest.run()
