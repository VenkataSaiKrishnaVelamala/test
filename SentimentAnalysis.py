# coding: utf-8
from mrjob.job import MRJob
from mrjob.step import MRStep
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
sentiAna=SentimentIntensityAnalyzer()
import re

class TwitterAnalysis(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_text_eachlin,
                   reducer=self.reducer_clean),
            MRStep(mapper=self.mapper_sentiment_analysis,
                   reducer=self.reducer_agg_sentiment),
            MRStep(reducer=self.reducer_calc_percent)
                        
        ]
    
    def tokenize(self,text):
        tokens=word_tokenize(text)
        return tokens
    
    def analyze_sentiment(tweet):
        score=sentiAna.polarity_scores(tweet)
        compound=score['compound']
        if compound >= 0.05:
            return "positive"
        elif compound<= -0.05:
            return "negative"
        else:
            return "neutral"

    def cleantext(self,text):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.lower().strip()
    
    
    def mapper_text_eachlin(self, _, tweet):
        yield "token", tweet

    def reducer_clean(self,key,text):
        cleaned=self.cleantext(text)
        yield None,cleaned

    
    def mapper_sentiment_analysis(self,_, text):
        sentiment=self.analyze_sentiment(text)
        yield sentiment, 1

    def reducer_agg_sentiment(self,key,values):
        yield key,sum(values)

    def reducer_calc_percent(self,key,value):
        totaltweets=sum(value)
        for sentiment, count in zip(key,value):
            percent=(count/totaltweets)*100
            yield sentiment, percent
       

if __name__ == '__main__':
    TwitterAnalysis.run()
