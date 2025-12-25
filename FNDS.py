
import pandas as pd
import numpy as np

true = pd.read_csv('True.csv')

fake = pd.read_csv('Fake.csv')

true.head()

fake.head()

true['label']=1

fake['label']=0

true.head()

news = pd.concat([fake, true],axis=0)

news.head()

news.tail()

news.isnull().sum()

news = news.drop(['title','subject','date'],axis=1)

news.head()

news = news.sample(frac=1)

news.head()

news.reset_index(inplace=True)

news.head()

news.drop(['index'],axis=1,inplace=True)

news.head()

import re

def wordopt(text):
  text = text.lower()

  text = re.sub(r'https?://\S+|www\.\S+','',text)

  text = re.sub(r'<.*?>','',text)

  text = re.sub(r'[^\w\s]','',text)

  text = re.sub(r'\d','',text)

  text = re.sub(r'\n','',text)

  return text

news['text'] = news['text'].apply(wordopt)

news['text']

x = news['text']
y = news['label']

x

y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

x_train.shape

x_test.shape

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()

xv_train = vectorization.fit_transform(x_train)

xv_test = vectorization.transform(x_test)

xv_train

xv_test

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

LR.score(xv_test, y_test)

print(classification_report(y_test, pred_lr))

from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()

DTC.fit(xv_train, y_train)

pred_dtc = DTC.predict(xv_test)

DTC.score(xv_test, y_test)

print(classification_report(y_test,pred_dtc))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(xv_train, y_train)

predict_rfc = rfc.predict(xv_test)

rfc.score(xv_test, y_test)

print(classification_report(y_test, predict_rfc))

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(xv_train, y_train)

pred_gbc = gbc.predict(xv_test)

gbc.score(xv_test, y_test)

print(classification_report(y_test, pred_gbc))

def output_label(n):
  if n==0:
    return "It is a Fake News"
  elif n==1:
    return "It is a Genuine News"

def manual_testing(news):
    testing_news = {"text": [news]}

    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)

    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_lr = LR.predict(new_xv_test)
    pred_gbc = gbc.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)

    return (
        "\n\nLR Prediction: {}"
        "\nGBC Prediction: {}"
        "\nRFC Prediction: {}".format(
            output_label(pred_lr[0]),
            output_label(pred_gbc[0]),
            output_label(pred_rfc[0])
        )
    )

news_article = str(input())

manual_testing(news_article)

