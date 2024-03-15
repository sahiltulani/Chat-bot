from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import json
import random
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords

stop_word = set(stopwords.words('english'))
def tranform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    for i in text:
        if not i.isalnum:
            text.remove(i)
        elif i in stop_word:
            text.remove(i)
    for i in text:
        if i in string.punctuation:
            text.remove(i)
    for i in range(len(text)):
        text[i]=ps.stem(text[i])
    return ' '.join(text) 
ps = PorterStemmer()

# Load data from the text file
with open('data.txt', 'r') as file:
    data = json.load(file)
data_sample=pd.DataFrame(data['data'])
X=[]
Y=[]
for item in data['data']:
    for i in item['Patterns']:
        X.append(item['Tag']) 
        Y.append(i)
chatbot_traing=pd.DataFrame({'Patterns':Y,'Tag':X})
chatbot_traing['Tag']=chatbot_traing['Tag'].map({'Name':0,'Yourself':1,'Education':2,'Experience':3,'Projects':4,'Hobbies':5})
print(chatbot_traing)
chatbot_traing['Prcossing']=chatbot_traing['Patterns'].apply(tranform_text)
print(chatbot_traing)
vectorizer = TfidfVectorizer()
X_train=vectorizer.fit_transform(chatbot_traing['Prcossing'])
mlb = MultiLabelBinarizer()
Y_train=chatbot_traing['Tag']
classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
classifier.fit(X_train,Y_train)
user_input='tell us about your project'
user_input=tranform_text(user_input)
predicted_tags = classifier.predict(vectorizer.transform([user_input]))

# Transform back to original tags
# predicted_tags = mlb.inverse_transform(predicted_tags)
print(predicted_tags)