from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import json
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd



stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stop_words and word not in string.punctuation]
    return ' '.join(text) 

# Load data from the text file
with open('data.txt', 'r') as file:
    data = json.load(file)
chatbot_data=pd.DataFrame(data['data'])
chatbot_data=pd.DataFrame(data['data'],index=chatbot_data['Tag'])

X = []
Y = []

for item in data['data']:
    for pattern in item['Patterns']:
        X.append(transform_text(pattern))  
        Y.append(item['Tag'])

chatbot_training = pd.DataFrame({'Patterns': X, 'Tag': Y})

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(chatbot_training['Patterns'])

label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(chatbot_training['Tag'])  

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, Y_train)

import joblib
joblib.dump(classifier, 'classifier.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

