import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score
import os

data = pd.read_csv("train.csv", sep=",")
#print(data)
#data = data.drop_duplicates(subset=['Claim'])

label_column = "Label"
features_columns = [x for x in data.columns if (x != label_column and x != 'Fact-checked Article')]

data['Claim'] = data['Claim'].str.lower()
data['Claim'] = data['Claim'].str.replace("covid-19" , "covid")
data['Claim'] = data['Claim'].str.replace("covid 19" , "covid")
data['Claim'] = data['Claim'].str.replace("coronavirus" , "covid")
data['Claim'] = data['Claim'].str.replace("corona" , "covid")

features = data[features_columns]
labels = data[label_column]
#print(features , labels)

features_array = features.values #numpy array with country,review date, claim, and person attributes
label_array = labels.values #fact label

print(features_array.shape)

labels = data.Label

test_data = pd.read_csv("test.csv", sep=",")
test_data['Claim'] = test_data['Claim'].str.lower()
test_data['Claim'] = test_data['Claim'].str.replace("covid-19" , "covid")
test_data['Claim'] = test_data['Claim'].str.replace("covid 19" , "covid")
test_data['Claim'] = test_data['Claim'].str.replace("coronavirus" , "covid")
test_data['Claim'] = test_data['Claim'].str.replace("corona" , "covid")

test_features = test_data.Claim

"""
train_features = data[['Country (mentioned)',
                  'Review Date',
                  'Claim',
                  'Source']]

test_features = data[['Country (mentioned)',
                  'Review Date',
                  'Claim',
                  'Source']]
"""


#create vocabulary of most frequent words in training data claims
tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', min_df = 1, max_df = .8, max_features = 6500,
encoding='latin-1', ngram_range=(1, 2), stop_words='english')

#encode source data
encoder = OneHotEncoder(categories='auto', handle_unknown='ignore', dtype='int')
train_source = data.Source
train_source = train_source.values.reshape(-1,1)
train_source_encoded = encoder.fit_transform(train_source)

test_source = test_data.Source
test_source = test_source.values.reshape(-1,1)
test_source_encoded = encoder.transform(test_source)




#encode country data
enc = OneHotEncoder(categories='auto', handle_unknown='ignore', dtype='int')
train_country = data['Country (mentioned)']
train_country = train_country.values.reshape(-1,1)
train_country_encoded = enc.fit_transform(train_country)

test_country = test_data['Country (mentioned)']
test_country = test_country.values.reshape(-1,1)
test_country_encoded = enc.transform(test_country)

train_claim = tfidf.fit_transform(data.Claim) 
test_claim = tfidf.fit_transform(test_data.Claim)

"""
#get list of tfidf vocabulary
#print(tfidf.get_feature_names())

col = ['feat_' + i for i in tfidf.get_feature_names()]
temp = pd.DataFrame(train_claim.todense(), columns=col)
print(temp)
temp.to_csv('testy.csv', index=False)

"""

#combine all collumn-data using hstack
train_features = hstack([train_source_encoded, train_country_encoded ,train_claim])
test_features = hstack([test_source_encoded, test_country_encoded ,test_claim])
print(test_features.shape)

#model = LinearSVC(class_weight='balanced')
model = LinearSVC() #best model that not just zeros 
#model = PassiveAggressiveClassifier(max_iter = 50)
#model = RandomForestClassifier() 


model.fit(train_features, labels)
y_pred = model.predict(test_features)

np.set_printoptions(threshold=np.inf)
print(y_pred)

#Create csv file with output
output_id = list(range(1,711))
#print(len(output_id) , len(y_pred))
dataset = pd.DataFrame({'Id':output_id , 'Predicted': y_pred})
dataset.to_csv('out.csv', index=False)
