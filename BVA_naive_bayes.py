import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def clean_text(text):
    '''lower text, remove punctuation, numbers, extra spaces, and stop words'''
    text = text.lower()
    text = re.sub(r'[\.:;,()\']', ' ', text)
    text = re.sub(r'[0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join(word for word in text.split()
                    if word not in stopwords.words('english'))
    return text


# Read in csv from directory
df = pd.read_csv("PTSD_claims_data.csv")


# Apply clean text function from above to do some simple preprocessing
df['sentences'] = df['sentences'].apply(clean_text)
df['rhetrole'] = df['rhetrole'].apply(clean_text)


# Let X define the predictors and y define the labels
X = df['sentences']
y = df['rhetrole']


# Split data into training, validation, and test sets ()
# what is the test_size = .20 comming from???
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20)  # test size ==??
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.125)

bayes = Pipeline([('vect', CountVectorizer()),
                  ('tfidf', TfidfTransformer()),
                  ('clf', MultinomialNB())  # MultinomialNB())
                  ])

# First you train the model using fit, then you predict
bayes.fit(X_train, y_train)
y_pred = bayes.predict(X_val)  # compare and with predictied values
accuracy = accuracy_score(y_val, y_pred)


# The final prediction on the testing set using the C val from above
y_pred = bayes.predict(X_test)
accuracy_score = accuracy_score(y_test, y_pred)


# Print the classsifcation report (precsion, recall, etc)
# and the confusion_matrix
print(classification_report(y_test, y_pred,
                            target_names=unique_labels(y_test)))
print(pd.crosstab(y_pred, y_test,
                  colnames=['Actual'],
                  rownames=['Predicted'],
                  margins=True).to_string())
