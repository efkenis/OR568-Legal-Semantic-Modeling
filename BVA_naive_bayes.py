import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
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


# Uncomment to show bar chart of rhetorical roles
ax = df['rhetrole'].value_counts().plot(kind='bar', rot=0)
plt.show()


# Let X define the predictors and y define the labels
X = df['sentences']
y = df['rhetrole']


# Split data into training, validation, and test sets ()
#what is the test_size = .20 comming from???
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  # test size ==??
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.125)

# Running logistic regression model using several values for the
# regularizer (which is C)
C_vals = [1, 10, 50, 100, 1000, 2000]
acc = 0
#log_reg = 0
bayes_ = 0
for c in C_vals:
    print(f'value of c: {c}')

    bayes = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', BernoulliNB())# MultinomialNB())

                       ])

    # First you train the model using fit, then you predict
    bayes.fit(X_train, y_train)
    y_pred = bayes.predict(X_val) #compare and with predictied values
    accuracy = accuracy_score(y_val, y_pred)
    #train it for the 1st value of c


    # This conditional looks for the C_val that has the highest accuracy
    # on the validation set and uses it for the final prediction below
    if accuracy > acc:  #if this one better then save it
        acc = accuracy
        #log_reg = logreg
        bayes_ = bayes

    print(f"accuracy: {accuracy}")


# The final prediction on the testing set using the C val from above
y_pred = bayes.predict(X_test)
accuracy_score = accuracy_score(y_test, y_pred)


# Print the classsifcation report (preicsion, recall, etc)
# and the confusion_matrix
print(classification_report(y_test, y_pred,
                            target_names=unique_labels(y_test)))
print(pd.crosstab(y_pred, y_test,
                  colnames=['Actual'],
                  rownames=['Predicted'],
                  margins=True).to_string())

