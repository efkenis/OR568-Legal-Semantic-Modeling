import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels


class Data():
    def __init__(self, df):
        self.X = df['sentences']
        self.y = df['rhetrole']

    def traintestsplit(self, prop):
        return train_test_split(self.X, self.y, test_size=prop)


def createTokenizeMatrix(X_train, X_test, tokenizer):
    tokenizer.fit_on_texts(X_train)

    # Convert train and test sets to tokens
    X_train_tokens = tokenizer.texts_to_matrix(X_train, mode='tfidf')
    X_test_tokens = tokenizer.texts_to_matrix(X_test, mode='tfidf')

    return X_train_tokens, X_test_tokens, tokenizer.word_index


def convertLabelToCategorical(y_train, y_test):
    # Convert labels to a one-hot representation
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train_encode = encoder.transform(y_train)
    y_test_encode = encoder.transform(y_test)

    num_classes = np.max(y_train_encode)+1
    y_train_cat = to_categorical(y_train_encode, num_classes)
    y_test_cat = to_categorical(y_test_encode, num_classes)

    return y_train_cat, y_test_cat, y_test_encode, num_classes


def create_model(max_words, num_classes):
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def newPrediction(new_sentence, tokenizer, model, labels):
    seq = tokenizer.texts_to_matrix(new_sentence)
    pred_sent = model.predict(seq)
    pred_class_sent = model.predict_classes(seq)
    print("New sentence: ", new_sentence[0])
    for i in zip(labels, pred_sent.tolist()[0]):
        print(i)
    print("Prediction: ", labels[pred_class_sent])


def main():
    # j_files = [file for file in os.listdir(os.getcwd()) if file.endswith('.json')]
    # df = preprocess(j_files)

    df = pd.read_pickle("BVA_Master0409.pkl")

    data = Data(df)
    X_train, X_test, y_train, y_test = data.traintestsplit(0.2)

    labels = unique_labels(y_test)

    max_words = 2500
    tokenizer = Tokenizer(num_words=max_words)
    X_train_tokens, X_test_tokens, word_index = createTokenizeMatrix(
        X_train, X_test, tokenizer)
    y_train_cat, y_test_cat, y_test_encode, num_classes = convertLabelToCategorical(
        y_train, y_test)

    model = create_model(max_words, num_classes)

    model.fit(X_train_tokens, y_train_cat,
              validation_split=0.125, epochs=8, batch_size=64, verbose=0)
    print("------------------------------------------------------------------")
    score = model.evaluate(X_test_tokens, y_test_cat, batch_size=64, verbose=1)
    print('Test score: ', score[0])
    print('Test accuracy', score[1])
    print("------------------------------------------------------------------")

    pred = model.predict(X_test_tokens)
    pred_class = model.predict_classes(X_test_tokens)

    print(classification_report(y_test_encode, pred_class, target_names=labels))
    print(pd.crosstab(labels[pred_class], y_test,
                      colnames=['Actual'],
                      rownames=["Predicted"],
                      margins=True).to_string())

    # new_sentence = ["The board finds this case to be an issue"]
    # newPrediction(new_sentence, tokenizer, model, labels)


if __name__ == "__main__":
    main()
