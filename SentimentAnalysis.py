import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv("Headline_Trainingdata.csv", sep=',', quotechar='"', header=0, usecols=['text', 'sentiment'])
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', ' ', x)))

    size = 2200
    tokenizer = Tokenizer(num_words=size, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X)

    model = Sequential()
    model.add(Embedding(size, 300, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.5))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    Y = pd.get_dummies(data['sentiment']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=2)

    score, accuracy = model.evaluate(X_test, Y_test, verbose=2)
    print("Accuracy: %.2f" % (accuracy))


if __name__ == "__main__":
    main()