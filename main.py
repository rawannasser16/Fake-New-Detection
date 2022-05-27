import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression

# Global Variable
months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
          'december']
contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have"
}
stopwords = nltk.corpus.stopwords.words('english')
punctuation = '''()!"#$%&'*+,-./:;<=>?@[\]^_`{|}~â”€'''
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
# Regular expression for finding contractions
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
words = set(nltk.corpus.words.words())
le = LabelEncoder()


# to remove HTML tag
def html_remover(data):
    beauti = BeautifulSoup(data, 'html.parser')
    return beauti.get_text()


# to remove URL
def url_remover(data):
    return re.sub(r'https\S', '', data)


def web_associated(data):
    text = html_remover(data)
    text = url_remover(text)
    return text


def expand_contractions(text):
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, text)


def remove_punctuation(text):
    punctuationFree = ""
    for char in text:
        if char in punctuation:
            continue
        else:
            punctuationFree = punctuationFree + char
    return punctuationFree


def remove_non_ascii(s):
    return "".join(c for c in s if ord(c) < 128)


def remove_numbers(text):
    OutputWithoutNumbers = re.sub(r'\d', '', text)
    return OutputWithoutNumbers.strip()


def remove_months(text):
    textTemp = text
    textTemp = textTemp.split()
    for word in textTemp:
        if word in months:
            textTemp.remove(word)
    TextWithoutMonths = ' '.join(textTemp)
    return TextWithoutMonths


# Remove None English Word
def remove_non_english_word(text):
    EnglishText = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
    return EnglishText


def RemoveEnglishAlphabetLettersSoloInString(textList):
    TextList = textList.copy()
    for word in textList:
        if len(word) == 1:
            TextList.remove(word)
    return TextList


def tokenization(text):
    nltk_tokens_word = nltk.word_tokenize(text)
    return nltk_tokens_word


# defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    textWithoutStopWord = [i for i in text if i not in stopwords]
    return textWithoutStopWord


# defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text


def word_count(text):
    WordsCount = dict()
    TempSent = ' '.join(text)
    for word in TempSent.split():
        if word in WordsCount:
            WordsCount[word] += 1
        else:
            WordsCount[word] = 1

    return WordsCount


def uniqueWords(documents):
    words_set1 = set()
    for document in documents:
        words_set1 = words_set1.union(set(document))
    return words_set1


def ComputeTermFrequency(dataOfDoc, WordsSetUnique):
    NumberDoc = len(dataOfDoc)
    n_words_set = len(WordsSetUnique)  # Number of unique words in the
    UniqueWordDF = pd.DataFrame(np.zeros((NumberDoc, n_words_set)), columns=WordsSetUnique)
    # Compute Term Frequency (TF)
    index = 0
    for doc in dataOfDoc:
        for w in doc:
            UniqueWordDF[w][index] = UniqueWordDF[w][index] + (1 / len(doc))
        index += 1
    return UniqueWordDF


if __name__ == '__main__':
    NewsData = pd.read_csv("fake_or_real_news.csv")

    # Cleaning
    NewsData.drop("ID", axis=1, inplace=True)
    NewsData.dropna(inplace=True)
    NewsData.drop_duplicates(inplace=True)

    # Remove HTML TAGS AND URL
    NewsData['text'] = NewsData['text'].apply(lambda x: web_associated(x))

    # remove punctuation '!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'
    NewsData['text'] = NewsData['text'].apply(lambda x: remove_punctuation(x))

    # â€” or â€™ or œ
    NewsData['text'] = NewsData['text'].apply(lambda x: remove_non_ascii(x))

    # remove Lower Text
    NewsData['text'] = NewsData['text'].apply(lambda x: x.lower())

    # remove Numbers Digit
    NewsData['text'] = NewsData['text'].apply(lambda x: remove_numbers(x))

    # remove Months
    NewsData['text'] = NewsData['text'].apply(lambda x: remove_months(x))

    # remove None English word
    NewsData['text'] = NewsData['text'].apply(lambda x: remove_non_english_word(x))

    # Expanding Contractions in the reviews
    NewsData['text'] = NewsData['text'].apply(lambda x: expand_contractions(x))

    # applying function to the column
    NewsData['text'] = NewsData['text'].apply(lambda x: tokenization(x))

    # Remove English Letter
    NewsData['text'] = NewsData['text'].apply(lambda x: RemoveEnglishAlphabetLettersSoloInString(x))

    # remove stop words
    NewsData['text'] = NewsData['text'].apply(lambda x: remove_stopwords(x))

    # lemmatizer
    NewsData['text'] = NewsData['text'].apply(lambda x: lemmatizer(x))

    # Remove English Letter
    NewsData['text'] = NewsData['text'].apply(lambda x: RemoveEnglishAlphabetLettersSoloInString(x))

    # Label Encoder For Categorical Sets
    NewsData.loc[:, "label"] = le.fit_transform(NewsData.loc[:, "label"])

    # Retrieve Count Words In Documents
    NewsData['word_count'] = NewsData['text'].apply(lambda x: word_count(x))

    # Retrieve Unique Words
    words_set = uniqueWords(NewsData['text'])
    # print(words_set)

    # Csv File Contain All Cleaning-Preprocessing Data and Count-Words in Doc
    NewsData.to_csv("Cleaning-Preprocessing Data.csv", encoding='utf-8', header=True, index=False)

    newsText = NewsData['text']
    tag = NewsData['label']

    train_size = int(len(NewsData) * .7)

    train_text = NewsData['text'][:train_size]
    train_label = NewsData['label'][:train_size]

    test_text = NewsData['text'][train_size:]
    test_label = NewsData['label'][train_size:]

    tokenizer = Tokenizer(num_words=None, lower=False)
    tokenizer.fit_on_texts(newsText)

    x_train = tokenizer.texts_to_matrix(train_text, mode='tfidf')
    x_test = tokenizer.texts_to_matrix(test_text, mode='tfidf')

    num_of_classes = int((len(set(NewsData['label']))))

    # SVM MODEL
    SVMmod = svm.SVC(kernel='sigmoid')
    SVMmod.fit(x_train, train_label)
    y_pred_SVM = SVMmod.predict(x_test)
    print(f"Accuracy of our Svm Model: {round(metrics.accuracy_score(test_label, y_pred_SVM) * 100)}%")
    print("*" * 90)

    # SVM Model SVC Kernel Linear
    SVMLinear = svm.LinearSVC()
    SVMLinear.fit(x_train, train_label)
    y_pred_SVM_Linear = SVMLinear.predict(x_test)
    print(f"Accuracy of our Svm Linear Model: {round(metrics.accuracy_score(test_label, y_pred_SVM_Linear) * 100)}%")
    print("*" * 90)

    # SVM Model SVC Kernel Non Linear
    SVMNonLinear = svm.SVC(kernel='poly')
    SVMNonLinear.fit(x_train, train_label)
    y_pred_SVM_ploy = SVMNonLinear.predict(x_test)
    print(f"Accuracy of our Svm Non Linear Model: {round(metrics.accuracy_score(test_label, y_pred_SVM_ploy) * 100)}%")
    print("*" * 90)

    # naive base model
    gnb = GaussianNB()
    y_pred_gnb = gnb.fit(x_train, train_label).predict(x_test)
    print(f"Accuracy of our naive_bayes model: {round(metrics.accuracy_score(test_label, y_pred_gnb) * 100)}%")
    print("*" * 90)

    # KNN MODEL
    KNN = KNeighborsClassifier()
    y_pred_KNN = KNN.fit(x_train, train_label).predict(x_test)
    print(f"Accuracy of our KNN model: {round(metrics.accuracy_score(test_label, y_pred_KNN) * 100)}%")
    print("*" * 90)

    # Decision Tree Model
    decision_tree = tree.DecisionTreeClassifier(criterion='gini')
    y_pred_tree = decision_tree.fit(x_train, train_label).predict(x_test)
    print(f"Accuracy of our Decision Tree Model: {round(metrics.accuracy_score(test_label, y_pred_tree) * 100)}%")
    print("*" * 90)

    # LOGISTIC REGRESSION
    Logistic = LogisticRegression()
    Logistic.fit(x_train, train_label)
    Accuracy = Logistic.score(x_test, test_label)
    print(f"Accuracy of our Logistic Regression model: {round(Accuracy * 100)}%")
